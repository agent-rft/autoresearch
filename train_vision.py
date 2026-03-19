"""
Vision-Language training on space-multimodal-dataset.
Uses custom lightweight vision encoders (no external deps needed).
Architecture: Vision Encoder + LLM with multimodal fusion.
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import io
from dataclasses import dataclass
from PIL import Image
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.transforms as T

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("No torchvision - using PIL transforms only")

from prepare import Tokenizer

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

IMG_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


@dataclass
class VLConfig:
    vocab_size: int = 32768
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    vision_embed_dim: int = 256
    context_len: int = 128
    img_size: int = IMG_SIZE
    vision_type: str = "cnn"


class LightweightTokenizer:
    def __init__(self, base_tokenizer):
        self.base = base_tokenizer
        self.img_token = 2

    def encode(self, text):
        ids = self.base.encode(text)
        return ids

    def decode(self, ids):
        return self.base.decode(ids)


def load_space_dataset():
    """Load space multimodal dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        ds = load_dataset("AIOmarRehan/space-multimodal-dataset", split="train")
        print(f"Loaded dataset: {len(ds)} samples")
        return ds
    except ImportError:
        print("datasets package not available, using manual load")
        return None


def pil_loader(url_or_path):
    if url_or_path.startswith("http"):
        response = requests.get(url_or_path, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(url_or_path).convert("RGB")


class SpaceVLMDataset(Dataset):
    def __init__(self, split="train", img_size=224):
        self.img_size = img_size
        self.split = split

        try:
            from datasets import load_dataset

            self.ds = load_dataset("AIOmarRehan/space-multimodal-dataset", split=split)
            self.use_hf = True
        except:
            self.use_hf = False
            print("Using mock dataset (datasets package unavailable)")
            self.ds = list(range(250))

        if HAS_TORCHVISION:
            self.transform = T.Compose(
                [
                    T.Resize((img_size, img_size)),
                    T.ToTensor(),
                    T.Normalize(IMAGE_MEAN, IMAGE_STD),
                ]
            )
        else:
            self.transform = None

    def _process_image_pil(self, img):
        if img is None:
            return torch.zeros(3, self.img_size, self.img_size)
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(io.BytesIO(img)).convert("RGB")
            except:
                return torch.zeros(3, self.img_size, self.img_size)
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.transform:
            return self.transform(img)
        img_arr = torch.tensor(
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        )
        img_arr = (
            img_arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0
        )
        for c in range(3):
            img_arr[c] = (img_arr[c] - IMAGE_MEAN[c]) / IMAGE_STD[c]
        return img_arr

    def __len__(self):
        if self.use_hf:
            return len(self.ds)
        return 250

    def __getitem__(self, idx):
        if self.use_hf:
            row = self.ds[idx]
            img = row["image"]
            text = row["text"]
        else:
            img = None
            text = f"Satellite image of space showing celestial bodies and atmospheric phenomena. Description: Sample {idx} of space observation data with detailed analysis."

        img_tensor = self._process_image_pil(img)
        return img_tensor, text


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + self.shortcut(x)


class CNNEncoder(nn.Module):
    """Lightweight CNN vision encoder."""

    def __init__(self, embed_dim=256, img_size=224):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size

        channels = [3, 64, 128, 256, 512]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                ConvBlock(channels[i], channels[i + 1], stride=2 if i < 3 else 1)
            )

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.proj = nn.Sequential(
            nn.Linear(512 * 16, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 17, embed_dim) * 0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(B, 512, 16).transpose(1, 2)
        x = self.proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        return x


class SimpleViTEncoder(nn.Module):
    """Simple custom ViT encoder."""

    def __init__(self, embed_dim=256, img_size=224, patch_size=16, depth=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size, bias=False)
        self.norm_pre = nn.LayerNorm(embed_dim)

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=4,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_pre(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class HybridEncoder(nn.Module):
    """CNN features + lightweight transformer."""

    def __init__(self, embed_dim=256, depth=2):
        super().__init__()
        channels = [3, 64, 128, 256]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(ConvBlock(channels[i], channels[i + 1], stride=2))
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=depth,
        )

        self.proj = nn.Linear(256, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 50, embed_dim) * 0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        max_len = min(x.size(1), self.pos_embed.size(1))
        x[:, :max_len] = x[:, :max_len] + self.pos_embed[:, :max_len]

        x = self.trans(x)
        return x


class ResNetStyleEncoder(nn.Module):
    """Simple ResNet-style encoder."""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, embed_dim, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 17, embed_dim) * 0.02)

        self.proj = nn.Identity() if embed_dim == 256 else nn.Linear(256, embed_dim)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ConvBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        return x


def build_vision_encoder(vision_type, embed_dim, img_size):
    if vision_type == "cnn":
        return CNNEncoder(embed_dim, img_size)
    elif vision_type == "vit":
        return SimpleViTEncoder(embed_dim, img_size, patch_size=16, depth=4)
    elif vision_type == "hybrid":
        return HybridEncoder(embed_dim, depth=2)
    elif vision_type == "resnet":
        return ResNetStyleEncoder(embed_dim)
    else:
        raise ValueError(f"Unknown vision type: {vision_type}")


class VLMModel(nn.Module):
    """Vision-Language Model: Vision Encoder + LLM."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vision_encoder = build_vision_encoder(
            config.vision_type, config.vision_embed_dim, config.img_size
        )

        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.context_len = config.context_len

        self.img_token_id = config.vocab_size - 1

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.image_projection = nn.Linear(config.vision_embed_dim, config.n_embd)

        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.n_layer)]
        )
        self.norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.x0_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for p in self.parameters():
            if p.dim() > 1 and not isinstance(p, nn.Embedding):
                torch.nn.init.xavier_uniform_(p)

    def forward(self, images, input_ids, targets=None):
        B, T = input_ids.size()

        img_features = self.vision_encoder(images)
        img_embeds = self.image_projection(img_features)

        tok_embeds = self.token_embedding(input_ids)

        x = tok_embeds
        x0 = x

        for layer in self.layers:
            x = layer(x, img_embeds)
            x = x + self.x0_scale * x0

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1
            )
            return loss
        return logits

    def generate(self, images, tokenizer, max_new_tokens=50, temperature=1.0):
        self.eval()
        B = images.size(0)

        prompt = "Describe this space image:"
        input_ids = tokenizer.encode(prompt)
        input_ids = [
            tokenizer.base.eod_id
            if isinstance(tokenizer.base.eod_id, int)
            and tokenizer.base.eod_id is not None
            else 3
        ] * (len(input_ids) // 4)

        input_ids = torch.tensor([input_ids], device=device).expand(B, -1)

        for _ in range(max_new_tokens):
            logits = self.forward(images, input_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if (
                next_token.item() == tokenizer.base.eod_id
                if hasattr(tokenizer.base, "eod_id")
                else 0
            ):
                break

        return input_ids


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True, bias=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
        )
        self.norm1 = nn.RMSNorm(config.n_embd)
        self.norm2 = nn.RMSNorm(config.n_embd)
        self.img_norm = nn.RMSNorm(config.n_embd)

    def forward(self, x, img_embeds):
        img_ctx = self.img_norm(img_embeds)
        x = x + self.attn(self.norm1(x), img_ctx, img_ctx)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def setup_training():
    config = VLConfig(
        vocab_size=8192,
        n_layer=4,
        n_head=4,
        n_embd=256,
        vision_embed_dim=256,
        context_len=128,
        img_size=224,
        vision_type="cnn",
    )

    tokenizer = Tokenizer.from_directory()
    vl_tokenizer = LightweightTokenizer(tokenizer)

    train_ds = SpaceVLMDataset(split="train", img_size=config.img_size)
    val_ds = SpaceVLMDataset(split="train", img_size=config.img_size)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = VLMModel(config).to(device)

    nparams = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {nparams:,} total, {trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )

    return model, optimizer, train_loader, val_loader, vl_tokenizer, config


def train_epoch(model, loader, optimizer, tokenizer, epoch, config):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (images, texts) in enumerate(loader):
        images = images.to(device)

        input_ids = []
        targets = []
        max_len = 0

        for text in texts:
            ids = tokenizer.encode(text)
            input_ids.append([tokenizer.base.eod_id] + ids + [tokenizer.base.eod_id])
            max_len = max(max_len, len(input_ids[-1]))

        for i in range(len(input_ids)):
            while len(input_ids[i]) < max_len:
                input_ids[i].append(-1)

        input_ids = torch.tensor(input_ids, device=device)
        tgt = input_ids.clone()
        tgt[input_ids == tokenizer.base.eod_id] = -1

        logits = model(images, input_ids[:, :-1], tgt[:, 1:])

        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            tgt[:, 1:].reshape(-1),
            ignore_index=-1,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}"
            )

    return total_loss / n_batches


def evaluate(model, loader, tokenizer, config):
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for images, texts in loader:
            images = images.to(device)

            input_ids = []
            targets = []
            max_len = 0

            for text in texts:
                ids = tokenizer.encode(text)
                input_ids.append(
                    [tokenizer.base.eod_id] + ids + [tokenizer.base.eod_id]
                )
                max_len = max(max_len, len(input_ids[-1]))

            for i in range(len(input_ids)):
                while len(input_ids[i]) < max_len:
                    input_ids[i].append(-1)

            input_ids = torch.tensor(input_ids, device=device)
            tgt = input_ids.clone()
            tgt[input_ids == tokenizer.base.eod_id] = -1

            logits = model(images, input_ids[:, :-1], tgt[:, 1:])

            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                tgt[:, 1:].reshape(-1),
                ignore_index=-1,
            )

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def save_checkpoint(model, optimizer, epoch, loss, config, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": asdict(config)
        if hasattr(asdict, "__call__")
        else {"vision_type": config.vision_type},
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def asdict(obj):
    result = {}
    for k, v in obj.__dict__.items():
        if not k.startswith("_"):
            result[k] = v
    return result


def main():
    print("=" * 60)
    print("Vision-Language Training on Space Multimodal Dataset")
    print("=" * 60)

    model, optimizer, train_loader, val_loader, tokenizer, config = setup_training()

    save_dir = os.path.expanduser("~/.cache/autoresearch/vision_checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 20
    save_every = max(1, num_epochs // 5)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, tokenizer, epoch, config
        )
        val_loss = evaluate(model, val_loader, tokenizer, config)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(save_dir, f"best_vlm_{config.vision_type}.pt")
            save_checkpoint(model, optimizer, epoch, best_loss, config, save_path)

        if epoch % save_every == 0 or epoch == num_epochs - 1:
            save_path = os.path.join(
                save_dir, f"checkpoint_epoch{epoch:02d}_{config.vision_type}.pt"
            )
            save_checkpoint(model, optimizer, epoch, val_loss, config, save_path)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining complete! Best val_loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print("Files:")
    for f in os.listdir(save_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
