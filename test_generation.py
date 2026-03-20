"""
Test generation on trained VLM checkpoints.
Tests space images, pets images, math images, and text-only samples.
"""

import os
import io

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

IMG_SIZE = 224


class SpaceVLMDataset(Dataset):
    def __init__(self, split="train", img_size=IMG_SIZE, max_samples=50):
        self.img_size = img_size
        from datasets import load_dataset

        self.ds = load_dataset("AIOmarRehan/space-multimodal-dataset", split=split)
        if max_samples:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

    def _norm(self, img):
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(io.BytesIO(img)).convert("RGB")
            except:
                return torch.zeros(3, self.img_size, self.img_size)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        np_img = np.array(img)
        a = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (a - m) / s

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row.get("image") or row.get("image_path")
        return self._norm(img), row["text"]


class OxfordPetsDataset(Dataset):
    def __init__(self, split="train", img_size=IMG_SIZE, max_samples=50):
        self.img_size = img_size
        from datasets import load_dataset

        self.ds = load_dataset("enterprise-explorers/oxford-pets", split=split)
        if max_samples:
            np.random.seed(42)
            idxs = np.random.permutation(len(self.ds))[:max_samples]
            self.ds = self.ds.select(idxs)

    def _norm(self, img):
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(io.BytesIO(img)).convert("RGB")
            except:
                return torch.zeros(3, self.img_size, self.img_size)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        np_img = np.array(img)
        a = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (a - m) / s

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        dog = row["dog"]
        label = row["label"]
        gt = f"A {label} {'dog' if dog else 'cat'}."
        return self._norm(row["image"]), gt


class MathVisionDataset(Dataset):
    def __init__(self, split="test", img_size=IMG_SIZE, max_samples=30):
        self.img_size = img_size
        from datasets import load_dataset

        self.ds = load_dataset("macabdul9/MathVision", split=split)
        if max_samples:
            np.random.seed(42)
            idxs = np.random.permutation(len(self.ds))[:max_samples]
            self.ds = self.ds.select(idxs)

    def _norm(self, img):
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(io.BytesIO(img)).convert("RGB")
            except:
                return torch.zeros(3, self.img_size, self.img_size)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        np_img = np.array(img)
        a = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (a - m) / s

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        q = row["question"].replace("<image1>", "").strip()
        a = str(row["answer"]) if row["answer"] else ""
        gt = f"Q: {q} A: {a}"
        img = row.get("decoded_image") or row.get("image")
        return self._norm(img), gt


class FineWebTextDataset(Dataset):
    def __init__(self, split="train", max_samples=30):
        from datasets import load_dataset

        self.ds = load_dataset(
            "Lambent/creative-writing-2048-fineweb-edu-sample", split=split
        )
        if max_samples:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return None, self.ds[idx]["text"]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + self.shortcut(x)


class ResNetStyleEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, embed_dim, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.num_patches = 16
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls, std=0.02)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ConvBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1)


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=256, img_size=224):
        super().__init__()
        ch = [3, 64, 128, 256, 512]
        layers = [ConvBlock(ch[i], ch[i + 1], stride=2) for i in range(4)]
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(nn.Linear(512, embed_dim))
        self.num_patches = 16
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1)


def build_encoder(vision_type, embed_dim, img_size=224):
    if vision_type == "resnet":
        return ResNetStyleEncoder(embed_dim)
    elif vision_type == "cnn":
        return CNNEncoder(embed_dim, img_size)
    else:
        return ResNetStyleEncoder(embed_dim)


class VLConfig:
    def __init__(
        self,
        vocab_size=8192,
        n_layer=4,
        n_head=4,
        n_embd=256,
        vision_embed_dim=256,
        context_len=128,
        img_size=224,
        vision_type="resnet",
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.vision_embed_dim = vision_embed_dim
        self.context_len = context_len
        self.img_size = img_size
        self.vision_type = vision_type


class VLMModel(nn.Module):
    def __init__(self, config, use_img_tokens=True):
        super().__init__()
        self.cfg = config
        self.use_img_tokens = use_img_tokens
        self.vision_encoder = build_encoder(
            config.vision_type, config.vision_embed_dim, config.img_size
        )
        self.num_img_tokens = self.vision_encoder.num_patches + 1
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.img_proj = nn.Linear(config.vision_embed_dim, config.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.context_len + self.num_img_tokens, config.n_embd)
        )
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.x0_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, images, token_ids, labels=None):
        B, T = token_ids.size()
        img_emb = None
        if images is not None and self.use_img_tokens:
            img_feats = self.vision_encoder(images)
            img_emb = self.img_proj(img_feats)
        tok_emb = self.token_emb(token_ids)
        if img_emb is not None:
            x = torch.cat([img_emb, tok_emb], dim=1)
            pos = self.pos_emb[:, : self.num_img_tokens + T]
            x = x + pos
            logits = self.lm_head(self._tfwd(x))[:, self.num_img_tokens :, :]
        else:
            x = tok_emb + self.pos_emb[:, :T]
            logits = self.lm_head(self._tfwd(x))
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-1,
            )
            return loss
        return logits

    def _tfwd(self, x):
        x0 = x
        for blk in self.blocks:
            x = blk(x)
            x = x + self.x0_scale * x0
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_heads = max(1, cfg.n_embd // 64)
        self.attn = nn.MultiheadAttention(
            cfg.n_embd, n_heads, batch_first=True, bias=False, dropout=0.0
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd * 4),
            nn.SiLU(),
            nn.Linear(cfg.n_embd * 4, cfg.n_embd),
        )
        self.norm1 = nn.RMSNorm(cfg.n_embd)
        self.norm2 = nn.RMSNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VLTokenizer:
    def __init__(self, base_tok):
        self.base = base_tok
        self.bos_id = base_tok.get_bos_token_id()
        self.pad_id = 0

    def encode(self, text, max_len=None):
        ids = self.base.encode(text)
        ids = [self.bos_id] + ids
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        return self.base.decode(
            [i for i in ids if i != self.pad_id and i != self.bos_id]
        )


def generate(model, tokenizer, img, prompt, max_new=60, temp=0.7):
    model.eval()
    if img is not None:
        img = img.unsqueeze(0).to(device)
    ids = tokenizer.encode(prompt)
    ids = ids[:120]
    with torch.no_grad():
        for _ in range(max_new):
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            logits = model(img, input_ids)
            logits = logits[0, -1, :] / temp
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id == tokenizer.pad_id or len(ids) > 127:
                break
            ids.append(next_id)
    return tokenizer.decode(ids)


def test_checkpoint(ckpt_path, dataset, prompt_fn, n_samples=3, max_new=50, temp=0.7):
    from prepare import Tokenizer as BaseTok

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = VLConfig(**cfg_dict)

    use_img = dataset[0][0] is not None
    model = VLMModel(cfg, use_img_tokens=use_img).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = VLTokenizer(BaseTok.from_directory())

    results = []
    for i in range(min(n_samples, len(dataset))):
        img, gt = dataset[i]
        prompt = prompt_fn(gt)
        img_dev = img.unsqueeze(0).to(device) if img is not None else None
        gen = generate(model, tokenizer, img_dev, prompt, max_new, temp)
        results.append({"gt": gt, "gen": gen})

    del model
    torch.cuda.empty_cache()
    return results


def main():
    from prepare import Tokenizer as BaseTok

    tok = VLTokenizer(BaseTok.from_directory())

    print("=" * 70)
    print("GENERATION TESTS")
    print("=" * 70)

    # Load datasets
    space_ds = SpaceVLMDataset(max_samples=50)
    pets_ds = OxfordPetsDataset(max_samples=50)
    math_ds = MathVisionDataset(max_samples=30)
    fw_ds = FineWebTextDataset(max_samples=30)

    def fmt_prompt_space(gt):
        words = gt.split()
        return " ".join(words[:7])

    def fmt_prompt_pets(gt):
        words = gt.split()
        return " ".join(words[:3])

    def fmt_prompt_math(gt):
        q = gt.rsplit(" A:", 1)[0]
        return q[:80] + " A:"

    def fmt_prompt_fw(text):
        return text[:30]

    # =====================================================================
    # Experiment 1: Freeze vs Finetune
    # =====================================================================
    print("\n### Freeze vs Finetune (ResNet on Space) ###")
    print()

    ft_ckpt = os.path.expanduser(
        "~/.cache/autoresearch/vision_checkpoints_ft/best_finetune_resnet.pt"
    )
    fr_ckpt = os.path.expanduser(
        "~/.cache/autoresearch/vision_checkpoints_ft/best_frozen_resnet.pt"
    )

    print("**Finetuned (loss=0.8522):**")
    for i in [0, 5, 10]:
        img, gt = space_ds[i]
        prompt = fmt_prompt_space(gt)
        gen = generate_from_ckpt(ft_ckpt, tok, img, prompt)
        print(f"  [{i}] GT: {gt}")
        print(f"      GEN: {gen}")
        print()

    print("**Frozen (loss=0.8873):**")
    for i in [0, 5, 10]:
        img, gt = space_ds[i]
        prompt = fmt_prompt_space(gt)
        gen = generate_from_ckpt(fr_ckpt, tok, img, prompt)
        print(f"  [{i}] GT: {gt}")
        print(f"      GEN: {gen}")
        print()

    # =====================================================================
    # Experiment 2: HP Sweep
    # =====================================================================
    print("\n### HP Sweep (Space, ResNet) ###")
    print()

    hp_ckpts = {
        "lr5e-4 (0.0125)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_hp/best_resnet_ft_lr5.pt"
        ),
        "lr3e-4 (0.0628)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_hp/best_resnet_ft_lr3.pt"
        ),
        "40ep (0.1318)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_hp/best_resnet_ft_long40.pt"
        ),
        "big-6L384 (0.2021)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_hp/best_resnet_ft_big.pt"
        ),
        "cnn-big (0.2505)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_hp/best_cnn_ft_big.pt"
        ),
        "deep8 (0.8154)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_hp/best_resnet_ft_deep8.pt"
        ),
    }

    for name, ckpt_path in hp_ckpts.items():
        print(f"**{name}:**")
        for i in [0, 2, 4]:
            img, gt = space_ds[i]
            prompt = fmt_prompt_space(gt)
            gen = generate_from_ckpt(ckpt_path, tok, img, prompt)
            print(f"  [{i}] GT: {gt}")
            print(f"      GEN: {gen}")
        print()

    # =====================================================================
    # Experiment 3: Mixed Training
    # =====================================================================
    print("\n### Mixed Training ###")
    print()

    mixed_ckpts = {
        "balanced (0.1107)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_mixed/best_mixed_balanced.pt"
        ),
        "vision_focus (0.1745)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_mixed/best_mixed_vision_focus.pt"
        ),
        "natural (0.2260)": os.path.expanduser(
            "~/.cache/autoresearch/vision_checkpoints_mixed/best_mixed_natural.pt"
        ),
    }

    for name, ckpt_path in mixed_ckpts.items():
        print(f"**{name}:**")
        # Space
        print("  [Space]")
        for i in [0, 1]:
            img, gt = space_ds[i]
            prompt = fmt_prompt_space(gt)
            gen = generate_from_ckpt_mixed(ckpt_path, tok, img, prompt)
            print(f"    [{i}] GT: {gt}")
            print(f"        GEN: {gen}")
        # Pets
        print("  [Pets]")
        for i in [0, 1]:
            img, gt = pets_ds[i]
            prompt = fmt_prompt_pets(gt)
            gen = generate_from_ckpt_mixed(ckpt_path, tok, img, prompt)
            print(f"    [{i}] GT: {gt}")
            print(f"        GEN: {gen}")
        # Math
        print("  [Math]")
        for i in [0, 1]:
            img, gt = math_ds[i]
            prompt = fmt_prompt_math(gt)
            gen = generate_from_ckpt_mixed(ckpt_path, tok, img, prompt)
            print(f"    [{i}] GT: {gt[:80]}")
            print(f"        GEN: {gen}")
        # Text
        print("  [FineWeb text]")
        for i in [0, 1]:
            _, gt = fw_ds[i]
            prompt = fmt_prompt_fw(gt)
            gen = generate_from_ckpt_mixed(ckpt_path, tok, None, prompt, max_new=30)
            print(f"    [{i}] GT: {gt[:80]}")
            print(f"        GEN: {gen}")
        print()


def generate_from_ckpt(ckpt_path, tokenizer, img, prompt, max_new=50, temp=0.7):
    from prepare import Tokenizer as BaseTok

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = VLConfig(**cfg_dict)
    model = VLMModel(cfg, use_img_tokens=True).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if img is not None:
        img = img.unsqueeze(0).to(device)
    ids = tokenizer.encode(prompt)
    ids = ids[:120]

    with torch.no_grad():
        for _ in range(max_new):
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            logits = model(img, input_ids)
            logits = logits[0, -1, :] / temp
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id == tokenizer.pad_id or len(ids) > 127:
                break
            ids.append(next_id)

    del model
    torch.cuda.empty_cache()
    return tokenizer.decode(ids)


def generate_from_ckpt_mixed(ckpt_path, tokenizer, img, prompt, max_new=50, temp=0.7):
    from prepare import Tokenizer as BaseTok

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = VLConfig(**cfg_dict)
    use_img = img is not None
    model = VLMModel(cfg, use_img_tokens=use_img).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if img is not None:
        img = img.unsqueeze(0).to(device)
    ids = tokenizer.encode(prompt)
    ids = ids[:120]

    with torch.no_grad():
        for _ in range(max_new):
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            logits = model(img, input_ids)
            logits = logits[0, -1, :] / temp
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id == tokenizer.pad_id or len(ids) > 127:
                break
            ids.append(next_id)

    del model
    torch.cuda.empty_cache()
    return tokenizer.decode(ids)


if __name__ == "__main__":
    main()
