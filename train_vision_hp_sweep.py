"""
Hyperparameter sweep on best encoder (ResNet).
Experiments:
  1. resnet_ft_long   - ResNet finetune, 40 epochs (more training)
  2. resnet_ft_big    - ResNet finetune, n_embd=384, n_layer=6 (larger model)
  3. resnet_ft_lr3    - ResNet finetune, lr=3e-4 (higher LR)
  4. resnet_ft_deep   - ResNet finetune, n_layer=8 (deeper model)
  5. cnn_ft_big       - CNN finetune, n_embd=384, n_layer=6 (larger CNN)
"""

import os
import gc
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io

device = torch.device("cuda")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

IMG_SIZE = 224


class SpaceVLMDataset(Dataset):
    def __init__(self, split="train", img_size=IMG_SIZE):
        self.img_size = img_size
        self.split = split
        from datasets import load_dataset

        self.ds = load_dataset("AIOmarRehan/space-multimodal-dataset", split=split)
        print(f"Loaded HF dataset: {len(self.ds)} samples")

    def _normalize_image(self, img):
        if img is None:
            return torch.zeros(3, self.img_size, self.img_size)
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
        img_arr = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img_arr - mean) / std

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        text = row["text"]
        img = row.get("image") or row.get("image_path")
        img_tensor = self._normalize_image(img)
        return img_tensor, text


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


def build_encoder(vision_type, embed_dim, img_size):
    if vision_type == "cnn":
        return CNNEncoder(embed_dim, img_size)
    elif vision_type == "resnet":
        return ResNetStyleEncoder(embed_dim)
    else:
        raise ValueError(f"Unknown: {vision_type}")


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
        vision_type="cnn",
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
    def __init__(self, config):
        super().__init__()
        self.cfg = config
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
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.001)
        for p in self.parameters():
            if p.dim() > 1 and not isinstance(p, nn.Embedding):
                nn.init.xavier_uniform_(p)

    def forward(self, images, token_ids, labels=None):
        B, T = token_ids.size()
        img_feats = self.vision_encoder(images)
        img_emb = self.img_proj(img_feats)
        tok_emb = self.token_emb(token_ids)
        seq_len = T + self.num_img_tokens
        x = torch.cat([img_emb, tok_emb], dim=1)
        pos = self.pos_emb[:, :seq_len]
        x = x + pos
        x0 = x
        for blk in self.blocks:
            x = blk(x)
            x = x + self.x0_scale * x0
        x = self.norm(x)
        logits = self.lm_head(x[:, self.num_img_tokens :, :])
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-1,
            )
            return loss
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_heads = max(1, cfg.n_embd // 64)
        self.attn = nn.MultiheadAttention(
            cfg.n_embd, n_heads, batch_first=True, bias=False, dropout=0.1
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


def collate_fn(batch, tokenizer, max_len=128):
    texts = [b[1] for b in batch]
    imgs = torch.stack([b[0] for b in batch])
    ids = [tokenizer.encode(t, max_len=max_len - 1) for t in texts]
    max_batch_len = max(len(i) for i in ids)
    padded = torch.full((len(ids), max_batch_len), tokenizer.pad_id, dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq)
    labels = padded.clone()
    labels[padded == tokenizer.pad_id] = -1
    return imgs, padded, labels


def train_vlm(
    name,
    vision_type,
    n_layer,
    n_embd,
    vision_dim,
    lr,
    vision_lr,
    epochs,
    batch_size,
    save_dir=None,
):
    from prepare import Tokenizer

    n_heads = max(1, n_embd // 64)
    cfg = VLConfig(
        vocab_size=8192,
        n_layer=n_layer,
        n_head=n_heads,
        n_embd=n_embd,
        vision_embed_dim=vision_dim,
        context_len=128,
        img_size=224,
        vision_type=vision_type,
    )

    tokenizer = VLTokenizer(Tokenizer.from_directory())

    ds = SpaceVLMDataset(split="train", img_size=cfg.img_size)
    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, 128),
        num_workers=0,
    )

    model = VLMModel(cfg).to(device)
    vis_ids = {id(p) for p in model.vision_encoder.parameters()}
    param_groups = [
        {
            "params": [p for p in model.parameters() if id(p) in vis_ids],
            "lr": vision_lr,
        },
        {"params": [p for p in model.parameters() if id(p) not in vis_ids], "lr": lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1, betas=(0.9, 0.95))

    nparams = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Total: {nparams:,} params ({nparams / 1e6:.1f}M)")

    if save_dir is None:
        save_dir = os.path.expanduser("~/.cache/autoresearch/vision_checkpoints_hp")
    os.makedirs(save_dir, exist_ok=True)

    save_every = max(1, epochs // 5)
    best_loss = float("inf")
    all_params = list(model.parameters())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for imgs, input_ids, labels in train_loader:
            imgs = imgs.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss = model(imgs, input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:02d}/{epochs - 1} | loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = os.path.join(save_dir, f"best_{name}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "loss": best_loss,
                    "cfg": dict(vars(cfg)),
                },
                ckpt,
            )

        if epoch % save_every == 0 or epoch == epochs - 1:
            ckpt = os.path.join(save_dir, f"epoch{epoch:02d}_{name}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "loss": avg_loss,
                    "cfg": dict(vars(cfg)),
                },
                ckpt,
            )

        gc.collect()
        torch.cuda.empty_cache()

    print(f"  [{name}] Best loss: {best_loss:.4f}")
    return best_loss


def main():
    print("=" * 60)
    print("Hyperparameter Sweep: ResNet + CNN Finetuning")
    print("=" * 60)

    sweep_configs = [
        # name, vision, n_layer, n_embd, vision_dim, lr, vision_lr, epochs, batch
        ("resnet_ft_long40", "resnet", 4, 256, 256, 1e-4, 1e-5, 40, 8),
        ("resnet_ft_big", "resnet", 6, 384, 384, 1e-4, 1e-5, 20, 8),
        ("resnet_ft_lr3", "resnet", 4, 256, 256, 3e-4, 3e-5, 20, 8),
        ("resnet_ft_deep8", "resnet", 8, 256, 256, 1e-4, 1e-5, 20, 8),
        ("cnn_ft_big", "cnn", 6, 384, 384, 1e-4, 1e-5, 20, 8),
        ("resnet_ft_lr5", "resnet", 4, 256, 256, 5e-4, 5e-5, 20, 8),
    ]

    results = {}

    for cfg_tuple in sweep_configs:
        name = cfg_tuple[0]
        vision_type, n_layer, n_embd, vision_dim, lr, vision_lr, epochs, batch = (
            cfg_tuple[1:]
        )

        print(f"\n{'=' * 60}")
        print(
            f"[{name}] {vision_type} | layers={n_layer} | n_embd={n_embd} | "
            f"vision_dim={vision_dim} | lr={lr} | vision_lr={vision_lr} | "
            f"epochs={epochs} | batch={batch}"
        )
        print(f"{'=' * 60}")

        try:
            loss = train_vlm(
                name=name,
                vision_type=vision_type,
                n_layer=n_layer,
                n_embd=n_embd,
                vision_dim=vision_dim,
                lr=lr,
                vision_lr=vision_lr,
                epochs=epochs,
                batch_size=batch,
            )
            results[name] = {"loss": loss, "status": "ok", "params": cfg_tuple}
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback

            traceback.print_exc()
            results[name] = {"loss": float("inf"), "status": f"error: {e}"}

    baselines = {
        "resnet_finetune_20ep": 0.8522,
        "resnet_frozen_20ep": 0.8873,
    }

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 60)
    print(
        f"\n{'Name':<30} {'Best Loss':<12} {'Status':<8} {'vs_baseline':<15} {'Params'}"
    )
    print("-" * 85)

    baseline = 0.8522
    for name, res in sorted(results.items(), key=lambda x: x[1]["loss"]):
        delta = baseline - res["loss"]
        tag = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        params = (
            res.get("params", ("", "", "", "", "", "", "", ""))
            if res["status"] == "ok"
            else ""
        )
        cfg_str = (
            f"l={params[2]} e={params[3]} vd={params[4]} lr={params[5]}"
            if params
            else ""
        )
        print(f"{name:<30} {res['loss']:<12.4f} {res['status']:<8} {tag:<15} {cfg_str}")

    print(f"\n{'Baseline (resnet finetune 20ep)':<30} 0.8522")
    print(f"{'Baseline (resnet frozen 20ep)':<30} 0.8873")


if __name__ == "__main__":
    main()
