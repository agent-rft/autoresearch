"""
Mixed Multimodal Training: Space + Oxford Pets + MathVision + FineWeb text.
Samples from all 4 datasets with configurable mixing ratios.
Architecture: VLM with ResNet encoder (best from previous experiments).
"""

import os
import gc
import io

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

device = torch.device("cuda")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

IMG_SIZE = 224


class SpaceVLMDataset(Dataset):
    def __init__(self, split="train", img_size=IMG_SIZE):
        self.img_size = img_size
        from datasets import load_dataset

        self.ds = load_dataset("AIOmarRehan/space-multimodal-dataset", split=split)
        print(f"  Space dataset: {len(self.ds)} samples")

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
        return img_tensor, text, "vision"


class OxfordPetsDataset(Dataset):
    def __init__(self, split="train", img_size=IMG_SIZE):
        self.img_size = img_size
        from datasets import load_dataset

        self.ds = load_dataset("enterprise-explorers/oxford-pets", split=split)
        print(f"  Oxford Pets dataset: {len(self.ds)} samples")

    def _normalize_image(self, img):
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
        label = row["label"]
        dog = row["dog"]
        animal = "dog" if dog else "cat"
        text = f"A {label} {animal}."
        img_tensor = self._normalize_image(row["image"])
        return img_tensor, text, "vision"


class MathVisionDataset(Dataset):
    def __init__(self, split="test", img_size=IMG_SIZE, max_samples=None):
        self.img_size = img_size
        from datasets import load_dataset

        self.ds = load_dataset("macabdul9/MathVision", split=split)
        if max_samples:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        print(f"  MathVision dataset: {len(self.ds)} samples")

    def _normalize_image(self, img):
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
        question = row["question"].replace("<image1>", "").strip()
        answer = str(row["answer"]) if row["answer"] else ""
        text = f"Q: {question} A: {answer}"
        img = row.get("decoded_image") or row.get("image")
        img_tensor = self._normalize_image(img)
        return img_tensor, text, "vision"


class FineWebTextDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        from datasets import load_dataset

        self.ds = load_dataset(
            "Lambent/creative-writing-2048-fineweb-edu-sample", split=split
        )
        if max_samples:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        print(f"  FineWeb dataset: {len(self.ds)} samples")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        text = row["text"]
        return None, text, "text"


class MixedDataCollator:
    def __init__(self, tokenizer, max_len=128, vision_weight=1.0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vision_weight = vision_weight

    def __call__(self, batch):
        items = [b for b in batch if b[0] is not None]
        text_only = [b for b in batch if b[0] is None]

        if items:
            imgs = torch.stack([b[0] for b in items])
            texts = [b[1] for b in items]
            ids = [self.tokenizer.encode(t, max_len=self.max_len - 1) for t in texts]
            max_batch_len = max(len(i) for i in ids)
            padded = torch.full(
                (len(ids), max_batch_len), self.tokenizer.pad_id, dtype=torch.long
            )
            for i, seq in enumerate(ids):
                padded[i, : len(seq)] = torch.tensor(seq)
            labels = padded.clone()
            labels[padded == self.tokenizer.pad_id] = -1
            return imgs, padded, labels, "vision"

        if text_only:
            texts = [b[1] for b in text_only]
            ids = [self.tokenizer.encode(t, max_len=self.max_len - 1) for t in texts]
            max_batch_len = max(len(i) for i in ids)
            padded = torch.full(
                (len(ids), max_batch_len), self.tokenizer.pad_id, dtype=torch.long
            )
            for i, seq in enumerate(ids):
                padded[i, : len(seq)] = torch.tensor(seq)
            labels = padded.clone()
            labels[padded == self.tokenizer.pad_id] = -1
            return None, padded, labels, "text"

        return None, None, None, None


class WeightedMixSampler(torch.utils.data.Sampler):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights
        self.lengths = [len(d) for d in datasets]
        self.total = sum(self.lengths)
        self.cumulative = [0] + list(np.cumsum(self.lengths))

    def __iter__(self):
        # Generate indices proportionally
        indices = []
        for i, (ds, w) in enumerate(zip(self.datasets, self.weights)):
            n = int(
                np.round(
                    len(ds)
                    * w
                    * self.total
                    / sum(w * l for w, l in zip(self.weights, self.lengths))
                )
            )
            n = min(n, len(ds))
            sampled = torch.randperm(len(ds))[:n].tolist()
            indices.extend([(i, idx) for idx in sampled])
        # Shuffle combined
        import random

        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.total


class MixedDataset(Dataset):
    def __init__(self, datasets, weights, epoch_size=2000):
        self.datasets = datasets
        self.weights = weights
        self.lengths = [len(d) for d in datasets]
        self.epoch_size = epoch_size
        self._build_indices()

    def _build_indices(self):
        probs = np.array(
            [w * l for w, l in zip(self.weights, self.lengths)], dtype=float
        )
        probs /= probs.sum()
        per_ds = (probs * self.epoch_size).astype(int)
        per_ds = np.maximum(per_ds, 1)
        indices = []
        for di, n in enumerate(per_ds):
            ids = np.random.permutation(self.lengths[di])[:n].tolist()
            indices.extend([(di, i) for i in ids])
        np.random.shuffle(indices)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        di, i = self.indices[idx]
        return self.datasets[di][i]

    def on_epoch_end(self):
        self._build_indices()


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
        self.vision_encoder = ResNetStyleEncoder(config.vision_embed_dim)
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
        img_emb = None

        if images is not None and self.use_img_tokens:
            img_feats = self.vision_encoder(images)
            img_emb = self.img_proj(img_feats)

        tok_emb = self.token_emb(token_ids)

        if img_emb is not None:
            x = torch.cat([img_emb, tok_emb], dim=1)
            pos = self.pos_emb[:, : self.num_img_tokens + T]
            x = x + pos
            logits = self.lm_head(self._transformer_forward(x))[
                :, self.num_img_tokens :, :
            ]
        else:
            x = tok_emb + self.pos_emb[:, :T]
            logits = self.lm_head(self._transformer_forward(x))

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-1,
            )
            return loss
        return logits

    def _transformer_forward(self, x):
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


def mixed_collate(batch, tokenizer, max_len=128):
    imgs = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    types = [b[2] for b in batch]

    has_vision = any(t == "vision" for t in types)

    if has_vision:
        valid_imgs = [b[0] for b in batch if b[0] is not None]
        valid_texts = [b[1] for b in batch if b[0] is not None]
        if valid_imgs:
            imgs_t = torch.stack(valid_imgs)
            ids = [tokenizer.encode(t, max_len=max_len - 1) for t in valid_texts]
            max_batch_len = max(len(i) for i in ids)
            padded = torch.full(
                (len(ids), max_batch_len), tokenizer.pad_id, dtype=torch.long
            )
            for i, seq in enumerate(ids):
                padded[i, : len(seq)] = torch.tensor(seq)
            labels = padded.clone()
            labels[padded == tokenizer.pad_id] = -1
            return imgs_t, padded, labels

    text_texts = [b[1] for b in batch if b[0] is None]
    if text_texts:
        ids = [tokenizer.encode(t, max_len=max_len - 1) for t in text_texts]
        max_batch_len = max(len(i) for i in ids)
        padded = torch.full(
            (len(ids), max_batch_len), tokenizer.pad_id, dtype=torch.long
        )
        for i, seq in enumerate(ids):
            padded[i, : len(seq)] = torch.tensor(seq)
        labels = padded.clone()
        labels[padded == tokenizer.pad_id] = -1
        return None, padded, labels

    return None, None, None


def train_mixed(
    name,
    datasets,
    dataset_weights,
    vision_type="resnet",
    n_layer=4,
    n_embd=256,
    vision_dim=256,
    lr=3e-4,
    vision_lr=3e-5,
    epochs=20,
    batch_size=8,
    max_len=128,
    save_dir=None,
):
    from prepare import Tokenizer

    print(f"\n  [{name}] Loading datasets...")
    for ds in datasets:
        print(f"    {type(ds).__name__}: {len(ds)} samples")

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
    mixed_ds = MixedDataset(datasets, dataset_weights, epoch_size=2000)
    print(f"  Mixed dataset size: {len(mixed_ds)}")

    train_loader = DataLoader(
        mixed_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: mixed_collate(b, tokenizer, max_len),
        num_workers=0,
    )

    model = VLMModel(cfg, use_img_tokens=True).to(device)
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
    print(f"  [{name}] Model: {nparams:,} params ({nparams / 1e6:.1f}M)")

    if save_dir is None:
        save_dir = os.path.expanduser("~/.cache/autoresearch/vision_checkpoints_mixed")
    os.makedirs(save_dir, exist_ok=True)

    save_every = max(1, epochs // 5)
    best_loss = float("inf")
    all_params = list(model.parameters())

    for epoch in range(epochs):
        mixed_ds.on_epoch_end()
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            imgs, input_ids, labels = batch
            if input_ids is None:
                continue

            imgs_dev = imgs.to(device) if imgs is not None else None
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss = model(imgs_dev, input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"  Epoch {epoch:02d}/{epochs - 1} | loss={avg_loss:.4f} | batches={n_batches}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = os.path.join(save_dir, f"best_{name}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "loss": best_loss,
                    "cfg": dict(vars(cfg)),
                    "name": name,
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
                    "name": name,
                },
                ckpt,
            )

        gc.collect()
        torch.cuda.empty_cache()

    print(f"  [{name}] Best loss: {best_loss:.4f}")
    return best_loss


def main():
    print("=" * 60)
    print("Mixed Multimodal Training: Space + Pets + Math + FineWeb")
    print("=" * 60)

    results = {}

    space_ds = SpaceVLMDataset()
    pets_ds = OxfordPetsDataset()
    math_ds = MathVisionDataset(split="test")
    fineweb_ds = FineWebTextDataset()

    configs = [
        {
            "name": "mixed_natural",
            "weights": [1.0, 1.0, 1.0, 1.0],
            "desc": "Natural mix (Pets dominant: 66% of samples)",
        },
        {
            "name": "mixed_balanced",
            "weights": [29.56, 1.0, 4.90, 3.61],
            "desc": "Balanced: equal representation from each dataset",
        },
        {
            "name": "mixed_vision_focus",
            "weights": [3.0, 3.0, 3.0, 1.0],
            "desc": "Vision 3x weight over text-only",
        },
    ]

    for cfg in configs:
        name = cfg["name"]
        weights = cfg["weights"]
        print(f"\n{'=' * 60}")
        print(f"[{name}] Weights: {weights} | {cfg['desc']}")
        print(f"{'=' * 60}")

        try:
            loss = train_mixed(
                name=name,
                datasets=[space_ds, pets_ds, math_ds, fineweb_ds],
                dataset_weights=weights,
                vision_type="resnet",
                n_layer=4,
                n_embd=256,
                vision_dim=256,
                lr=3e-4,
                vision_lr=3e-5,
                epochs=20,
                batch_size=8,
            )
            results[name] = {
                "loss": loss,
                "status": "ok",
                "weights": weights,
                "desc": cfg["desc"],
            }
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback

            traceback.print_exc()
            results[name] = {
                "loss": float("inf"),
                "status": f"error: {e}",
                "weights": weights,
            }

    print("\n" + "=" * 60)
    print("MIXED TRAINING RESULTS")
    print("=" * 60)
    baseline = 0.8522
    for name, res in sorted(results.items(), key=lambda x: x[1]["loss"]):
        delta = baseline - res["loss"]
        tag = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        print(
            f"  {name:<30} loss={res['loss']:.4f}  vs_baseline={tag}  [{res['status']}]"
        )
    print(f"\n  Baseline (resnet_finetune single-dataset): 0.8522")


if __name__ == "__main__":
    main()
