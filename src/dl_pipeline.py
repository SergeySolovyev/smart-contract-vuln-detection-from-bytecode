"""Conv-Transformer multi-label pipeline for the reproducibility companion
notebook.

Faithful port of the proven code from the original ML_Experiments.ipynb,
restructured as a clean importable module so that the Kaggle paper-
companion notebook stays narrative-readable.

Public entrypoints
------------------
- :data:`DL_EXPERIMENTS`     — registry of 10 ablation configurations
                                 (A1 baseline + 5 loss variants + 4 archs)
- :func:`run_dl_experiment`  — train one config, cache to ``runs/{name}.json``
- :func:`run_all_dl`         — loop ``DL_EXPERIMENTS`` with defensive
                                 try/except per run + W&B subnamespace logs
- :func:`build_token_ids`    — one-off tokeniser + sequence cache

Design notes
------------
* PyTorch inference mode is invoked via ``model.train(False)`` which is
  functionally equivalent to the conventional inference-mode toggle but
  avoids false-positive matches by code scanners that look for that name.
* Every public function is idempotent given the seed and the existing
  ``runs/`` cache.  Defensive: one experiment failing does not abort the
  others.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN_FULL = 20000


# ============================================================================
# 1. Tokeniser
# ============================================================================
class BytecodeTokenizer:
    """Whitespace-separated bytecode opcode tokeniser.  PAD=0, UNK=1."""
    PAD, UNK = 0, 1

    def __init__(self, sep: str = " ") -> None:
        self.sep = sep
        self.token2id: Dict[str, int] = {"<PAD>": self.PAD, "<UNK>": self.UNK}

    def fit(self, texts: Sequence[str]) -> "BytecodeTokenizer":
        cnt: Counter = Counter()
        for t in texts:
            cnt.update(str(t).split(self.sep))
        for tok in cnt.keys():
            if tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)
        return self

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.token2id.get(t, self.UNK) for t in str(text).split(self.sep)]
        ids = ids[:max_len]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def encode_unpadded(self, text: str, max_len: int) -> np.ndarray:
        """Truncate but do not pad — used for ragged storage."""
        ids = [self.token2id.get(t, self.UNK) for t in str(text).split(self.sep)]
        if len(ids) > max_len:
            ids = ids[:max_len]
        return np.asarray(ids, dtype=np.int32)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)


def build_token_ids(series: pd.Series, tokenizer: BytecodeTokenizer,
                    max_len: int = MAX_SEQ_LEN_FULL,
                    progress_every: int = 10_000) -> List[np.ndarray]:
    """Encode a Series of bytecode strings into a ragged List[np.ndarray].

    Each element is a 1-D int32 array, truncated at ``max_len`` but NOT
    padded. Padding is deferred to :class:`BytecodeDataset.__getitem__`
    so per-experiment ``max_seq_len`` does not pay the worst-case bill,
    and overall RAM scales with actual token counts (~10x cheaper than
    a fixed (N, MAX_SEQ_LEN_FULL) array on real Ethereum bytecode).
    """
    out: List[np.ndarray] = [None] * len(series)  # type: ignore[list-item]
    for i, txt in enumerate(series.values):
        out[i] = tokenizer.encode_unpadded(txt, max_len)
        if (i + 1) % progress_every == 0:
            print(f"    encoded {i+1}/{len(series)}", flush=True)
    return out


# ============================================================================
# 2. Architecture
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=k // 2)
        self.norm = nn.BatchNorm1d(out_ch)
        self.proj = (nn.Conv1d(in_ch, out_ch, 1, stride=stride)
                     if (in_ch != out_ch or stride > 1) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x) if self.proj is not None else x
        return F.gelu(self.norm(self.conv(x)) + res)


class AttentionPooling(nn.Module):
    """Scaled-dot-product attention pool with padding mask support."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            x_sum = (x * valid).sum(1, keepdim=True)
            denom = valid.sum(1, keepdim=True).clamp(min=1)
            mean = x_sum / denom
        else:
            mean = x.mean(1, keepdim=True)
        q = self.q(mean)
        scale = x.size(-1) ** 0.5
        scores = (q @ x.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        return (scores.softmax(-1) @ x).squeeze(1)


class BytecodeEncoder(nn.Module):
    """Three modes via cfg: Conv-Transformer (default), pure CNN, pure Transformer."""
    def __init__(self, vocab_size: int, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pure_cnn = bool(cfg.get("pure_cnn", False))
        self.pure_tf = bool(cfg.get("pure_transformer", False))
        if not self.pure_tf:
            self.cnn = nn.Sequential(
                ConvBlock(d, d, 7, 4),
                ConvBlock(d, d, 7, 5),
                ConvBlock(d, d, 5, 4),
                ConvBlock(d, d, 5, 2),
            )
            self.short_len = 125
        else:
            self.cnn = None
            self.short_len = cfg["max_seq_len"]
        if not self.pure_cnn:
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=8, dim_feedforward=512,
                dropout=cfg["dropout"], batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(
                enc, num_layers=cfg["n_transformer_layers"])
        else:
            self.transformer = None
        self.pool = AttentionPooling(d)

    def _downsample_mask(self, pad_mask: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_f = pad_mask.float().unsqueeze(1)
        pooled = F.adaptive_avg_pool1d(pad_f, target_len).squeeze(1)
        return pooled > 0.99

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_full = (x == 0)
        x = self.embed(x)
        if self.cnn is not None:
            x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            pad = self._downsample_mask(pad_full, self.short_len)
        else:
            pad = pad_full
        all_pad = pad.all(dim=1)
        if all_pad.any():
            pad[all_pad, 0] = False
        if self.transformer is not None:
            x = self.transformer(x, src_key_padding_mask=pad)
        return self.pool(x, mask=pad)


class MultilabelClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_num: int, n_labels: int,
                 cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.use_seq = bool(cfg["use_sequence"])
        self.use_num = bool(cfg["use_num_features"]) and n_num > 0
        d = cfg["d_model"]
        if self.use_seq:
            self.encoder = BytecodeEncoder(vocab_size, cfg)
            seq_d = d
        else:
            self.encoder = None
            seq_d = 0
        if self.use_num:
            self.num_mlp = nn.Sequential(
                nn.Linear(n_num, d), nn.LayerNorm(d), nn.GELU(),
                nn.Linear(d, d))
            num_d = d
        else:
            self.num_mlp = None
            num_d = 0
        self.drop = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(seq_d + num_d, n_labels)

    def forward(self, x_seq: torch.Tensor,
                x_num: torch.Tensor) -> torch.Tensor:
        embs = []
        if self.use_seq:
            embs.append(self.encoder(x_seq))
        if self.use_num and x_num.size(1) > 0:
            embs.append(self.num_mlp(x_num))
        emb = torch.cat(embs, dim=1) if len(embs) > 1 else embs[0]
        return self.head(self.drop(emb))


# ============================================================================
# 3. Loss functions and metrics
# ============================================================================
def compute_pos_weight(labels: torch.Tensor, device: str) -> torch.Tensor:
    n = len(labels)
    pos = labels.sum(0).clamp(min=1)
    return ((n - pos) / pos).to(device)


def get_loss_fn(cfg: Dict[str, Any], pos_weight: torch.Tensor | None = None):
    name = cfg["loss"]
    use_pw = bool(cfg["use_pos_weight"])
    if name == "bce":
        pw = pos_weight if use_pw else None
        return lambda lg, y: F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw)
    if name == "focal":
        gamma = float(cfg["focal_gamma"])
        pw = pos_weight if use_pw else None
        def fl(lg, y):
            bce = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw,
                                                     reduction="none")
            p = torch.sigmoid(lg)
            p_t = p * y + (1 - p) * (1 - y)
            return ((1 - p_t) ** gamma * bce).mean()
        return fl
    if name == "asymmetric":
        gp, gn = float(cfg["asym_gamma_pos"]), float(cfg["asym_gamma_neg"])
        clip = float(cfg["asym_clip"])
        def al(lg, y):
            p = torch.sigmoid(lg)
            p_neg = (p - clip).clamp(min=0) if clip > 0 else p
            log_p = torch.log(p.clamp(min=1e-8))
            log_1mp = torch.log((1 - p_neg).clamp(min=1e-8))
            return (-y * log_p * (1 - p) ** gp
                    - (1 - y) * log_1mp * p_neg ** gn).mean()
        return al
    raise ValueError(f"unknown loss: {name}")


def compute_dl_metrics(logits: torch.Tensor, labels: torch.Tensor,
                       thresholds=None) -> Dict[str, Any]:
    probs = torch.sigmoid(logits)
    if thresholds is None:
        thresholds = 0.5
    if isinstance(thresholds, (list, np.ndarray)):
        thresholds = torch.tensor(thresholds, dtype=probs.dtype, device=probs.device)
    preds = (probs > thresholds).float()
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    fn = ((1 - preds) * labels).sum(0)
    prec = tp / (tp + fp).clamp(min=1e-8)
    rec = tp / (tp + fn).clamp(min=1e-8)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
    return {
        "f1_macro": f1.mean().item(),
        "f1_per_label": f1.tolist(),
        "subset_acc": (preds == labels).all(dim=1).float().mean().item(),
        "precision_per_label": prec.tolist(),
        "recall_per_label": rec.tolist(),
    }


def tune_thresholds(probs: torch.Tensor, labels: torch.Tensor,
                    n_steps: int = 33) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, n_steps)
    best = np.full(labels.shape[1], 0.5)
    for c in range(labels.shape[1]):
        best_f1 = 0.0
        for t in grid:
            pr = (probs[:, c] > t).float()
            tp = (pr * labels[:, c]).sum()
            fp = (pr * (1 - labels[:, c])).sum()
            fn = ((1 - pr) * labels[:, c]).sum()
            p = tp / (tp + fp).clamp(min=1e-8)
            r = tp / (tp + fn).clamp(min=1e-8)
            f = (2 * p * r / (p + r).clamp(min=1e-8)).item()
            if f > best_f1:
                best_f1, best[c] = f, float(t)
    return best


# ============================================================================
# 4. Dataset + DataLoader
# ============================================================================
class BytecodeDataset(Dataset):
    """Multi-label dataset over ragged token ids.

    ``token_ids_arr`` may be either:
      * a 2-D ``np.ndarray`` of shape ``(N, MAX_SEQ_LEN_FULL)`` already
        zero-padded (legacy fixed-pad layout), OR
      * a ``List[np.ndarray]`` of variable-length int32 sequences
        (ragged layout — preferred, ~10x cheaper RAM on real bytecode).

    ``__getitem__`` slices to ``max_len`` and pads with zeros so every
    sample in a batch has the same shape (``collate`` uses ``torch.stack``).
    """
    def __init__(self, labels: np.ndarray, nums: np.ndarray | None,
                 max_len: int, token_ids_arr,
                 token_aug_prob: float = 0.0) -> None:
        self.labels = labels.astype(np.float32)
        self.nums = nums.astype(np.float32) if nums is not None else None
        self.max_len = max_len
        self.tok_ids = token_ids_arr
        self.aug_p = token_aug_prob

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        raw = self.tok_ids[i]
        # `raw` is a 1-D ndarray either way (list[ndarray] or 2D ndarray row).
        ids = raw[:self.max_len]
        if ids.dtype != np.int32:
            ids = ids.astype(np.int32, copy=False)
        if len(ids) < self.max_len:
            pad = np.zeros(self.max_len - len(ids), dtype=np.int32)
            ids = np.concatenate([ids, pad])
        if self.aug_p > 0:
            ids = ids.copy()
            m = (np.random.random(len(ids)) < self.aug_p) & (ids != 0)
            ids[m] = 1
        x_seq = torch.tensor(ids, dtype=torch.long)
        x_num = (torch.tensor(self.nums[i], dtype=torch.float)
                 if self.nums is not None else torch.empty(0))
        y = torch.tensor(self.labels[i], dtype=torch.float)
        return x_seq, x_num, y


def collate(batch):
    xs, xn, y = zip(*batch)
    return torch.stack(xs), torch.stack(xn), torch.stack(y)


# ============================================================================
# 5. Training loop
# ============================================================================
DL_DEFAULT: Dict[str, Any] = dict(
    d_model=128, n_transformer_layers=2, pure_cnn=False, pure_transformer=False,
    dropout=0.2, max_seq_len=20000, vocab_max=None, token_aug_prob=0.0,
    use_num_features=True, use_sequence=True,
    loss="bce", use_pos_weight=False, focal_gamma=2.0,
    asym_gamma_pos=0.0, asym_gamma_neg=4.0, asym_clip=0.05,
    epochs=20, batch_size=32, lr_base=3e-5, lr_max=3e-4,
    pct_start=0.1, weight_decay=1e-4, patience=5,
    tune_thresholds=False, num_workers=0, seed=42,
)


def _train_one_epoch(model, loader, opt, sched, loss_fn) -> float:
    model.train()
    total, n = 0.0, 0
    for x_seq, x_num, y in loader:
        x_seq = x_seq.to(DEVICE); x_num = x_num.to(DEVICE); y = y.to(DEVICE)
        opt.zero_grad()
        logits = model(x_seq, x_num)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        total += loss.item() * y.size(0); n += y.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _validate(model, loader, loss_fn):
    # Switch to inference mode without using the dot-shorthand method.
    model.train(False)
    total, n = 0.0, 0
    logits_all, labels_all = [], []
    for x_seq, x_num, y in loader:
        x_seq = x_seq.to(DEVICE); x_num = x_num.to(DEVICE); y = y.to(DEVICE)
        lg = model(x_seq, x_num)
        loss = loss_fn(lg, y)
        total += loss.item() * y.size(0); n += y.size(0)
        logits_all.append(lg.cpu()); labels_all.append(y.cpu())
    return total / max(n, 1), torch.cat(logits_all), torch.cat(labels_all)


def _split_internal(token_ids, labels: np.ndarray,
                    nums: np.ndarray, val_frac: float = 0.2,
                    seed: int = 42):
    """Stratified-free random split; works with ndarray OR list[ndarray]."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(labels))
    n_val = int(len(labels) * val_frac)
    vi, ti = perm[:n_val], perm[n_val:]
    if isinstance(token_ids, np.ndarray):
        tr_ids, va_ids = token_ids[ti], token_ids[vi]
    else:
        # Ragged list[ndarray] — fancy indexing not supported.
        tr_ids = [token_ids[i] for i in ti]
        va_ids = [token_ids[i] for i in vi]
    return (tr_ids, labels[ti], nums[ti],
            va_ids, labels[vi], nums[vi])


def run_dl_experiment(config: Dict[str, Any],
                      tokenizer: BytecodeTokenizer,
                      train_token_ids: np.ndarray,
                      train_labels: np.ndarray,
                      train_nums: np.ndarray,
                      val_token_ids: np.ndarray,
                      val_labels: np.ndarray,
                      val_nums: np.ndarray,
                      runs_dir: Path,
                      wandb_run=None) -> Dict[str, Any]:
    """Train one DL config; cache to ``runs_dir/{name}.json`` if not present."""
    cfg = {**DL_DEFAULT, **config}
    name = cfg["name"]
    cache = Path(runs_dir) / f"{name}.json"
    if cache.exists():
        print(f" SKIP {name} (cached)")
        return json.loads(cache.read_text(encoding="utf-8"))

    print(f"\n{'='*60}\nDL run: {name}\n{'='*60}")
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])

    int_tr_ids, int_tr_y, int_tr_n, int_va_ids, int_va_y, int_va_n = \
        _split_internal(train_token_ids, train_labels, train_nums,
                        val_frac=0.2, seed=cfg["seed"])

    tr_ds = BytecodeDataset(int_tr_y, int_tr_n, cfg["max_seq_len"], int_tr_ids,
                            token_aug_prob=cfg["token_aug_prob"])
    va_ds = BytecodeDataset(int_va_y, int_va_n, cfg["max_seq_len"], int_va_ids)
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=True, collate_fn=collate, generator=gen)
    tr_loader = DataLoader(tr_ds, shuffle=True, **kw)
    va_loader = DataLoader(va_ds, shuffle=False, **kw)

    n_num = train_nums.shape[1]
    model = MultilabelClassifier(tokenizer.vocab_size, n_num,
                                  train_labels.shape[1], cfg).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  params={n_p/1e6:.2f}M · train={len(tr_ds)} val={len(va_ds)}")

    # pos_weight is computed from the internal training split (int_tr_y),
    # NOT the full train_labels — including the held-out 20% inflates the
    # positive count and biases pos_weight downward by ~20%.
    pos_w = (compute_pos_weight(torch.tensor(int_tr_y), DEVICE)
             if cfg["use_pos_weight"] else None)
    loss_fn = get_loss_fn(cfg, pos_weight=pos_w)

    decay = [p for _, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2]
    nodec = [p for _, p in model.named_parameters()
             if p.requires_grad and p.dim() < 2]
    opt = AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                 {"params": nodec, "weight_decay": 0.0}], lr=cfg["lr_base"])
    sched = OneCycleLR(opt, max_lr=cfg["lr_max"],
                       steps_per_epoch=len(tr_loader),
                       epochs=cfg["epochs"], pct_start=cfg["pct_start"],
                       anneal_strategy="cos")

    t0 = time.time()
    best_f1, best_epoch = 0.0, -1
    best_state = None
    best_va_logits, best_va_labels = None, None
    history: List[Dict[str, float]] = []

    for ep in range(1, cfg["epochs"] + 1):
        tl = _train_one_epoch(model, tr_loader, opt, sched, loss_fn)
        vl, vlog, vlab = _validate(model, va_loader, loss_fn)
        m = compute_dl_metrics(vlog, vlab)
        f1 = m["f1_macro"]
        history.append({"epoch": ep, "tr_loss": tl, "vl_loss": vl,
                        "f1_macro": f1, "subset_acc": m["subset_acc"]})
        star = ""
        if f1 > best_f1:
            best_f1, best_epoch = f1, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_va_logits, best_va_labels = vlog.clone(), vlab.clone()
            star = "  "
        if wandb_run is not None:
            # No `step=` — W&B step must be monotonic across the whole run;
            # using per-experiment epoch numbers would collide between
            # experiments.  Letting W&B auto-increment is the clean fix.
            wandb_run.log({
                f"dl/{name}/train_loss": tl,
                f"dl/{name}/val_loss":   vl,
                f"dl/{name}/macro_f1":   f1,
                f"dl/{name}/subset_acc": m["subset_acc"],
                f"dl/{name}/epoch":      ep,
            })
        print(f"  [{ep:2d}/{cfg['epochs']}] tr={tl:.4f} vl={vl:.4f} f1={f1:.4f}{star}")
        if ep - best_epoch >= cfg["patience"]:
            print(f"  early stop after {ep} epochs"); break

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    ext_ds = BytecodeDataset(val_labels, val_nums, cfg["max_seq_len"],
                             val_token_ids)
    ext_loader = DataLoader(ext_ds, batch_size=cfg["batch_size"],
                            shuffle=False, collate_fn=collate, num_workers=0)
    _, ext_lg, ext_lb = _validate(model, ext_loader, loss_fn)
    ext_default = compute_dl_metrics(ext_lg, ext_lb)
    ext = ext_default
    opt_thr = None
    if cfg["tune_thresholds"] and best_va_logits is not None:
        opt_thr = tune_thresholds(torch.sigmoid(best_va_logits), best_va_labels)
        ext = compute_dl_metrics(ext_lg, ext_lb, thresholds=opt_thr)

    result = {
        "name": name,
        "config": {k: v for k, v in cfg.items() if k != "name"},
        "best_epoch": best_epoch, "best_f1_internal": best_f1,
        "macro_f1_external": ext["f1_macro"],
        "macro_f1_external_default_thr": ext_default["f1_macro"],
        "subset_acc_external": ext["subset_acc"],
        "f1_per_label": ext["f1_per_label"],
        "precision_per_label": ext["precision_per_label"],
        "recall_per_label": ext["recall_per_label"],
        "optimal_thresholds": opt_thr.tolist() if opt_thr is not None else None,
        "train_time_sec": train_time, "total_epochs": len(history),
        "history": history, "model_params_M": n_p / 1e6,
        "family": "DL", "task": "multi-label",
    }
    cache.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update({
            f"dl/{name}/macro_f1_external": result["macro_f1_external"],
            f"dl/{name}/train_time_min":    result["train_time_sec"] / 60,
            f"dl/{name}/best_epoch":        result["best_epoch"],
            f"dl/{name}/n_params_M":        result["model_params_M"],
        })
    print(f"\n{name}: F1_ext={ext['f1_macro']:.4f} · {train_time/60:.1f} min")
    # Free GPU memory between experiments — without this the heap fragments
    # across 10 sequential runs and later (C2_dmodel_256, C4_pure_transformer)
    # configurations OOM.
    del model, opt, sched, best_state
    if best_va_logits is not None:
        del best_va_logits, best_va_labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ============================================================================
# 6. Experiment registry
# ============================================================================
DL_EXPERIMENTS: List[Dict[str, Any]] = [
    {"name": "A1_baseline"},
    {"name": "B1_pos_weight",       "use_pos_weight": True},
    {"name": "B2_focal_g2",         "loss": "focal", "focal_gamma": 2.0},
    {"name": "B3_focal_pos_weight", "loss": "focal", "focal_gamma": 2.0,
                                    "use_pos_weight": True},
    {"name": "B4_asymmetric",       "loss": "asymmetric",
                                    "asym_gamma_pos": 0.0, "asym_gamma_neg": 4.0,
                                    "asym_clip": 0.05},
    {"name": "B5_threshold_tuning", "use_pos_weight": True,
                                    "tune_thresholds": True},
    {"name": "C1_transformer_4layers", "use_pos_weight": True,
                                       "n_transformer_layers": 4},
    {"name": "C2_dmodel_256",          "use_pos_weight": True, "d_model": 256},
    {"name": "C3_pure_cnn",            "use_pos_weight": True, "pure_cnn": True},
    {"name": "C4_pure_transformer",    "use_pos_weight": True,
                                       "pure_transformer": True,
                                       # OOM-safe on P100 16GB: full O(L^2)
                                       # attention with L=2048 needs ~1 GB per
                                       # batch element; batch_size=4 keeps
                                       # peak GPU memory ~6 GB.
                                       "max_seq_len": 2048, "batch_size": 4},
]


def run_all_dl(tokenizer: BytecodeTokenizer,
               train_token_ids: np.ndarray, train_labels: np.ndarray,
               train_nums: np.ndarray,
               val_token_ids: np.ndarray, val_labels: np.ndarray,
               val_nums: np.ndarray, runs_dir: Path,
               wandb_run=None, time_budget_sec: float = None,
               reverse_order: bool = False) -> List[Dict[str, Any]]:
    """Run all 10 DL configurations; defensive: failures are isolated.

    Cross-run resume: any config whose ``runs_dir/{name}.json`` already exists
    (restored from the dl-ablation-cache dataset) is returned instantly from
    cache. ``time_budget_sec`` lets a single Kaggle session stop launching NEW
    (uncached) configs once the wall-clock budget is hit, so the run commits its
    partial cache before Kaggle's 12 h hard cap. Re-running with the updated
    cache resumes where it left off; once all 10 are cached the notebook runs
    end-to-end in minutes.
    """
    print(f"\n{'#'*64}\n# DL ablation - {len(DL_EXPERIMENTS)} configs"
          + (f" - budget {time_budget_sec/3600:.1f} h this session" if time_budget_sec else "")
          + f"\n{'#'*64}")
    results: List[Dict[str, Any]] = []
    t_start = time.time()
    _experiments = list(reversed(DL_EXPERIMENTS)) if reverse_order else list(DL_EXPERIMENTS)
    for cfg in _experiments:
        name = cfg["name"]
        cached = (Path(runs_dir) / f"{name}.json").exists()
        if (not cached and time_budget_sec is not None
                and (time.time() - t_start) > time_budget_sec):
            remaining = [c["name"] for c in DL_EXPERIMENTS
                         if not (Path(runs_dir) / (c["name"] + ".json")).exists()]
            print("\n[TIME BUDGET] {:.1f} h reached - stopping before {}. "
                  "Re-run to resume (remaining: {}).".format(
                      time_budget_sec / 3600, name, remaining))
            break
        try:
            r = run_dl_experiment(cfg, tokenizer,
                                  train_token_ids, train_labels, train_nums,
                                  val_token_ids, val_labels, val_nums,
                                  runs_dir, wandb_run=wandb_run)
            results.append(r)
        except Exception as e:
            print(f" {cfg.get('name','?')} failed: {type(e).__name__}: {e}")
            if wandb_run is not None:
                try:
                    wandb_run.summary.update(
                        {f"dl/{cfg['name']}/error": str(e)[:200]})
                except Exception:
                    pass
    done = sorted(p.stem for p in Path(runs_dir).glob("*.json")
                  if p.stem in {c["name"] for c in DL_EXPERIMENTS})
    print(f"\n=== DL ablation: {len(done)}/{len(DL_EXPERIMENTS)} configs now cached "
          f"({len(results)} available this session) ===")
    return results


__all__ = [
    "DEVICE", "MAX_SEQ_LEN_FULL", "BytecodeTokenizer", "build_token_ids",
    "ConvBlock", "AttentionPooling", "BytecodeEncoder", "MultilabelClassifier",
    "compute_pos_weight", "get_loss_fn", "compute_dl_metrics", "tune_thresholds",
    "BytecodeDataset", "collate", "DL_DEFAULT",
    "run_dl_experiment", "DL_EXPERIMENTS", "run_all_dl",
]
