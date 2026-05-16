"""
HBA Char-LM - Hierarchical Boolean Attention  v2
=================================================

v2 変更点 (2026-05-10):
  - Best checkpoint 保存 + 学習終了後 restore (最優先)
  - Early stopping (patience=2 log_epochs)
  - warm_hold_frac 25% -> 50%
  - デフォルト n_layers=3, hidden_dim=96, n_heads=8 (~200K params)
  - Hard collapse 閾値キャリブレーション (val set で最良 threshold を探索)
  - log_every=2 で粒度上げ
  - CLI フラグ追加: --patience, --warm-hold, --auto-calibrate/--no-calibrate

設計書: meetings/2026-05-10_chiba-rebrainstorm-hba.md
診断:  meetings/2026-05-10_hba-v1-results.md

核心:
  - DLGN ルーター: 位置 i が位置 j を見るか -> {0,1} Boolean 決定
  - Float Value Aggregation: masked sum で float のまま集約 (誤差累積ゼロ)
  - Entropy 正則化: ルーターが "全部0" or "全部1" に collapse するのを防ぐ
  - Hard collapse: ルーター部分のみ collapse、value/FFN/embeddings は float

必須要件:
  - ルーター entropy 正則化を外さない (これがなければ同じ墓場)
  - head に spectral_norm
  - gradient clip max_norm=1.0
"""

import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_dlgn import (
  DynamicDLGNModel,
  TemperatureScheduler,
)

# ---------------------------------------------------------------------------
# 0. Data preparation
# ---------------------------------------------------------------------------

TINY_SHAKESPEARE_URL = (
  "https://raw.githubusercontent.com/karpathy/char-rnn/"
  "master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_CACHE = Path(__file__).parent / "tinyshakespeare.txt"
MAX_BYTES = 80_000

def load_text() -> str:
  if TINY_SHAKESPEARE_CACHE.exists():
  text = TINY_SHAKESPEARE_CACHE.read_text(encoding="utf-8")
  print(f"[data] cache hit: {len(text):,} chars")
  return text[:MAX_BYTES]
  print("[data] downloading TinyShakespeare...")
  try:
  with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=15) as r:
  raw = r.read().decode("utf-8")
  TINY_SHAKESPEARE_CACHE.write_text(raw, encoding="utf-8")
  print(f"[data] downloaded: {len(raw):,} chars -> using {MAX_BYTES:,}")
  return raw[:MAX_BYTES]
  except Exception as e:
  print(f"[data] download failed ({e}) -> synthetic fallback")
  return _make_synthetic_text(80_000)

def _make_synthetic_text(n: int) -> str:
  base = "abcdefghijklmnopqrstuvwxyz 0123456789\n"
  import random
  random.seed(42)
  chunks, cur = , "a"
  for _ in range(n):
  chunks.append(cur)
  idx = base.find(cur)
  cur = base[(idx + 1) % len(base)] if random.random() < 0.8 else random.choice(base)
  return "".join(chunks)

def build_vocab(text: str):
  chars = sorted(set(text))
  stoi = {c: i for i, c in enumerate(chars)}
  itos = {i: c for c, i in stoi.items()}
  return stoi, itos

def make_dataset(text: str, stoi: dict, context_len: int):
  ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
  n = len(ids) - context_len
  X = torch.stack([ids[i:i + context_len] for i in range(n)])
  Y = ids[context_len:]
  return X, Y

# ---------------------------------------------------------------------------
# 1. BooleanAttentionLayer
# ---------------------------------------------------------------------------

class BooleanAttentionLayer(nn.Module):
  """
  DLGN ルーター + float value aggregation の attention 層。

  ルーター:
  bilinear: Q[i] * K[j] で mask logit を生成
  -> sigmoid で [0,1] (soft mask, 学習時)
  -> causal mask 適用 (j > i は強制 0)
  -> masked weighted sum で attended を計算

  Float value:
  通常の value_proj (Linear) で value を作り、mask で weighted average
  誤差累積なし (float のまま集約)

  Entropy 正則化:
  mask (sigmoid出力) の binary entropy を計算して損失に加算
  H = -p*log(p+eps) - (1-p)*log(1-p+eps) を全ペアで平均
  get_mask_entropy() で forward 後に取得可能
  """

  def __init__(self, hidden_dim: int, n_heads: int = 8,
  router_hidden: list = None):
  super().__init__()
  self.hidden_dim = hidden_dim
  self.n_heads = n_heads
  self.head_dim = hidden_dim // n_heads

  if router_hidden is None:
  router_hidden = [hidden_dim, hidden_dim // 2]
  self._router_hidden = router_hidden

  # Bilinear router: Q x K で mask logit を生成
  self.q_proj = nn.Linear(hidden_dim, n_heads)
  self.k_proj = nn.Linear(hidden_dim, n_heads)

  # value projection: float (value 側は float のまま)
  self.value_proj = nn.Linear(hidden_dim, hidden_dim)
  self.out_proj = nn.Linear(hidden_dim, hidden_dim)

  # forward 後に entropy を参照できるようにキャッシュ
  self._last_mask_entropy: torch.Tensor = torch.tensor(0.0)

  def forward(self, x_emb: torch.Tensor, tau: float) -> torch.Tensor:
  """
  x_emb: (B, C, D)
  tau: temperature (低いほど Boolean に近づく)
  returns: attended (B, C, D)
  """
  B, C, D = x_emb.shape

  # Bilinear router
  Q = self.q_proj(x_emb)  # (B, C, H)
  K = self.k_proj(x_emb)  # (B, C, H)
  mask_logit = Q.unsqueeze(2) * K.unsqueeze(1)  # (B, C_i, C_j, H)

  # tau でスケール
  mask_logit_scaled = mask_logit / max(tau, 1e-3)
  mask_soft = torch.sigmoid(mask_logit_scaled)  # (B, C_i, C_j, H)

  # Causal mask
  causal = torch.tril(torch.ones(C, C, device=x_emb.device))
  causal = causal.unsqueeze(0).unsqueeze(-1)  # (1, C, C, 1)
  mask_soft = mask_soft * causal  # (B, C, C, H)

  # Entropy (正則化用)
  eps = 1e-6
  p = mask_soft
  h = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
  causal_count = causal.sum().item() * self.n_heads
  self._last_mask_entropy = h.sum() / max(1.0, causal_count)

  # Float Value Aggregation
  V = self.value_proj(x_emb)  # (B, C, D)
  V_heads = V.reshape(B, C, self.n_heads, self.head_dim)

  m = mask_soft.permute(0, 3, 1, 2)  # (B, H, C_i, C_j)
  V_h = V_heads.permute(0, 2, 1, 3)  # (B, H, C_j, head_dim)

  attended_heads = torch.matmul(m, V_h)  # (B, H, C_i, head_dim)
  mask_sum = m.sum(dim=-1, keepdim=True)
  attended_heads = attended_heads / (mask_sum + eps)

  attended = attended_heads.permute(0, 2, 1, 3)  # (B, C_i, H, head_dim)
  attended = attended.reshape(B, C, D)

  return self.out_proj(attended)

  def get_mask_entropy(self) -> torch.Tensor:
  return self._last_mask_entropy

# ---------------------------------------------------------------------------
# 2. HBALanguageModel (soft, 学習用)
# ---------------------------------------------------------------------------

class HBALanguageModel(nn.Module):
  """
  Embedding -> Sinusoidal PE -> BooleanAttentionLayer x N -> Final Linear

  各層の後に軽い FFN (通常の Linear, DLGN にしない)。
  最後の位置 (t = context_len-1) の hidden から logits を出す。

  head に spectral_norm 適用。
  """

  def __init__(
  self,
  vocab_size: int,
  context_len: int,
  hidden_dim: int = 96,
  n_layers: int = 3,
  n_heads: int = 8,
  router_hidden: list = None,
  ):
  super().__init__()
  self.vocab_size = vocab_size
  self.context_len = context_len
  self.hidden_dim = hidden_dim
  self.n_layers = n_layers
  self.n_heads = n_heads

  # Embedding
  self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
  pe = self._make_sinusoidal_pe(context_len, hidden_dim)
  self.register_buffer('pos_enc', pe)

  # BooleanAttentionLayer x N
  self.attn_layers = nn.ModuleList([
  BooleanAttentionLayer(hidden_dim, n_heads, router_hidden)
  for _ in range(n_layers)
  ])

  # FFN x N
  self.ffn_layers = nn.ModuleList([
  nn.Sequential(
  nn.Linear(hidden_dim, hidden_dim * 4),
  nn.GELU(),
  nn.Linear(hidden_dim * 4, hidden_dim),
  )
  for _ in range(n_layers)
  ])

  # LayerNorm
  self.attn_norms = nn.ModuleList([
  nn.LayerNorm(hidden_dim) for _ in range(n_layers)
  ])
  self.ffn_norms = nn.ModuleList([
  nn.LayerNorm(hidden_dim) for _ in range(n_layers)
  ])
  self.norm = nn.LayerNorm(hidden_dim)

  # Head (spectral_norm 適用: 指摘継承)
  self.head = nn.utils.spectral_norm(nn.Linear(hidden_dim, vocab_size))

  @staticmethod
  def _make_sinusoidal_pe(context_len: int, hidden_dim: int) -> torch.Tensor:
  pe = torch.zeros(context_len, hidden_dim)
  pos = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1)
  div = torch.exp(
  torch.arange(0, hidden_dim, 2, dtype=torch.float)
  * (-math.log(10000.0) / hidden_dim)
  )
  pe[:, 0::2] = torch.sin(pos * div)
  if hidden_dim % 2 == 0:
  pe[:, 1::2] = torch.cos(pos * div)
  else:
  pe[:, 1::2] = torch.cos(pos * div[:-1])
  return pe

  def forward(self, x_seq: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
  B, C = x_seq.shape
  h = self.tok_emb(x_seq) + self.pos_enc.unsqueeze(0)

  for i in range(self.n_layers):
  h_norm = self.attn_norms[i](h)
  attended = self.attn_layers[i](h_norm, tau)
  h = h + attended

  h_norm = self.ffn_norms[i](h)
  h = h + self.ffn_layers[i](h_norm)

  h_last = self.norm(h[:, -1, :])
  return self.head(h_last)

  def get_mask_entropies(self) -> list:
  return [layer.get_mask_entropy() for layer in self.attn_layers]

  def param_count(self) -> int:
  return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# 3. HardBooleanAttentionLayer (collapse 後)
# ---------------------------------------------------------------------------

class HardBooleanAttentionLayer(nn.Module):
  """
  BooleanAttentionLayer の collapse 版。
  bilinear router で mask logit を計算し、
  threshold で hard {0,1} mask に変換 (v2: threshold キャリブレーション対応)。
  value_proj / out_proj は float のまま維持。
  """

  def __init__(
  self,
  q_proj: nn.Linear,
  k_proj: nn.Linear,
  value_proj: nn.Linear,
  out_proj: nn.Linear,
  hidden_dim: int,
  n_heads: int,
  context_len: int,
  threshold: float = 0.0,
  ):
  super().__init__()
  self.q_proj = q_proj
  self.k_proj = k_proj
  self.value_proj = value_proj
  self.out_proj = out_proj
  self.hidden_dim = hidden_dim
  self.n_heads = n_heads
  self.head_dim = hidden_dim // n_heads
  self.context_len = context_len
  self.threshold = threshold

  causal = torch.tril(torch.ones(context_len, context_len))
  self.register_buffer('causal', causal)

  @torch.no_grad()
  def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
  B, C, D = x_emb.shape
  eps = 1e-6

  Q = self.q_proj(x_emb)
  K = self.k_proj(x_emb)
  mask_logit = Q.unsqueeze(2) * K.unsqueeze(1)  # (B, C, C, H)

  # v2: threshold でキャリブレーション
  mask_hard = (mask_logit > self.threshold).float()

  causal = self.causal.unsqueeze(0).unsqueeze(-1)
  mask_hard = mask_hard * causal

  V = self.value_proj(x_emb)
  V_heads = V.reshape(B, C, self.n_heads, self.head_dim)

  m = mask_hard.permute(0, 3, 1, 2)
  V_h = V_heads.permute(0, 2, 1, 3)

  attended_heads = torch.matmul(m, V_h)
  mask_sum = m.sum(dim=-1, keepdim=True)
  attended_heads = attended_heads / (mask_sum + eps)

  attended = attended_heads.permute(0, 2, 1, 3).reshape(B, C, D)
  return self.out_proj(attended)

# ---------------------------------------------------------------------------
# 4. HardHBALanguageModel (collapse 後の推論専用)
# ---------------------------------------------------------------------------

class HardHBALanguageModel(nn.Module):
  """
  HBALanguageModel の collapse 版。
  router のみ hard {0,1}、value/FFN/embeddings は float のまま。
  """

  def __init__(
  self,
  tok_emb: nn.Embedding,
  pos_enc: torch.Tensor,
  hard_attn_layers: nn.ModuleList,
  ffn_layers: nn.ModuleList,
  attn_norms: nn.ModuleList,
  ffn_norms: nn.ModuleList,
  norm: nn.LayerNorm,
  head: nn.Linear,
  vocab_size: int,
  context_len: int,
  ):
  super().__init__()
  self.tok_emb = tok_emb
  self.register_buffer('pos_enc', pos_enc)
  self.hard_attn_layers = hard_attn_layers
  self.ffn_layers = ffn_layers
  self.attn_norms = attn_norms
  self.ffn_norms = ffn_norms
  self.norm = norm
  self.head = head
  self.vocab_size = vocab_size
  self.context_len = context_len
  self.n_layers = len(hard_attn_layers)

  @torch.no_grad()
  def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
  B, C = x_seq.shape
  h = self.tok_emb(x_seq) + self.pos_enc.unsqueeze(0)

  for i in range(self.n_layers):
  h_norm = self.attn_norms[i](h)
  attended = self.hard_attn_layers[i](h_norm)
  h = h + attended

  h_norm = self.ffn_norms[i](h)
  h = h + self.ffn_layers[i](h_norm)

  h_last = self.norm(h[:, -1, :])
  return self.head(h_last)

  def param_count(self) -> int:
  return sum(p.numel() for p in self.parameters())

def collapse_hba(
  model: HBALanguageModel,
  threshold: float = 0.0,
) -> HardHBALanguageModel:
  """HBALanguageModel -> HardHBALanguageModel への変換
  bilinear router を引き継ぎ、forward で threshold で hard {0,1} を生成。
  """
  hard_attn_layers = nn.ModuleList()
  for layer in model.attn_layers:
  hard_layer = HardBooleanAttentionLayer(
  q_proj=layer.q_proj,
  k_proj=layer.k_proj,
  value_proj=layer.value_proj,
  out_proj=layer.out_proj,
  hidden_dim=layer.hidden_dim,
  n_heads=layer.n_heads,
  context_len=model.context_len,
  threshold=threshold,
  )
  hard_attn_layers.append(hard_layer)

  return HardHBALanguageModel(
  tok_emb=model.tok_emb,
  pos_enc=model.pos_enc,
  hard_attn_layers=hard_attn_layers,
  ffn_layers=model.ffn_layers,
  attn_norms=model.attn_norms,
  ffn_norms=model.ffn_norms,
  norm=model.norm,
  head=model.head,
  vocab_size=model.vocab_size,
  context_len=model.context_len,
  )

# ---------------------------------------------------------------------------
# 5. Hard threshold calibration (v2 新規)
# ---------------------------------------------------------------------------

def calibrate_hard_threshold(
  model: HBALanguageModel,
  X_val: torch.Tensor,
  Y_val: torch.Tensor,
  batch_size: int = 256,
  thresholds: list = None,
) -> tuple:
  """
  val set で各 threshold の hard PPL を測定し、最良 threshold を返す。

  Returns:
  best_threshold (float), best_hard_ppl (float), results (list of (thresh, ppl))
  """
  if thresholds is None:
  thresholds = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]

  print("\n[calibrate] Hard threshold calibration on val set:")
  print(f"  {'Threshold':>10} {'Hard PPL':>10}")
  print(f"  {'-' * 22}")

  results = 
  best_threshold = 0.0
  best_hard_ppl = float('inf')

  for thresh in thresholds:
  hard_model = collapse_hba(model, threshold=thresh)
  hard_model.eval()

  total_loss, n_b = 0.0, 0
  with torch.no_grad():
  for i in range(0, len(X_val), batch_size):
  xb = X_val[i:i + batch_size]
  yb = Y_val[i:i + batch_size]
  logits = hard_model(xb)
  total_loss += F.cross_entropy(logits, yb).item()
  n_b += 1
  ppl = math.exp(total_loss / max(1, n_b))
  results.append((thresh, ppl))
  print(f"  {thresh:>10.2f} {ppl:>10.4f}")

  if ppl < best_hard_ppl:
  best_hard_ppl = ppl
  best_threshold = thresh

  print(f"  -> Best threshold: {best_threshold:.2f}  Hard PPL: {best_hard_ppl:.4f}")
  return best_threshold, best_hard_ppl, results

# ---------------------------------------------------------------------------
# 6. Perplexity / throughput utilities
# ---------------------------------------------------------------------------

def compute_perplexity(
  model, X_val: torch.Tensor, Y_val: torch.Tensor,
  batch_size: int = 256, temperature: float = None
) -> float:
  model.eval()
  total_loss, n_batches = 0.0, 0
  with torch.no_grad():
  for i in range(0, len(X_val), batch_size):
  xb = X_val[i:i + batch_size]
  yb = Y_val[i:i + batch_size]
  if temperature is not None:
  logits = model(xb, tau=temperature)
  else:
  logits = model(xb)
  total_loss += F.cross_entropy(logits, yb).item()
  n_batches += 1
  return math.exp(total_loss / max(1, n_batches))

def measure_tokens_per_sec(
  model, X_sample: torch.Tensor,
  use_temp: bool = False, tau: float = 0.1,
  runs: int = 20,
) -> float:
  model.eval()
  with torch.no_grad():
  for _ in range(3):
  if use_temp:
  model(X_sample, tau=tau)
  else:
  model(X_sample)
  t0 = time.perf_counter()
  for _ in range(runs):
  if use_temp:
  model(X_sample, tau=tau)
  else:
  model(X_sample)
  return X_sample.shape[0] * runs / (time.perf_counter() - t0)

# ---------------------------------------------------------------------------
# 7. Training loop v2 (Best checkpoint + Early stopping)
# ---------------------------------------------------------------------------

def train_hba(
  model: HBALanguageModel,
  X_train: torch.Tensor, Y_train: torch.Tensor,
  X_val: torch.Tensor, Y_val: torch.Tensor,
  epochs: int = 60,
  batch_size: int = 256,
  lr: float = 3e-3,
  lambda_ent: float = 0.01,
  temp_scheduler: TemperatureScheduler = None,
  log_every: int = 2,
  patience: int = 2,
) -> tuple:
  """
  HBA 専用学習ループ v2。

  v2 追加:
  - Best checkpoint 保存 (val PPL がベストなら state_dict をメモリに保存)
  - Early stopping (patience: ベスト更新なしが続いたら打ち切り)

  loss = F.cross_entropy(logits, y) - lambda_ent * mean_router_entropy

  Returns:
  ppl_history: list of (epoch, val_ppl)
  best_ppl: float
  best_epoch: int
  """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  ppl_history = 
  best_ppl = float('inf')
  best_state = None
  best_epoch = -1
  no_improve_count = 0  # patience カウンタ (log_epoch 単位)

  print(f"\n  {'Ep':>4} {'tau':>6} {'CE':>8} {'EntReg':>9} {'TotLoss':>9} "
  f"{'ValPPL':>9} {'AvgMaskH':>10} {'Best?':>6}")
  print(f"  {'-' * 70}")

  for epoch in range(epochs):
  tau = temp_scheduler.temperature if temp_scheduler else 1.0
  model.train()
  perm = torch.randperm(len(X_train))
  X_s, Y_s = X_train[perm], Y_train[perm]

  ep_ce, ep_ent_reg, ep_total, n_b = 0.0, 0.0, 0.0, 0

  for i in range(0, len(X_train), batch_size):
  xb = X_s[i:i + batch_size]
  yb = Y_s[i:i + batch_size]
  optimizer.zero_grad()

  logits = model(xb, tau=tau)
  ce = F.cross_entropy(logits, yb)

  # Entropy 正則化
  entropies = model.get_mask_entropies()
  mean_entropy = torch.stack(entropies).mean()
  ent_reg = -lambda_ent * mean_entropy

  loss = ce + ent_reg

  loss.backward()
  # gradient clip
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()

  ep_ce += ce.item()
  ep_ent_reg += ent_reg.item()
  ep_total += loss.item()
  n_b += 1

  lr_sched.step()
  if temp_scheduler:
  temp_scheduler.step()

  if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
  tau_disp = temp_scheduler.temperature if temp_scheduler else 1.0
  ppl = compute_perplexity(model, X_val, Y_val, temperature=tau_disp)
  ppl_history.append((epoch + 1, ppl))

  with torch.no_grad():
  _ = model(X_val[:min(256, len(X_val))], tau=tau_disp)
  entropies = model.get_mask_entropies()
  avg_mask_h = torch.stack(entropies).mean().item()

  # Best checkpoint 保存
  is_best = ppl < best_ppl
  if is_best:
  best_ppl = ppl
  best_epoch = epoch + 1
  # detach().clone() でメモリに保存 (GPU -> CPU も対応)
  best_state = {
  k: v.detach().clone()
  for k, v in model.state_dict().items()
  }
  no_improve_count = 0
  else:
  no_improve_count += 1

  best_mark = " *" if is_best else ""
  print(
  f"  {epoch+1:>4} {tau_disp:>6.3f} "
  f"{ep_ce/n_b:>8.4f} {ep_ent_reg/n_b:>9.4f} "
  f"{ep_total/n_b:>9.4f} {ppl:>9.2f} {avg_mask_h:>10.6f}{best_mark}",
  flush=True,
  )

  # Early stopping チェック
  if patience > 0 and no_improve_count >= patience:
  print(
  f"\n  [early stop] no improvement for {patience} log-epochs "
  f"(best={best_ppl:.4f} @ ep{best_epoch}). Stopping."
  )
  break

  # Best checkpoint を restore
  if best_state is not None:
  model.load_state_dict(best_state)
  print(f"\n  [best] restored from epoch {best_epoch}  PPL={best_ppl:.4f}")
  else:
  print("\n  [best] no checkpoint saved (unexpected)")

  return ppl_history, best_ppl, best_epoch

# ---------------------------------------------------------------------------
# 8. Text generation
# ---------------------------------------------------------------------------

def generate_text(
  model, seed_text: str, stoi: dict, itos: dict,
  context_len: int, n_chars: int = 80,
  temperature: float = 1.0,
  use_model_tau: bool = False,
  tau: float = 0.1,
) -> str:
  model.eval()
  unk_id = stoi.get(' ', 0)
  seed_ids = [stoi.get(c, unk_id) for c in seed_text]
  if len(seed_ids) < context_len:
  seed_ids = [unk_id] * (context_len - len(seed_ids)) + seed_ids
  ctx = seed_ids[-context_len:]

  result = list(seed_text[-context_len:])
  with torch.no_grad():
  for _ in range(n_chars):
  x = torch.tensor([ctx], dtype=torch.long)
  if use_model_tau:
  logits = model(x, tau=max(0.1, tau))
  else:
  logits = model(x)

  if temperature <= 0.0:
  next_id = logits.argmax(dim=-1).item()
  else:
  probs = F.softmax(logits[0] / temperature, dim=-1)
  next_id = torch.multinomial(probs, 1).item()

  result.append(itos[next_id])
  ctx = ctx[1:] + [next_id]

  return "".join(result)

# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
  torch.manual_seed(42)

  print("=" * 68)
  print("  HBA Char-LM v2 - Hierarchical Boolean Attention")
  print(f"  Device: CPU  PyTorch {torch.__version__}")
  print("=" * 68)

  # ── CLI 引数 ──────────────────────────────────────────────────────────
  EPOCHS  = 60
  HIDDEN_DIM  = 96  # v2: 64 -> 96
  N_LAYERS  = 3  # v2: 2 -> 3
  N_HEADS  = 8  # v2: 4 -> 8
  LAMBDA_ENT  = 0.01
  PATIENCE  = 2  # v2 新規: early stopping patience (log_epoch 単位)
  WARM_HOLD  = 0.5  # v2: 0.25 -> 0.50
  LOG_EVERY  = 2  # v2: 6 -> 2 (粒度上げ)
  AUTO_CALIBRATE = True  # v2 新規: hard threshold calibration

  for arg in sys.argv[1:]:
  if arg.startswith('--epochs='):
  EPOCHS = int(arg.split('=', 1)[1])
  print(f"[CLI] epochs: {EPOCHS}")
  elif arg.startswith('--hidden-dim='):
  HIDDEN_DIM = int(arg.split('=', 1)[1])
  print(f"[CLI] hidden_dim: {HIDDEN_DIM}")
  elif arg.startswith('--n-layers='):
  N_LAYERS = int(arg.split('=', 1)[1])
  print(f"[CLI] n_layers: {N_LAYERS}")
  elif arg.startswith('--n-heads='):
  N_HEADS = int(arg.split('=', 1)[1])
  print(f"[CLI] n_heads: {N_HEADS}")
  elif arg.startswith('--lambda-ent='):
  LAMBDA_ENT = float(arg.split('=', 1)[1])
  print(f"[CLI] lambda_ent: {LAMBDA_ENT}")
  elif arg.startswith('--patience='):
  PATIENCE = int(arg.split('=', 1)[1])
  print(f"[CLI] patience: {PATIENCE}")
  elif arg.startswith('--warm-hold='):
  WARM_HOLD = float(arg.split('=', 1)[1])
  print(f"[CLI] warm_hold: {WARM_HOLD}")
  elif arg == '--no-calibrate':
  AUTO_CALIBRATE = False
  print("[CLI] auto_calibrate: OFF")
  elif arg == '--auto-calibrate':
  AUTO_CALIBRATE = True
  print("[CLI] auto_calibrate: ON")

  BATCH  = 256
  LR  = 3e-3
  CONTEXT_LEN = 16

  # ── Data ─────────────────────────────────────────────────────────────
  text = load_text()
  stoi, itos = build_vocab(text)
  vocab_size = len(stoi)
  print(f"\n[vocab] size={vocab_size}  text_len={len(text):,}")

  split = int(len(text) * 0.9)
  X_train, Y_train = make_dataset(text[:split], stoi, CONTEXT_LEN)
  X_val,  Y_val  = make_dataset(text[split:],  stoi, CONTEXT_LEN)
  print(f"[data] train={len(X_train):,}  val={len(X_val):,}  "
  f"context_len={CONTEXT_LEN}")

  # ── Model ─────────────────────────────────────────────────────────────
  router_hidden = [HIDDEN_DIM, HIDDEN_DIM // 2]

  model = HBALanguageModel(
  vocab_size=vocab_size,
  context_len=CONTEXT_LEN,
  hidden_dim=HIDDEN_DIM,
  n_layers=N_LAYERS,
  n_heads=N_HEADS,
  router_hidden=router_hidden,
  )

  print(f"\n[HBA v2] params: {model.param_count():,}")
  print(f"  hidden_dim={HIDDEN_DIM}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
  print(f"  router_hidden={router_hidden}")
  print(f"  patience={PATIENCE} log-epochs  warm_hold={WARM_HOLD}")
  print(f"  log_every={LOG_EVERY}  auto_calibrate={AUTO_CALIBRATE}")

  # ── Temperature scheduler ─────────────────────────────────────────────
  temp_sched = TemperatureScheduler(
  t_start=1.0, t_end=0.1,
  total_epochs=EPOCHS,
  warm_hold_frac=WARM_HOLD,
  )

  # ── Training ──────────────────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print(f"  HBA v2 training  epochs={EPOCHS}  batch={BATCH}  lr={LR}")
  print(f"  Temperature: 1.0 -> 0.1  (warm_hold={WARM_HOLD*100:.0f}%)")
  print(f"  Entropy regularization: lambda_ent={LAMBDA_ENT}")
  print(f"  Early stopping: patience={PATIENCE} log-epochs "
  f"({'disabled' if PATIENCE == 0 else 'enabled'})")
  print(f"{'=' * 68}")

  t0 = time.perf_counter()
  ppl_history, best_ppl, best_epoch = train_hba(
  model, X_train, Y_train, X_val, Y_val,
  epochs=EPOCHS,
  batch_size=BATCH,
  lr=LR,
  lambda_ent=LAMBDA_ENT,
  temp_scheduler=temp_sched,
  log_every=LOG_EVERY,
  patience=PATIENCE,
  )
  train_time = time.perf_counter() - t0
  print(f"\n  Training time: {train_time:.1f}s ({train_time/60:.1f}min)")
  print(f"  Best epoch: {best_epoch}  Best val PPL: {best_ppl:.4f}")

  # ── Final soft PPL (best restored モデルで評価) ───────────────────
  final_tau = temp_sched.temperature
  final_ppl_soft = compute_perplexity(model, X_val, Y_val, temperature=final_tau)

  # ── 平均ルーター活性化率 ──────────────────────────────────────────
  with torch.no_grad():
  _ = model(X_val[:256], tau=final_tau)
  entropies = model.get_mask_entropies()
  final_avg_entropy = torch.stack(entropies).mean().item()

  print(f"\n[Router Stats]")
  print(f"  Final mask entropy (best model): {final_avg_entropy:.6f}")
  print(f"  (0 に近い = collapse、{math.log(2):.4f} 付近 = 最大多様性)")

  collapse_thresh = 0.01
  if final_avg_entropy < collapse_thresh:
  print(f"  WARNING: ルーター collapse の疑い (entropy < {collapse_thresh})")
  else:
  print(f"  OK: ルーター entropy は健全範囲")

  # ── Inference speed (soft) ────────────────────────────────────────
  sample_x = X_val[:512]
  speed_soft = measure_tokens_per_sec(
  model, sample_x, use_temp=True, tau=final_tau
  )

  # ── Hard threshold calibration (v2 新規) ──────────────────────────
  if AUTO_CALIBRATE:
  best_threshold, best_hard_ppl_calib, calib_results = calibrate_hard_threshold(
  model, X_val, Y_val, batch_size=BATCH
  )
  print(f"  [calibrate] Adopted threshold: {best_threshold:.2f}  "
  f"Hard PPL: {best_hard_ppl_calib:.4f}")
  else:
  best_threshold = 0.0
  print(f"  [calibrate] Skipped (--no-calibrate). Using threshold=0.0")

  # ── Hard collapse (キャリブレーション済み threshold で) ───────────
  print(f"\n[collapse] HBA -> HardHBA "
  f"(router {{0,1}}, threshold={best_threshold:.2f})...")
  hard_model = collapse_hba(model, threshold=best_threshold)
  hard_model.eval()

  speed_hard = measure_tokens_per_sec(hard_model, sample_x, use_temp=False)

  # Hard PPL (最終確認)
  total_loss, n_b = 0.0, 0
  with torch.no_grad():
  for i in range(0, len(X_val), BATCH):
  xb = X_val[i:i + BATCH]
  yb = Y_val[i:i + BATCH]
  logits = hard_model(xb)
  total_loss += F.cross_entropy(logits, yb).item()
  n_b += 1
  ppl_hard = math.exp(total_loss / max(1, n_b))

  # ── Text generation ───────────────────────────────────────────────
  seeds = ["First ", "To be ", "The ki"]

  print(f"\n{'=' * 68}")
  print("  Generated text: HBA v2 soft - greedy")
  print(f"{'=' * 68}")
  for seed in seeds:
  gen = generate_text(
  model, seed, stoi, itos, CONTEXT_LEN,
  n_chars=60, temperature=0.0, use_model_tau=True, tau=final_tau,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 68}")
  print("  Generated text: HBA v2 soft - sampled (temp=0.8)")
  print(f"{'=' * 68}")
  for seed in seeds:
  gen = generate_text(
  model, seed, stoi, itos, CONTEXT_LEN,
  n_chars=60, temperature=0.8, use_model_tau=True, tau=final_tau,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 68}")
  print(f"  Generated text: HBA v2 Hard (threshold={best_threshold:.2f})")
  print(f"{'=' * 68}")
  for seed in seeds:
  gen = generate_text(
  hard_model, seed, stoi, itos, CONTEXT_LEN,
  n_chars=60, temperature=0.8, use_model_tau=False,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  # ── Comparison table ──────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print("  HBA v2 vs HBA v1 vs DLGN flat vs Transformer - 比較表")
  print(f"{'=' * 68}")
  print(f"  {'Model':<32} {'Params':>8} {'ValPPL':>8} {'HardPPL':>9} {'Speed(tok/s)':>14}")
  print(f"  {'-' * 77}")
  print(f"  {'HBA v2 (soft, best restored)':<32} {model.param_count():>8,} "
  f"{final_ppl_soft:>8.2f} {'-':>9} {speed_soft:>14,.0f}")
  print(f"  {'HBA v2 (hard, thresh=' + str(best_threshold) + ')':<32} {model.param_count():>8,} "
  f"{'-':>8} {ppl_hard:>9.2f} {speed_hard:>14,.0f}")
  # v1 ベスト (Ep12)
  print(f"  {'HBA v1 Best (Ep12) [前回]':<32} {'92,000':>8} "
  f"{'5.40':>8} {'37.47':>9} {'-':>14}")
  print(f"  {'HBA v1 Final (Ep60) [前回]':<32} {'92,000':>8} "
  f"{'9.75':>8} {'37.47':>9} {'22,828':>14}")
  print(f"  {'DLGN flat (soft) [前回]':<32} {'1,362,813':>8} "
  f"{'11.83':>8} {'15.16':>9} {'18,299':>14}")
  print(f"  {'Transformer [前回]':<32} {'108,861':>8} "
  f"{'4.86':>8} {'-':>9} {'51,291':>14}")

  # ── Perplexity curve ──────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print("  Perplexity curve (HBA v2, log_every=2)")
  print(f"{'=' * 68}")
  print(f"  {'Epoch':>6} {'ValPPL':>10} {'Note':>8}")
  print(f"  {'-' * 28}")
  for ep, ppl in ppl_history:
  note = " <- BEST" if ep == best_epoch else ""
  print(f"  {ep:>6} {ppl:>10.2f}{note}")

  # ── 判定 ──────────────────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print(f"{'=' * 68}")

  if final_ppl_soft < 4.86:
  verdict = "HBA v2 が Transformer を超えた！本命 HBA 勝利"
  elif final_ppl_soft < 5.40:
  verdict = "HBA v2 が v1 Best を超えた (TF まで あと一歩)"
  elif final_ppl_soft < 7.0:
  verdict = "HBA が TF に肉薄 (LoopedDLGN 全敗から大幅改善)"
  elif final_ppl_soft < 11.83:
  verdict = "HBA が DLGN flat より改善 (Boolean Attention 有効)"
  else:
  verdict = "HBA と DLGN flat 同等 - アーキ改善継続が必要"

  print(f"  判定: {verdict}")
  print(f"")
  print(f"  HBA v2 soft PPL (best restored): {final_ppl_soft:.2f}")
  print(f"  HBA v2 hard PPL (thresh={best_threshold:.2f}):  {ppl_hard:.2f}")
  print(f"  HBA v1 Best PPL [前回 Ep12]:  5.40")
  print(f"  DLGN flat PPL [前回]:  11.83")
  print(f"  Transformer PPL [前回]:  4.86")
  print(f"")
  print(f"  Best epoch: {best_epoch}  (early stopping patience={PATIENCE})")
  print(f"  Final mask entropy: {final_avg_entropy:.6f}")
  print(f"  (ルーター collapse {'あり' if final_avg_entropy < 0.01 else 'なし'})")
  print(f"  Adopted hard threshold: {best_threshold:.2f}")
  print(f"  Soft speed:  {speed_soft:,.0f} tok/s")
  print(f"  Hard speed:  {speed_hard:,.0f} tok/s  "
  f"({speed_hard / speed_soft:.2f}x soft)")
  print(f"  Hard vs TF [前回]: {speed_hard / 51291:.2f}x")
  print()
  print(f"  - ルーター entropy 正則化: 適用済み (lambda_ent={LAMBDA_ENT:.4f})")
  print("  - mean pooling 廃止: 各位置が独立保持 -> attention で選択的集約")
  print("  - Looped 反復なし: 誤差累積ゼロ")
  print(f"  [v2 改善]")
  print(f"  - Best checkpoint: 保存 + restore 済 (ep{best_epoch}, PPL={best_ppl:.4f})")
  print(f"  - Early stopping: patience={PATIENCE} log-epochs")
  print(f"  - warm_hold={WARM_HOLD}  log_every={LOG_EVERY}")
  print(f"  - Hard threshold calibration: {'ON' if AUTO_CALIBRATE else 'OFF'} "
  f"(best={best_threshold:.2f})")
  print()

if __name__ == "__main__":
  main()
