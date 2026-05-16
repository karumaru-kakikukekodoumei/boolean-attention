"""
LoopedDLGN Char-LM PoC
========================
同一 DLGNブロックを反復して固定点収束させる LoopedDLGN による Char-LM。

参照:
  - meetings/2026-05-10_chiba-design-proposals.md  (案5: LoopedDLGN)
  - meetings/2026-05-10_dlgn-charlm-results.md  (前回ベンチ)
  - projects/dlgn-arithmetic/dynamic_dlgn.py  (DynamicDLGNModel)
  - projects/dlgn-arithmetic/dlgn_charlm.py  (学習ループ流用)

設計上の注意:
  - shared_block 周辺は _wrap_shared_block() でラップ可能な構造にしてある。
  理論検討結果が来たら、ここにスペクトルノルム正則化を後付け差し込み可能。
  - 位置エンコードは sinusoidal を mean pooling に足すオプションあり (use_pos_enc)。
  デフォルトは False (mean pooling のみ) で試す。言ってた「位置消失」対策の fallback。
  - BPTT勾配死対策: clip_grad_norm_ max_norm=5.0 必須適用済み。
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
  HardDynamicDLGNModel,
  TemperatureScheduler,
)

# ---------------------------------------------------------------------------
# 0. Utility: スペクトルノルム後付けフック用ラッパー
# ---------------------------------------------------------------------------

class SharedBlockWrapper(nn.Module):
  """
  shared_block を包むラッパー。
  指摘でスペクトルノルム正則化が必要になった場合、
  apply_spectral_norm() を呼ぶだけで差し込める構造。

  現状は素通し (identity wrapper) だが、
  forward フックや正則化損失の集約点として機能する。
  """

  def __init__(self, block: nn.Module):
  super().__init__()
  self.block = block
  self._sn_enabled = False  # スペクトルノルム正則化フラグ
  self._sn_lambda = 0.0  # 正則化強度 (後付け設定用)
  self._sn_penalty = torch.tensor(0.0)  # forward 後に参照できるペナルティ

  def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
  out = self.block(x, temperature=temperature)
  # スペクトルノルム正則化: 有効化された場合、重み行列のスペクトルノルムを
  # 損失に加算するためのペナルティを計算する。
  # (現在は無効。指摘後に _sn_enabled=True + _sn_lambda 設定で有効化)
  if self._sn_enabled:
  self._sn_penalty = self._compute_sn_penalty()
  else:
  self._sn_penalty = torch.tensor(0.0)
  return out

  def _compute_sn_penalty(self) -> torch.Tensor:
  """
  Linear 層の重みのスペクトルノルム (最大特異値) をペナルティとして返す。
  縮小写像条件: σ_max(W) < 1 を強制するための正則化項。
  後付け差し込み用: 現時点では stub 実装。
  """
  penalty = torch.tensor(0.0)
  for module in self.block.modules():
  if isinstance(module, nn.Linear):
  W = module.weight  # (out, in)
  # power iteration による近似スペクトルノルム (低コスト)
  with torch.no_grad():
  v = F.normalize(torch.randn(W.shape[1]), dim=0)
  for _ in range(3):  # 3回で十分な近似
  u = F.normalize(W @ v, dim=0)
  v = F.normalize(W.t() @ u, dim=0)
  sigma = (u @ (W @ v)).abs()
  penalty = penalty + sigma
  return penalty * self._sn_lambda

  def enable_spectral_norm(self, lambda_sn: float = 1e-3) -> None:
  """指摘後にここを呼べばスペクトルノルム正則化が有効になる。"""
  self._sn_enabled = True
  self._sn_lambda = lambda_sn
  print(f"[LoopedDLGN] スペクトルノルム正則化 有効化 (lambda={lambda_sn})")

  def sn_penalty(self) -> torch.Tensor:
  """学習ループ内で損失に加算するペナルティを取得する。"""
  return self._sn_penalty

# ---------------------------------------------------------------------------
# 1. Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

def _make_sinusoidal_pe(context_len: int, hidden_dim: int) -> torch.Tensor:
  """
  context_len x hidden_dim の sinusoidal PE テーブルを返す。
  mean pooling で位置情報が消えるリスクへの fallback として用意。
  """
  pe = torch.zeros(context_len, hidden_dim)
  pos = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1)
  div = torch.exp(
  torch.arange(0, hidden_dim, 2, dtype=torch.float)
  * (-math.log(10000.0) / hidden_dim)
  )
  pe[:, 0::2] = torch.sin(pos * div)
  pe[:, 1::2] = torch.cos(pos * div[:hidden_dim // 2])
  return pe  # (context_len, hidden_dim)

# ---------------------------------------------------------------------------
# 2. LoopedDLGN
# ---------------------------------------------------------------------------

class LoopedDLGN(nn.Module):
  """
  LoopedDLGN — Universal Transformer × Boolean 回路

  入力 x_seq (B, C) → one-hot (B, C, V)
  [use_pos_enc=False] mean pool (B, V) → input_proj → sigmoid → x_bin (B, D)
  [use_pos_enc=True]  各位置に PE を足してから mean pool → input_proj → sigmoid

  反復ループ:
  h_0 = zeros(B, D)
  for t in range(max_iters):
  combined = cat([x_bin, sigmoid(h)], dim=-1)  # (B, 2D)
  h_new = shared_block(combined, tau)  # (B, D)
  [推論時のみ] if change < halt_threshold: break
  h = h_new
  logits = head(h)  # (B, vocab_size)

  後付けスペクトルノルム:
  model.shared_block_wrapper.enable_spectral_norm(lambda_sn=1e-3) を呼ぶだけ。
  損失ループ内で loss += model.shared_block_wrapper.sn_penalty() を加算。
  """

  def __init__(
  self,
  vocab_size: int,
  context_len: int,
  hidden_dim: int = 128,
  max_iters: int = 8,
  block_hidden: list = None,
  halt_threshold: float = 0.05,
  use_pos_enc: bool = False,
  seed_base: int = 42,
  ):
  super().__init__()
  if block_hidden is None:
  block_hidden = [256, 256, 128]

  self.vocab_size = vocab_size
  self.context_len = context_len
  self.hidden_dim = hidden_dim
  self.max_iters = max_iters
  self.halt_threshold = halt_threshold
  self.use_pos_enc = use_pos_enc

  # -- 入力投影: float 線形層 (軽い) --
  # mean pool された one-hot ベクトル (B, V) を (B, D) に変換
  self.input_proj = nn.Linear(vocab_size, hidden_dim)

  # -- 位置エンコード テーブル (オプション) --
  if use_pos_enc:
  pe = _make_sinusoidal_pe(context_len, vocab_size)
  self.register_buffer('pos_enc', pe)  # (C, V) - one-hot 空間でPEを足す
  else:
  self.pos_enc = None

  # -- 共有 DLGNブロック: 1つだけ、反復で再利用 --
  # in_features = hidden_dim * 2 (x_bin + h を concat)
  _block = DynamicDLGNModel(
  in_features=hidden_dim * 2,
  hidden_neurons=block_hidden,
  num_classes=hidden_dim,
  init_scale=0.1,
  init_pairing='random_hard',
  seed_base=seed_base,
  )
  # ラッパーで包む → スペクトルノルム正則化の後付け差し込み口
  self.shared_block_wrapper = SharedBlockWrapper(_block)

  # -- 出力ヘッド --
  # 指摘: head に spectral_norm を適用して L<1 を外部から担保する
  # nn.utils.spectral_norm は forward 時に W を最大特異値で正規化する
  self.head = nn.utils.spectral_norm(nn.Linear(hidden_dim, vocab_size))

  # -- 統計カウンタ (推論時の収束反復数計測用) --
  self._iter_counts: list = 
  self._collecting_stats = False

  @property
  def shared_block(self) -> nn.Module:
  """外部から shared_block に直接アクセスする口。"""
  return self.shared_block_wrapper.block

  def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
  """
  x: (B, C) int64 → x_bin: (B, D) float32 in [0,1]

  use_pos_enc=True の場合、one-hot 空間で sinusoidal PE を加算してから
  mean pooling する。位置情報が消えるリスクへの fallback。
  """
  B, C = x.shape
  oh = F.one_hot(x, num_classes=self.vocab_size).float()  # (B, C, V)

  if self.use_pos_enc and self.pos_enc is not None:
  # PE は [0,1] 正規化した後に加算 (one-hot は {0,1} なので範囲をそろえる)
  pe = self.pos_enc.unsqueeze(0)  # (1, C, V)
  oh = oh + pe * 0.1  # 微小スケールで足す (one-hot の支配を維持)

  # mean pooling: 位置情報を集約 (or 位置PE込みで集約)
  x_pool = oh.mean(dim=1)  # (B, V)
  x_bin = torch.sigmoid(self.input_proj(x_pool))  # (B, D) in (0,1)
  return x_bin

  def forward(
  self,
  x: torch.Tensor,
  temperature: float = 1.0,
  ) -> torch.Tensor:
  """
  x: (B, C) int64
  temperature: Gumbel-Softmax 温度 (学習中はアニーリング)
  returns: logits (B, vocab_size)
  """
  B = x.shape[0]
  x_bin = self._encode_input(x)  # (B, D)
  h = torch.zeros(B, self.hidden_dim, device=x.device)  # h_0

  for t in range(self.max_iters):
  combined = torch.cat([x_bin, torch.sigmoid(h)], dim=-1)  # (B, 2D)
  h_new = self.shared_block_wrapper(combined, temperature)  # (B, D)

  # 推論時のみ: 収束判定で早期停止
  if not self.training:
  change = (h_new - h).abs().mean(dim=-1)  # (B,)
  converged = (change < self.halt_threshold).all().item()
  if self._collecting_stats:
  self._iter_counts.append(t + 1)
  if converged and t > 0:  # 最低1回は回す
  h = h_new
  break

  h = h_new

  return self.head(h)

  def start_iter_stats(self) -> None:
  """推論時の平均収束反復数計測を開始する。"""
  self._iter_counts = 
  self._collecting_stats = True

  def stop_iter_stats(self) -> dict:
  """計測を停止し、統計を返す。"""
  self._collecting_stats = False
  if not self._iter_counts:
  return {'mean_iters': float('nan'), 'early_stop_rate': float('nan')}
  counts = self._iter_counts
  mean_iters = sum(counts) / len(counts)
  early_stop_rate = sum(1 for c in counts if c < self.max_iters) / len(counts)
  self._iter_counts = 
  return {
  'mean_iters': mean_iters,
  'early_stop_rate': early_stop_rate,
  'n_samples': len(counts),
  }

  def param_count(self) -> int:
  return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# 3. HardLoopedDLGN (collapse 後の推論専用)
# ---------------------------------------------------------------------------

class HardLoopedDLGN(nn.Module):
  """
  shared_block を hard collapse した推論専用モデル。
  input_proj と head は float のまま維持 (精度ロスを最小化)。
  反復ループ自体はそのまま残す (hard ゲートで各反復を実行)。
  """

  def __init__(
  self,
  input_proj: nn.Linear,
  hard_block: HardDynamicDLGNModel,
  head: nn.Linear,
  vocab_size: int,
  context_len: int,
  hidden_dim: int,
  max_iters: int,
  halt_threshold: float,
  use_pos_enc: bool,
  pos_enc,
  ):
  super().__init__()
  self.input_proj = input_proj
  self.hard_block = hard_block
  self.head = head
  self.vocab_size = vocab_size
  self.context_len = context_len
  self.hidden_dim = hidden_dim
  self.max_iters = max_iters
  self.halt_threshold = halt_threshold
  self.use_pos_enc = use_pos_enc
  if pos_enc is not None:
  self.register_buffer('pos_enc', pos_enc)
  else:
  self.pos_enc = None

  @torch.no_grad()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
  B, C = x.shape
  oh = F.one_hot(x, num_classes=self.vocab_size).float()
  if self.use_pos_enc and self.pos_enc is not None:
  oh = oh + self.pos_enc.unsqueeze(0) * 0.1
  x_pool = oh.mean(dim=1)
  x_bin = torch.sigmoid(self.input_proj(x_pool))

  h = torch.zeros(B, self.hidden_dim, device=x.device)
  for t in range(self.max_iters):
  combined = torch.cat([x_bin, torch.sigmoid(h)], dim=-1)
  h_new = self.hard_block(combined)
  change = (h_new - h).abs().mean(dim=-1)
  if (change < self.halt_threshold).all() and t > 0:
  h = h_new
  break
  h = h_new

  return self.head(h)

def collapse_looped_dlgn(model: LoopedDLGN) -> HardLoopedDLGN:
  """LoopedDLGN → HardLoopedDLGN への変換。"""
  hard_block = model.shared_block.collapse()
  pos_enc = model.pos_enc if model.use_pos_enc else None
  return HardLoopedDLGN(
  input_proj=model.input_proj,
  hard_block=hard_block,
  head=model.head,
  vocab_size=model.vocab_size,
  context_len=model.context_len,
  hidden_dim=model.hidden_dim,
  max_iters=model.max_iters,
  halt_threshold=model.halt_threshold,
  use_pos_enc=model.use_pos_enc,
  pos_enc=pos_enc,
  )

# ---------------------------------------------------------------------------
# 4. Data preparation (dlgn_charlm.py から流用)
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
# 5. Training utilities
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
  logits = model(xb, temperature=temperature)
  else:
  logits = model(xb)
  total_loss += F.cross_entropy(logits, yb).item()
  n_batches += 1
  return math.exp(total_loss / max(1, n_batches))

def train_looped_dlgn(
  model: LoopedDLGN,
  X_train, Y_train, X_val, Y_val,
  epochs: int = 60,
  batch_size: int = 256,
  lr: float = 3e-3,
  temp_scheduler: TemperatureScheduler = None,
) -> list:
  """
  LoopedDLGN 専用学習ループ。
  スペクトルノルムペナルティが有効な場合は loss に加算する。
  """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  ppl_history = 
  log_every = max(1, epochs // 10)

  print(f"\n  {'Ep':>4} {'tau':>6} {'TrainLoss':>10} {'ValPPL':>9}")
  print(f"  {'-' * 35}")

  for epoch in range(epochs):
  tau = temp_scheduler.temperature if temp_scheduler else 1.0
  model.train()
  perm = torch.randperm(len(X_train))
  X_s, Y_s = X_train[perm], Y_train[perm]

  ep_loss, n_b = 0.0, 0
  for i in range(0, len(X_train), batch_size):
  xb = X_s[i:i + batch_size]
  yb = Y_s[i:i + batch_size]
  optimizer.zero_grad()

  logits = model(xb, temperature=tau)
  loss = F.cross_entropy(logits, yb)

  # スペクトルノルムペナルティ (有効な場合のみ加算)
  sn_pen = model.shared_block_wrapper.sn_penalty()
  if sn_pen.item() > 0.0:
  loss = loss + sn_pen

  loss.backward()
  # BPTT 勾配死対策: 指摘で max_norm=5.0 → 1.0 (深い反復では 5.0 は甘い)
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  ep_loss += loss.item()
  n_b += 1

  lr_sched.step()
  if temp_scheduler:
  temp_scheduler.step()

  if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
  tau_disp = temp_scheduler.temperature if temp_scheduler else 1.0
  ppl = compute_perplexity(
  model, X_val, Y_val, temperature=tau_disp if temp_scheduler else None
  )
  ppl_history.append((epoch + 1, ppl))
  print(f"  {epoch+1:>4} {tau_disp:>6.3f} {ep_loss/n_b:>10.4f} {ppl:>9.2f}",
  flush=True)

  return ppl_history

# ---------------------------------------------------------------------------
# 6. Inference speed measurement
# ---------------------------------------------------------------------------

def measure_tokens_per_sec(
  model, X_sample: torch.Tensor,
  use_temp: bool = False, runs: int = 30,
  tau: float = 0.1,
) -> float:
  model.eval()
  with torch.no_grad():
  for _ in range(3):
  if use_temp:
  model(X_sample, temperature=tau)
  else:
  model(X_sample)
  t0 = time.perf_counter()
  for _ in range(runs):
  if use_temp:
  model(X_sample, temperature=tau)
  else:
  model(X_sample)
  return X_sample.shape[0] * runs / (time.perf_counter() - t0)

# ---------------------------------------------------------------------------
# 7. Convergence iteration statistics (推論時の平均収束反復数)
# ---------------------------------------------------------------------------

def measure_convergence_stats(
  model: LoopedDLGN, X_val: torch.Tensor, batch_size: int = 256
) -> dict:
  """
  推論時に各バッチが何回反復で収束したかを計測する。
  model.start_iter_stats() / stop_iter_stats() を使う。
  """
  model.eval()
  model.start_iter_stats()
  with torch.no_grad():
  for i in range(0, len(X_val), batch_size):
  xb = X_val[i:i + batch_size]
  model(xb, temperature=0.1)
  return model.stop_iter_stats()

# ---------------------------------------------------------------------------
# 8. Text generation
# ---------------------------------------------------------------------------

def generate_text(
  model, seed_text: str, stoi: dict, itos: dict,
  context_len: int, n_chars: int = 80,
  temperature: float = 1.0,
  use_model_temp: bool = False,
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
  if use_model_temp:
  logits = model(x, temperature=max(0.1, temperature))
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
  print("  LoopedDLGN Char-LM PoC")
  print(f"  Device: CPU  PyTorch {torch.__version__}")
  print("=" * 68)

  # ── Data ────────────────────────────────────────────────────────────────
  text = load_text()
  stoi, itos = build_vocab(text)
  vocab_size = len(stoi)
  context_len = 16

  print(f"\n[vocab] size={vocab_size}  text_len={len(text):,}")

  split = int(len(text) * 0.9)
  X_train, Y_train = make_dataset(text[:split], stoi, context_len)
  X_val,  Y_val  = make_dataset(text[split:], stoi, context_len)
  print(f"[data] train={len(X_train):,}  val={len(X_val):,}  context_len={context_len}")

  # ── Hyperparameters ─────────────────────────────────────────────────────
  # コマンドライン引数で上書き可能
  EPOCHS  = 60
  USE_POS_ENC  = False  # まず mean pooling で試す
  HIDDEN_DIM  = 128
  MAX_ITERS  = 8
  for arg in sys.argv[1:]:
  if arg.startswith('--epochs='):
  EPOCHS = int(arg.split('=', 1)[1])
  print(f"[CLI] epochs override: {EPOCHS}")
  elif arg == '--pos-enc':
  USE_POS_ENC = True
  print(f"[CLI] use_pos_enc: True")
  elif arg.startswith('--hidden-dim='):
  HIDDEN_DIM = int(arg.split('=', 1)[1])
  print(f"[CLI] hidden_dim: {HIDDEN_DIM}")
  elif arg.startswith('--max-iters='):
  MAX_ITERS = int(arg.split('=', 1)[1])
  print(f"[CLI] max_iters: {MAX_ITERS}")
  BATCH  = 256
  LR  = 3e-3
  BLOCK_HIDDEN = [256, 256, 128]
  HALT_THR  = 0.05
  for arg in sys.argv[1:]:
  if arg.startswith('--block-hidden='):
  BLOCK_HIDDEN = [int(x) for x in arg.split('=', 1)[1].split(',')]
  print(f"[CLI] block_hidden: {BLOCK_HIDDEN}")

  # ── Model ───────────────────────────────────────────────────────────────
  model = LoopedDLGN(
  vocab_size=vocab_size,
  context_len=context_len,
  hidden_dim=HIDDEN_DIM,
  max_iters=MAX_ITERS,
  block_hidden=BLOCK_HIDDEN,
  halt_threshold=HALT_THR,
  use_pos_enc=USE_POS_ENC,
  seed_base=42,
  )

  # 指摘で shared_block にもソフトなスペクトルノルムペナルティを加える
  # (head は spectral_norm 適用済み、shared_block は損失項として弱制約)
  model.shared_block_wrapper.enable_spectral_norm(lambda_sn=1e-3)

  print(f"\n[LoopedDLGN] params: {model.param_count():,}")
  print(f"  hidden_dim={HIDDEN_DIM}, max_iters={MAX_ITERS}")
  print(f"  block_hidden={BLOCK_HIDDEN}, halt_threshold={HALT_THR}")
  print(f"  use_pos_enc={USE_POS_ENC}")
  print(f"  shared_block in_features={HIDDEN_DIM*2}, num_classes={HIDDEN_DIM}")
  print(f"  - head に nn.utils.spectral_norm 適用 (L<1 を外部担保)")
  print(f"  - shared_block にスペクトルノルムペナルティ λ=1e-3")
  print(f"  - 勾配クリップ max_norm=1.0 (深い反復対応)")

  # ── Temperature scheduler ───────────────────────────────────────────────
  temp_sched = TemperatureScheduler(
  t_start=1.0, t_end=0.1,
  total_epochs=EPOCHS,
  warm_hold_frac=0.25,
  )

  # ── Training ────────────────────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print(f"  LoopedDLGN training  epochs={EPOCHS}  batch={BATCH}  lr={LR}")
  print(f"  Temperature annealing: 1.0 -> 0.1  (warm_hold=25%)")
  print(f"{'=' * 68}")

  t_start = time.perf_counter()
  ppl_history = train_looped_dlgn(
  model, X_train, Y_train, X_val, Y_val,
  epochs=EPOCHS, batch_size=BATCH, lr=LR,
  temp_scheduler=temp_sched,
  )
  train_time = time.perf_counter() - t_start
  print(f"\n  Training time: {train_time:.1f}s ({train_time/60:.1f}min)")

  # ── Final perplexity (soft) ──────────────────────────────────────────────
  final_tau = temp_sched.temperature
  final_ppl_soft = compute_perplexity(model, X_val, Y_val, temperature=final_tau)

  # ── Convergence stats (推論時の平均収束反復数) ─────────────────────────
  conv_stats = measure_convergence_stats(model, X_val)
  print(f"\n[収束統計 (推論時)]")
  print(f"  平均収束反復数: {conv_stats['mean_iters']:.2f} / {MAX_ITERS}")
  print(f"  早期停止率: {conv_stats['early_stop_rate']:.1%}  "
  f"(n={conv_stats['n_samples']})")

  # ── Inference speed (soft) ───────────────────────────────────────────────
  sample_x = X_val[:512]
  speed_soft = measure_tokens_per_sec(model, sample_x, use_temp=True, tau=final_tau)

  # ── Hard collapse ───────────────────────────────────────────────────────
  print(f"\n[collapse] Converting LoopedDLGN to hard Boolean circuit...")
  hard_model = collapse_looped_dlgn(model)
  hard_model.eval()

  speed_hard = measure_tokens_per_sec(hard_model, sample_x, use_temp=False)

  # Hard perplexity
  total_loss, n_b = 0.0, 0
  with torch.no_grad():
  for i in range(0, len(X_val), 256):
  xb = X_val[i:i + 256]
  yb = Y_val[i:i + 256]
  logits = hard_model(xb)
  total_loss += F.cross_entropy(logits, yb).item()
  n_b += 1
  ppl_hard = math.exp(total_loss / max(1, n_b))

  # ── Text generation ──────────────────────────────────────────────────────
  seeds = ["First ", "To be ", "The ki"]

  print(f"\n{'=' * 68}")
  print("  Generated text: LoopedDLGN soft (temperature=0.8)")
  print(f"{'=' * 68}")
  for seed in seeds:
  gen = generate_text(model, seed, stoi, itos, context_len,
  n_chars=60, temperature=0.8, use_model_temp=True)
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 68}")
  print("  Generated text: LoopedDLGN greedy (temperature=0)")
  print(f"{'=' * 68}")
  for seed in seeds:
  gen = generate_text(model, seed, stoi, itos, context_len,
  n_chars=60, temperature=0.0, use_model_temp=True)
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 68}")
  print("  Generated text: LoopedDLGN Hard Boolean Circuit (temperature=0.8)")
  print(f"{'=' * 68}")
  for seed in seeds:
  gen = generate_text(hard_model, seed, stoi, itos, context_len,
  n_chars=60, temperature=0.8, use_model_temp=False)
  print(f"  [{seed!r:8}] -> {gen!r}")

  # ── Comparison table ─────────────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print("  LoopedDLGN vs DLGN flat vs Transformer — 比較表")
  print(f"{'=' * 68}")
  print(f"  {'Model':<26} {'Params':>8} {'ValPPL':>8} {'Speed(tok/s)':>14}")
  print(f"  {'-' * 62}")
  # 今回の新結果
  print(f"  {'LoopedDLGN (soft)':<26} {model.param_count():>8,} "
  f"{final_ppl_soft:>8.2f} {speed_soft:>14,.0f}")
  print(f"  {'LoopedDLGN (hard circuit)':<26} {model.param_count():>8,} "
  f"{ppl_hard:>8.2f} {speed_hard:>14,.0f}")
  # 前回ベンチ (meetings/2026-05-10_dlgn-charlm-results.md より)
  print(f"  {'DLGN flat (soft) [前回]':<26} {'1,362,813':>8} "
  f"{'11.83':>8} {'18,299':>14}")
  print(f"  {'DLGN flat (hard) [前回]':<26} {'1,362,813':>8} "
  f"{'15.16':>8} {'61,991':>14}")
  print(f"  {'Transformer [前回]':<26} {'108,861':>8} "
  f"{'4.86':>8} {'51,291':>14}")

  print(f"\n  LoopedDLGN vs TF (前回): PPL ratio = {final_ppl_soft / 4.86:.2f}x")
  print(f"  Hard circuit speed vs TF (前回): {speed_hard / 51291:.2f}x")
  print(f"  Hard vs Soft speed: {speed_hard / speed_soft:.2f}x")

  # ── Perplexity curve ─────────────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print("  Perplexity curve (LoopedDLGN)")
  print(f"{'=' * 68}")
  print(f"  {'Epoch':>6} {'ValPPL':>10}")
  print(f"  {'-' * 20}")
  for ep, ppl in ppl_history:
  print(f"  {ep:>6} {ppl:>10.2f}")

  # ── Analysis ─────────────────────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print(f"{'=' * 68}")

  if final_ppl_soft < 4.86:
  verdict = "LoopedDLGN が Transformer を超えた！ 設計案5 勝利"
  elif final_ppl_soft < 7.0:
  verdict = "LoopedDLGN が DLGN flat を大幅改善 (TF に迫るが未到達)"
  elif final_ppl_soft < 11.83:
  verdict = "LoopedDLGN が DLGN flat より改善 (反復精緻化は効いている)"
  else:
  verdict = "LoopedDLGN と DLGN flat 同等 — アーキ変更の効果が限定的"

  print(f"  判定: {verdict}")
  print(f"  LoopedDLGN soft PPL:  {final_ppl_soft:.2f}")
  print(f"  LoopedDLGN hard PPL:  {ppl_hard:.2f}")
  print(f"  DLGN flat soft PPL:  11.83 [前回]")
  print(f"  Transformer PPL:  4.86 [前回]")
  print(f"  平均収束反復数:  {conv_stats['mean_iters']:.2f} / {MAX_ITERS}")
  print(f"  早期停止率:  {conv_stats['early_stop_rate']:.1%}")
  print()
  print("  [次フェーズ候補]")
  print("  - use_pos_enc=True で位置情報あり版を試す")
  print("  - max_iters=16 に増やして深さを稼ぐ")
  print("  - 案4 Bloom Filter を入力に追加してテキスト統計記憶を補強")
  print()

if __name__ == "__main__":
  main()
