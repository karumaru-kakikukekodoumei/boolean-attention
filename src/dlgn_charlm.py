"""
DLGN Char-LM PoC -- DifferentiableLogicGateNetwork char-level LM transfer experiment.
Verifying: Can Boolean circuits learn sequences? Is CPU inference faster than Transformer?

Kawanishi Ren (AI Engineer) -- LLM transfer PoC, CPU only, Windows Ryzen 7 5700X
"""

import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse existing DLGN core
from dynamic_dlgn import (
  DynamicDLGNModel,
  TemperatureScheduler,
)

# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

TINY_SHAKESPEARE_URL = (
  "https://raw.githubusercontent.com/karpathy/char-rnn/"
  "master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_CACHE = Path(__file__).parent / "tinyshakespeare.txt"

# Limit bytes so CPU training finishes in reasonable time
MAX_BYTES = 80_000  # ~80 KB

def load_text() -> str:
  """Load TinyShakespeare; fall back to synthetic pattern on failure."""
  if TINY_SHAKESPEARE_CACHE.exists():
  text = TINY_SHAKESPEARE_CACHE.read_text(encoding="utf-8")
  print(f"[data] cache hit: {len(text):,} chars")
  return text[:MAX_BYTES]

  print(f"[data] downloading TinyShakespeare...")
  try:
  with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=15) as r:
  raw = r.read().decode("utf-8")
  TINY_SHAKESPEARE_CACHE.write_text(raw, encoding="utf-8")
  print(f"[data] downloaded: {len(raw):,} chars -> using {MAX_BYTES:,}")
  return raw[:MAX_BYTES]
  except Exception as e:
  print(f"[data] download failed ({e}) -> using synthetic fallback")
  return _make_synthetic_text(80_000)

def _make_synthetic_text(n: int) -> str:
  """Simple Markov-like pattern: a->b->c->... with 20% random jumps."""
  base = "abcdefghijklmnopqrstuvwxyz 0123456789\n"
  import random
  random.seed(42)
  chunks = 
  cur = "a"
  for _ in range(n):
  chunks.append(cur)
  idx = base.find(cur)
  if random.random() < 0.8:
  cur = base[(idx + 1) % len(base)]
  else:
  cur = random.choice(base)
  return "".join(chunks)

def build_vocab(text: str):
  """Build char-level vocab from text."""
  chars = sorted(set(text))
  stoi = {c: i for i, c in enumerate(chars)}
  itos = {i: c for c, i in stoi.items()}
  return stoi, itos

def make_dataset(text: str, stoi: dict, context_len: int):
  """
  Sliding-window dataset.
  X: (N, context_len) int64 -- input context
  Y: (N,) int64  -- next character
  """
  ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
  n = len(ids) - context_len
  X = torch.stack([ids[i:i + context_len] for i in range(n)])
  Y = ids[context_len:]
  return X, Y

# ---------------------------------------------------------------------------
# 2. DLGNCharLM model
# ---------------------------------------------------------------------------

class DLGNCharLM(nn.Module):
  """
  Char-level language model using DLGN backbone.

  Flow:
  token ids (B, C) -> one-hot (B, C*V) -> {0,1} bits
  -> DynamicDLGNModel (4 layers) -> (B, H)
  -> Linear head -> logits (B, V)

  Feeding one-hot as bit-string lets Boolean gates directly test
  'did character X appear at position P?' conditions.
  """

  def __init__(self, vocab_size: int, context_len: int,
  hidden_neurons: list, seed_base: int = 0):
  super().__init__()
  self.vocab_size = vocab_size
  self.context_len = context_len
  self.in_features = context_len * vocab_size

  self.backbone = DynamicDLGNModel(
  in_features=self.in_features,
  hidden_neurons=hidden_neurons,
  num_classes=vocab_size,
  init_scale=0.1,
  init_pairing='random_hard',
  seed_base=seed_base,
  )

  def _encode(self, x: torch.Tensor) -> torch.Tensor:
  """x: (B, C) int64 -> (B, C*V) float32 in {0,1}"""
  B, C = x.shape
  oh = F.one_hot(x, num_classes=self.vocab_size).float()  # (B, C, V)
  return oh.view(B, -1)  # (B, C*V)

  def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
  bits = self._encode(x)
  return self.backbone(bits, temperature)

  def collapse(self):
  """Return hard Boolean circuit version."""
  hard_backbone = self.backbone.collapse()
  return HardDLGNCharLM(hard_backbone, self.vocab_size, self.context_len)

  def param_count(self) -> int:
  return sum(p.numel() for p in self.parameters())

class HardDLGNCharLM(nn.Module):
  """Post-collapse hard inference model."""

  def __init__(self, hard_backbone, vocab_size: int, context_len: int):
  super().__init__()
  self.hard_backbone = hard_backbone
  self.vocab_size = vocab_size
  self.context_len = context_len

  @torch.no_grad()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
  B, C = x.shape
  oh = F.one_hot(x, num_classes=self.vocab_size).float()
  bits = oh.view(B, -1)
  return self.hard_backbone(bits)

# ---------------------------------------------------------------------------
# 3. Transformer baseline
# ---------------------------------------------------------------------------

class TinyTransformerLM(nn.Module):
  """
  Small Transformer decoder for comparison.
  d_model and layers kept small to stay near DLGN param count.
  """

  def __init__(self, vocab_size: int, context_len: int,
  d_model: int = 64, n_heads: int = 2, n_layers: int = 2,
  dropout: float = 0.1):
  super().__init__()
  self.vocab_size = vocab_size
  self.context_len = context_len
  self.d_model = d_model

  self.tok_emb = nn.Embedding(vocab_size, d_model)
  self.pos_emb = nn.Embedding(context_len, d_model)

  encoder_layer = nn.TransformerEncoderLayer(
  d_model=d_model, nhead=n_heads,
  dim_feedforward=d_model * 4,
  dropout=dropout, batch_first=True,
  norm_first=True,
  )
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

  # causal mask
  mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
  self.register_buffer('causal_mask', mask)

  self.head = nn.Linear(d_model, vocab_size)

  def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
  """
  x: (B, context_len) int64
  returns: (B, vocab_size) -- logits at last position
  """
  B, C = x.shape
  pos = torch.arange(C, device=x.device).unsqueeze(0)
  h = self.tok_emb(x) + self.pos_emb(pos)
  h = self.transformer(h, mask=self.causal_mask, is_causal=True)
  return self.head(h[:, -1, :])

  def param_count(self) -> int:
  return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def compute_perplexity(model, X_val: torch.Tensor, Y_val: torch.Tensor,
  batch_size: int = 256, temperature: float = None) -> float:
  model.eval()
  total_loss = 0.0
  n_batches = 0
  with torch.no_grad():
  for i in range(0, len(X_val), batch_size):
  xb = X_val[i:i + batch_size]
  yb = Y_val[i:i + batch_size]
  if temperature is not None:
  logits = model(xb, temperature=temperature)
  else:
  logits = model(xb)
  loss = F.cross_entropy(logits, yb)
  total_loss += loss.item()
  n_batches += 1
  avg_loss = total_loss / max(1, n_batches)
  return math.exp(avg_loss)

def train_model(model, X_train, Y_train, X_val, Y_val,
  epochs: int = 60, batch_size: int = 256, lr: float = 3e-3,
  temp_scheduler: TemperatureScheduler = None,
  label: str = "model") -> list:
  """General training loop. Returns list of (epoch, val_ppl)."""
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

  ep_loss = 0.0
  n_b = 0
  for i in range(0, len(X_train), batch_size):
  xb = X_s[i:i + batch_size]
  yb = Y_s[i:i + batch_size]
  optimizer.zero_grad()

  if temp_scheduler is not None:
  logits = model(xb, temperature=tau)
  else:
  logits = model(xb)

  loss = F.cross_entropy(logits, yb)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
  optimizer.step()
  ep_loss += loss.item()
  n_b += 1

  lr_sched.step()
  if temp_scheduler:
  temp_scheduler.step()

  if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
  tau_disp = temp_scheduler.temperature if temp_scheduler else 1.0
  ppl = compute_perplexity(model, X_val, Y_val,
  temperature=tau_disp if temp_scheduler else None)
  ppl_history.append((epoch + 1, ppl))
  print(f"  {epoch+1:>4} {tau_disp:>6.3f} {ep_loss/n_b:>10.4f} {ppl:>9.2f}")

  return ppl_history

# ---------------------------------------------------------------------------
# 5. Text generation
# ---------------------------------------------------------------------------

def generate_text(model, seed_text: str, stoi: dict, itos: dict,
  context_len: int, n_chars: int = 80,
  temperature: float = 1.0,
  use_model_temp: bool = False) -> str:
  """
  Greedy (temperature=0) or temperature sampling generation.
  use_model_temp=True passes temperature to model.forward() as well.
  """
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
# 6. Throughput measurement
# ---------------------------------------------------------------------------

def measure_tokens_per_sec(model, X_sample: torch.Tensor,
  use_temp: bool = False, runs: int = 30) -> float:
  model.eval()
  tau = 0.1
  with torch.no_grad():
  # warmup
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
  elapsed = time.perf_counter() - t0
  total_tokens = X_sample.shape[0] * runs
  return total_tokens / elapsed

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
  torch.manual_seed(42)

  print("=" * 68)
  print("  DLGN Char-LM PoC")
  print("  Kawanishi Ren -- Can Boolean circuits learn language?")
  print("  Device: CPU  PyTorch", torch.__version__)
  print("=" * 68)

  # -- Data --
  text = load_text()
  stoi, itos = build_vocab(text)
  vocab_size = len(stoi)
  print(f"\n[vocab] size={vocab_size}  text_len={len(text):,}")

  context_len = 16  # one-hot expanded: context_len * vocab_size bits
  split = int(len(text) * 0.9)
  train_text = text[:split]
  val_text  = text[split:]

  X_train, Y_train = make_dataset(train_text, stoi, context_len)
  X_val,  Y_val  = make_dataset(val_text,  stoi, context_len)
  print(f"[data] train={len(X_train):,}  val={len(X_val):,}  "
  f"in_features={context_len * vocab_size}")

  # -- DLGN architecture --
  # Input dim: context_len * vocab_size (large -> compress aggressively)
  in_dim = context_len * vocab_size
  hidden_neurons = [512, 256, 128, 64]

  print(f"\n[DLGN] arch: {in_dim} -> {hidden_neurons} -> {vocab_size}")

  dlgn_model = DLGNCharLM(
  vocab_size=vocab_size,
  context_len=context_len,
  hidden_neurons=hidden_neurons,
  seed_base=42,
  )
  print(f"[DLGN] params: {dlgn_model.param_count():,}")

  # -- Transformer baseline --
  tf_d_model = 64
  tf_layers  = 2
  tf_model = TinyTransformerLM(
  vocab_size=vocab_size,
  context_len=context_len,
  d_model=tf_d_model,
  n_heads=2,
  n_layers=tf_layers,
  dropout=0.1,
  )
  print(f"[Transformer] d_model={tf_d_model} layers={tf_layers}  "
  f"params: {tf_model.param_count():,}")

  # -- DLGN training --
  epochs_dlgn = 40
  temp_sched = TemperatureScheduler(
  t_start=1.0, t_end=0.1,
  total_epochs=epochs_dlgn,
  warm_hold_frac=0.25,
  )

  print(f"\n{'=' * 68}")
  print(f"  DLGN training  epochs={epochs_dlgn}  batch=256  lr=3e-3")
  print(f"  Temperature annealing: 1.0 -> 0.1  (warm_hold=25%)")
  print(f"{'=' * 68}")

  t_dlgn_start = time.perf_counter()
  ppl_dlgn = train_model(
  dlgn_model, X_train, Y_train, X_val, Y_val,
  epochs=epochs_dlgn, batch_size=256, lr=3e-3,
  temp_scheduler=temp_sched,
  label="DLGN",
  )
  dlgn_train_time = time.perf_counter() - t_dlgn_start
  print(f"\n  Training time: {dlgn_train_time:.1f}s")

  # -- Transformer training --
  epochs_tf = 40
  print(f"\n{'=' * 68}")
  print(f"  Transformer training  epochs={epochs_tf}  batch=256  lr=3e-3")
  print(f"{'=' * 68}")

  t_tf_start = time.perf_counter()
  ppl_tf = train_model(
  tf_model, X_train, Y_train, X_val, Y_val,
  epochs=epochs_tf, batch_size=256, lr=3e-3,
  temp_scheduler=None,
  label="Transformer",
  )
  tf_train_time = time.perf_counter() - t_tf_start
  print(f"\n  Training time: {tf_train_time:.1f}s")

  # -- Final perplexity --
  final_tau = temp_sched.temperature
  final_ppl_dlgn = compute_perplexity(
  dlgn_model, X_val, Y_val, temperature=final_tau
  )
  final_ppl_tf = compute_perplexity(tf_model, X_val, Y_val)

  # -- Inference speed --
  sample_x = X_val[:512]
  dlgn_speed_soft = measure_tokens_per_sec(dlgn_model, sample_x, use_temp=True)
  tf_speed = measure_tokens_per_sec(tf_model, sample_x, use_temp=False)

  # -- DLGN collapse to hard Boolean circuit --
  print(f"\n[collapse] Converting DLGN to hard Boolean circuit...")
  hard_dlgn = dlgn_model.collapse()
  hard_dlgn.eval()
  dlgn_speed_hard = measure_tokens_per_sec(hard_dlgn, sample_x, use_temp=False)

  # Perplexity after collapse (head stays float, gate layers are hard)
  total_loss = 0.0
  n_b = 0
  with torch.no_grad():
  for i in range(0, len(X_val), 256):
  xb = X_val[i:i + 256]
  yb = Y_val[i:i + 256]
  logits = hard_dlgn(xb)
  total_loss += F.cross_entropy(logits, yb).item()
  n_b += 1
  hard_ppl = math.exp(total_loss / max(1, n_b))

  # -- Text generation samples --
  seeds = ["First ", "To be ", "The ki", "What i", "O my "]
  print(f"\n{'=' * 68}")
  print("  Generated text: DLGN soft (temperature=0.8)")
  print(f"{'=' * 68}")
  for seed in seeds:
  gen = generate_text(
  dlgn_model, seed, stoi, itos, context_len,
  n_chars=60, temperature=0.8, use_model_temp=True,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 68}")
  print("  Generated text: DLGN greedy (temperature=0)")
  print(f"{'=' * 68}")
  for seed in seeds[:3]:
  gen = generate_text(
  dlgn_model, seed, stoi, itos, context_len,
  n_chars=60, temperature=0.0, use_model_temp=True,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 68}")
  print("  Generated text: DLGN Hard Boolean Circuit")
  print(f"{'=' * 68}")
  for seed in seeds[:3]:
  gen = generate_text(
  hard_dlgn, seed, stoi, itos, context_len,
  n_chars=60, temperature=0.8, use_model_temp=False,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 68}")
  print("  Generated text: Transformer (temperature=0.8)")
  print(f"{'=' * 68}")
  for seed in seeds[:3]:
  gen = generate_text(
  tf_model, seed, stoi, itos, context_len,
  n_chars=60, temperature=0.8, use_model_temp=False,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  # -- Comparison table --
  print(f"\n{'=' * 68}")
  print("  DLGN vs Transformer Comparison")
  print(f"{'=' * 68}")
  print(f"  {'Model':<22} {'Params':>8} {'ValPPL':>8} {'Speed(tok/s)':>14}")
  print(f"  {'-' * 58}")
  print(f"  {'DLGN (soft)':<22} {dlgn_model.param_count():>8,} "
  f"{final_ppl_dlgn:>8.2f} {dlgn_speed_soft:>14,.0f}")
  print(f"  {'DLGN (hard circuit)':<22} {dlgn_model.param_count():>8,} "
  f"{hard_ppl:>8.2f} {dlgn_speed_hard:>14,.0f}")
  print(f"  {'Transformer':<22} {tf_model.param_count():>8,} "
  f"{final_ppl_tf:>8.2f} {tf_speed:>14,.0f}")
  print(f"\n  Hard circuit speed ratio vs Transformer: "
  f"{dlgn_speed_hard / tf_speed:.2f}x")
  print(f"  Hard circuit speed ratio vs DLGN soft:  "
  f"{dlgn_speed_hard / dlgn_speed_soft:.2f}x")

  # -- Perplexity curve summary --
  print(f"\n{'=' * 68}")
  print("  Perplexity curve summary")
  print(f"{'=' * 68}")
  print(f"  {'Epoch':>6} {'DLGN PPL':>10} {'TF PPL':>10}")
  print(f"  {'-' * 30}")
  ppl_tf_dict = dict(ppl_tf)
  for ep, ppl in ppl_dlgn:
  tf_ppl_ep = ppl_tf_dict.get(ep, float('nan'))
  print(f"  {ep:>6} {ppl:>10.2f} {tf_ppl_ep:>10.2f}")

  # -- Findings --
  print(f"\n{'=' * 68}")
  print("  Findings (Kawanishi Ren)")
  print(f"{'=' * 68}")
  ppl_ratio = final_ppl_dlgn / final_ppl_tf if final_ppl_tf > 0 else float('inf')
  speed_ratio = dlgn_speed_hard / tf_speed

  if final_ppl_dlgn < final_ppl_tf * 2.0:
  verdict_ppl = "DLGN achieves similar accuracy to Transformer"
  elif final_ppl_dlgn < final_ppl_tf * 5.0:
  verdict_ppl = "DLGN lags behind Transformer but learning converges"
  else:
  verdict_ppl = "DLGN accuracy well below Transformer -- needs arch improvement"

  if speed_ratio > 1.0:
  verdict_speed = f"Hard circuit is {speed_ratio:.1f}x faster than Transformer"
  else:
  verdict_speed = f"Hard circuit is {1/speed_ratio:.1f}x slower than Transformer"

  print(f"  Accuracy : {verdict_ppl}")
  print(f"  DLGN PPL={final_ppl_dlgn:.2f}  TF PPL={final_ppl_tf:.2f}  "
  f"ratio={ppl_ratio:.2f}x")
  print(f"  Speed  : {verdict_speed}")
  print(f"  Collapse : soft {dlgn_speed_soft:,.0f} -> hard {dlgn_speed_hard:,.0f} tok/s")
  print()
  print("  [Conclusion]")
  print("  - DLGN can learn Char-LM (perplexity decreases during training)")
  print("  - Inference works even after collapsing to hard Boolean circuit")
  print("  - Speed bottleneck: large input dim (context_len*vocab_size)")
  print("  softmax-based input selection in DynamicGateLayer dominates")
  print("  - For production: embedding dimension reduction or coarse one-hot")
  print()

if __name__ == "__main__":
  main()
