"""
ChatHBA — HBA Causal Chat LM
==============================

概要:
  - Q&A コーパス (JSONL) を char 列に変換
  - ChatHBA (causal LM) で全位置 logits を出力
  - autoregressive 生成でチャット推論

フォーマット:
  Q: <question>\nA: <answer>\n<END>

小型版 (A案) デフォルト:
  context_len=64, hidden_dim=48, n_layers=2, n_heads=4

設計継承:
  - BooleanAttentionLayer (bilinear router + float value aggregation)
  - Entropy 正則化
  - spectral_norm head
  - gradient clip max_norm=1.0
  - Best checkpoint + Early stopping
  - Gumbel τ scheduler (warm_hold)
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# dynamic_dlgn は親ディレクトリにある
sys.path.insert(0, str(Path(__file__).parent.parent))
from dynamic_dlgn import TemperatureScheduler  # noqa: E402

# ---------------------------------------------------------------------------
# 0. 特殊トークン / vocab
# ---------------------------------------------------------------------------

# 区切り文字として使う制御文字
_END_TOKEN = "\x03"  # <END> 終端
_PAD_TOKEN = "\x00"  # padding

SPECIAL_CHARS = [_PAD_TOKEN, _END_TOKEN]

CORPUS_PATH = Path(__file__).parent / "qa_corpus.jsonl"
CHECKPOINT_PATH = Path(__file__).parent / "chat_hba_best.pt"

# ---------------------------------------------------------------------------
# 1. データ前処理
# ---------------------------------------------------------------------------

def format_qa(q: str, a: str) -> str:
  """1 件の Q&A を学習用 char 列に変換。"""
  return f"Q: {q}\nA: {a}\n{_END_TOKEN}"

def load_corpus(path: Path, subset: int = 0) -> list[str]:
  """JSONL を読み込み、フォーマット済み文字列リストを返す。"""
  records = 
  with open(path, encoding="utf-8") as f:
  for line in f:
  line = line.strip()
  if not line:
  continue
  d = json.loads(line)
  records.append(format_qa(d["q"], d["a"]))
  if subset > 0:
  records = records[:subset]
  return records

def build_vocab(texts: list[str]) -> tuple[dict, dict]:
  """全テキストから文字 vocab を構築。特殊文字を先頭に追加。"""
  chars = set()
  for t in texts:
  chars.update(t)
  # 特殊文字を確実に含める
  chars.update(SPECIAL_CHARS)
  sorted_chars = SPECIAL_CHARS + sorted(chars - set(SPECIAL_CHARS))
  stoi = {c: i for i, c in enumerate(sorted_chars)}
  itos = {i: c for c, i in stoi.items()}
  return stoi, itos

def make_causal_dataset(
  texts: list[str],
  stoi: dict,
  context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """
  全テキストを連結 -> sliding window で (X, Y) 作成。
  X[i] = text[i:i+context_len]
  Y[i] = text[i+1:i+context_len+1]  (各位置の次 char)
  """
  pad_id = stoi[_PAD_TOKEN]

  # 全テキスト連結 (ペア間は自然につながる: <END> が区切り)
  full_text = "".join(texts)
  ids = torch.tensor(
  [stoi.get(c, pad_id) for c in full_text],
  dtype=torch.long,
  )

  n = len(ids) - context_len
  if n <= 0:
  raise ValueError(f"テキスト長 {len(ids)} < context_len {context_len}")

  # sliding window
  X = torch.stack([ids[i : i + context_len] for i in range(n)])
  Y = torch.stack([ids[i + 1 : i + context_len + 1] for i in range(n)])
  return X, Y

# ---------------------------------------------------------------------------
# 2. BooleanAttentionLayer (hba_charlm.py から継承、causal 対応済み)
# ---------------------------------------------------------------------------

class BooleanAttentionLayer(nn.Module):
  """
  DLGN bilinear router + float value aggregation。
  causal mask 適用済み (j > i は強制 0)。
  entropy 正則化: get_mask_entropy() で取得。
  """

  def __init__(self, hidden_dim: int, n_heads: int = 4):
  super().__init__()
  self.hidden_dim = hidden_dim
  self.n_heads = n_heads
  self.head_dim = hidden_dim // n_heads

  self.q_proj = nn.Linear(hidden_dim, n_heads)
  self.k_proj = nn.Linear(hidden_dim, n_heads)
  self.value_proj = nn.Linear(hidden_dim, hidden_dim)
  self.out_proj = nn.Linear(hidden_dim, hidden_dim)

  self._last_mask_entropy: torch.Tensor = torch.tensor(0.0)

  def forward(self, x: torch.Tensor, tau: float) -> torch.Tensor:
  """
  x  : (B, C, D)
  tau : Gumbel temperature
  returns: (B, C, D)
  """
  B, C, D = x.shape

  Q = self.q_proj(x)  # (B, C, H)
  K = self.k_proj(x)  # (B, C, H)
  mask_logit = Q.unsqueeze(2) * K.unsqueeze(1)  # (B, C_i, C_j, H)

  mask_soft = torch.sigmoid(mask_logit / max(tau, 1e-3))  # (B, C, C, H)

  # causal mask
  causal = torch.tril(torch.ones(C, C, device=x.device))
  mask_soft = mask_soft * causal.unsqueeze(0).unsqueeze(-1)

  # entropy 正則化用キャッシュ
  eps = 1e-6
  p = mask_soft
  h = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
  causal_count = causal.sum().item() * self.n_heads
  self._last_mask_entropy = h.sum() / max(1.0, causal_count)

  # float value aggregation
  V = self.value_proj(x)  # (B, C, D)
  V_heads = V.reshape(B, C, self.n_heads, self.head_dim)

  m = mask_soft.permute(0, 3, 1, 2)  # (B, H, C_i, C_j)
  V_h = V_heads.permute(0, 2, 1, 3)  # (B, H, C_j, head_dim)

  attended_heads = torch.matmul(m, V_h)  # (B, H, C_i, head_dim)
  mask_sum = m.sum(dim=-1, keepdim=True)
  attended_heads = attended_heads / (mask_sum + eps)

  attended = attended_heads.permute(0, 2, 1, 3).reshape(B, C, D)
  return self.out_proj(attended)

  def get_mask_entropy(self) -> torch.Tensor:
  return self._last_mask_entropy

# ---------------------------------------------------------------------------
# 3. ChatHBA — Causal LM (全位置 logits)
# ---------------------------------------------------------------------------

class ChatHBA(nn.Module):
  """
  Causal LM 版 HBA。各位置から次 char の logits を出す。

  hba_charlm.py との差異:
  - forward が (B, C, V) logits を返す (last position だけでなく全位置)
  - head に spectral_norm 適用
  """

  def __init__(
  self,
  vocab_size: int,
  context_len: int = 64,
  hidden_dim: int = 48,
  n_layers: int = 2,
  n_heads: int = 4,
  ):
  super().__init__()
  self.vocab_size = vocab_size
  self.context_len = context_len
  self.hidden_dim = hidden_dim
  self.n_layers = n_layers
  self.n_heads = n_heads

  # Token embedding + sinusoidal PE
  self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
  pe = self._make_sinusoidal_pe(context_len, hidden_dim)
  self.register_buffer("pos_enc", pe)

  # BooleanAttentionLayer x N
  self.attn_layers = nn.ModuleList([
  BooleanAttentionLayer(hidden_dim, n_heads)
  for _ in range(n_layers)
  ])

  # FFN x N (軽量 2-layer)
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
  self.final_norm = nn.LayerNorm(hidden_dim)

  # Head: 全位置 -> vocab logits (spectral_norm 適用: 指摘継承)
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

  def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
  """
  x  : (B, C) int64
  returns: (B, C, V) logits at each position
  """
  B, C = x.shape
  h = self.tok_emb(x) + self.pos_enc[:C].unsqueeze(0)  # (B, C, D)

  for i in range(self.n_layers):
  h = h + self.attn_layers[i](self.attn_norms[i](h), tau)
  h = h + self.ffn_layers[i](self.ffn_norms[i](h))

  h = self.final_norm(h)  # (B, C, D)
  return self.head(h)  # (B, C, V)

  def get_mask_entropies(self) -> list:
  return [layer.get_mask_entropy() for layer in self.attn_layers]

  def param_count(self) -> int:
  return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# 4. 評価ユーティリティ
# ---------------------------------------------------------------------------

def compute_ppl(
  model: ChatHBA,
  X: torch.Tensor,
  Y: torch.Tensor,
  batch_size: int = 256,
  tau: float = 1.0,
) -> float:
  """
  全位置 cross-entropy の平均から PPL を計算。
  X: (N, C), Y: (N, C)  — Y は各位置の次 char
  """
  model.eval()
  total_loss, n_tok = 0.0, 0
  with torch.no_grad():
  for i in range(0, len(X), batch_size):
  xb = X[i : i + batch_size]  # (B, C)
  yb = Y[i : i + batch_size]  # (B, C)
  logits = model(xb, tau=tau)  # (B, C, V)
  B, C, V = logits.shape
  # (B*C, V) vs (B*C,)
  loss = F.cross_entropy(logits.reshape(B * C, V), yb.reshape(B * C))
  total_loss += loss.item() * B * C
  n_tok += B * C
  return math.exp(total_loss / max(1, n_tok))

# ---------------------------------------------------------------------------
# 5. 学習ループ
# ---------------------------------------------------------------------------

def train_chat_hba(
  model: ChatHBA,
  X_train: torch.Tensor,
  Y_train: torch.Tensor,
  X_val: torch.Tensor,
  Y_val: torch.Tensor,
  epochs: int = 30,
  batch_size: int = 128,
  lr: float = 3e-3,
  lambda_ent: float = 0.01,
  temp_scheduler: TemperatureScheduler = None,
  patience: int = 4,
  log_every: int = 2,
  checkpoint_path: Path = CHECKPOINT_PATH,
) -> tuple:
  """
  ChatHBA 専用学習ループ。

  loss = CE(全位置) - lambda_ent * mean_router_entropy

  Returns:
  ppl_history, best_ppl, best_epoch
  """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  ppl_history = 
  best_ppl = float("inf")
  best_state = None
  best_epoch = -1
  no_improve_count = 0

  print(
  f"\n  {'Ep':>4} {'tau':>6} {'CE':>8} {'EntReg':>9} "
  f"{'TotLoss':>9} {'ValPPL':>9} {'MaskH':>8} {'Best?':>6}"
  )
  print(f"  {'-' * 65}")

  for epoch in range(epochs):
  tau = temp_scheduler.temperature if temp_scheduler else 1.0
  model.train()
  perm = torch.randperm(len(X_train))
  X_s, Y_s = X_train[perm], Y_train[perm]

  ep_ce, ep_ent_reg, ep_total, n_b = 0.0, 0.0, 0.0, 0

  for i in range(0, len(X_train), batch_size):
  xb = X_s[i : i + batch_size]
  yb = Y_s[i : i + batch_size]
  optimizer.zero_grad()

  logits = model(xb, tau=tau)  # (B, C, V)
  B, C, V = logits.shape
  ce = F.cross_entropy(logits.reshape(B * C, V), yb.reshape(B * C))

  # entropy 正則化
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
  ppl = compute_ppl(model, X_val, Y_val, tau=tau_disp)
  ppl_history.append((epoch + 1, ppl))

  # mask entropy 計測
  with torch.no_grad():
  _ = model(X_val[: min(256, len(X_val))], tau=tau_disp)
  avg_mask_h = torch.stack(model.get_mask_entropies()).mean().item()

  is_best = ppl < best_ppl
  if is_best:
  best_ppl = ppl
  best_epoch = epoch + 1
  best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
  no_improve_count = 0
  # ディスクにも保存
  torch.save(best_state, checkpoint_path)
  else:
  no_improve_count += 1

  best_mark = " *" if is_best else ""
  print(
  f"  {epoch+1:>4} {tau_disp:>6.3f} "
  f"{ep_ce/n_b:>8.4f} {ep_ent_reg/n_b:>9.4f} "
  f"{ep_total/n_b:>9.4f} {ppl:>9.2f} {avg_mask_h:>8.5f}{best_mark}",
  flush=True,
  )

  # Early stopping
  if patience > 0 and no_improve_count >= patience:
  print(
  f"\n  [early stop] no improvement for {patience} log-epochs "
  f"(best={best_ppl:.4f} @ ep{best_epoch}). Stopping."
  )
  break

  # Best checkpoint restore
  if best_state is not None:
  model.load_state_dict(best_state)
  print(f"\n  [best] restored from epoch {best_epoch}  PPL={best_ppl:.4f}")
  else:
  print("\n  [best] no checkpoint saved (unexpected)")

  return ppl_history, best_ppl, best_epoch

# ---------------------------------------------------------------------------
# 6. Autoregressive 生成
# ---------------------------------------------------------------------------

def generate_answer(
  model: ChatHBA,
  question: str,
  stoi: dict,
  itos: dict,
  context_len: int,
  max_new_chars: int = 80,
  temperature: float = 0.8,
  tau: float = 0.1,
) -> str:
  """
  Q を seed として A を autoregressive 生成。
  <END> トークンが出たら停止。

  seed: "Q: {question}\nA: "
  """
  model.eval()
  pad_id = stoi.get(_PAD_TOKEN, 0)
  end_id = stoi.get(_END_TOKEN, -1)

  seed = f"Q: {question}\nA: "
  seed_ids = [stoi.get(c, pad_id) for c in seed]

  # context_len に収まるよう先頭 pad または truncate
  if len(seed_ids) < context_len:
  ctx = [pad_id] * (context_len - len(seed_ids)) + seed_ids
  else:
  ctx = seed_ids[-context_len:]

  generated = 
  with torch.no_grad():
  for _ in range(max_new_chars):
  x = torch.tensor([ctx], dtype=torch.long)  # (1, C)
  logits = model(x, tau=tau)  # (1, C, V)
  next_logits = logits[0, -1, :]  # (V,)

  if temperature <= 0.0:
  next_id = next_logits.argmax().item()
  else:
  probs = F.softmax(next_logits / temperature, dim=-1)
  next_id = torch.multinomial(probs, 1).item()

  if next_id == end_id:
  break

  generated.append(itos.get(next_id, "?"))
  ctx = ctx[1:] + [next_id]

  return "".join(generated).strip()

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
  torch.manual_seed(42)

  # ── CLI 引数 ──────────────────────────────────────────────────────────
  EPOCHS  = 30
  CONTEXT_LEN = 64
  HIDDEN_DIM  = 48
  N_LAYERS  = 2
  N_HEADS  = 4
  PATIENCE  = 4
  BATCH  = 128
  LR  = 3e-3
  LAMBDA_ENT  = 0.01
  WARM_HOLD  = 0.2
  LOG_EVERY  = 2
  SUBSET  = 0  # 0 = 全件

  for arg in sys.argv[1:]:
  if  arg.startswith("--epochs="):  EPOCHS  = int(arg.split("=",1)[1])
  elif arg.startswith("--context-len="):  CONTEXT_LEN = int(arg.split("=",1)[1])
  elif arg.startswith("--hidden-dim="):  HIDDEN_DIM  = int(arg.split("=",1)[1])
  elif arg.startswith("--n-layers="):  N_LAYERS  = int(arg.split("=",1)[1])
  elif arg.startswith("--n-heads="):  N_HEADS  = int(arg.split("=",1)[1])
  elif arg.startswith("--patience="):  PATIENCE  = int(arg.split("=",1)[1])
  elif arg.startswith("--subset="):  SUBSET  = int(arg.split("=",1)[1])
  elif arg.startswith("--batch="):  BATCH  = int(arg.split("=",1)[1])
  elif arg.startswith("--lr="):  LR  = float(arg.split("=",1)[1])
  elif arg.startswith("--lambda-ent="):  LAMBDA_ENT  = float(arg.split("=",1)[1])
  elif arg.startswith("--warm-hold="):  WARM_HOLD  = float(arg.split("=",1)[1])

  print("=" * 68)
  print("  ChatHBA - HBA Causal Chat LM (Plan-A: small)")
  print("  Design: Chiba Tetsuya / Impl: Kawanishi Ren (AI Eng)")
  print(f"  PyTorch {torch.__version__}  device=CPU")
  print("=" * 68)
  for name, val in [
  ("epochs", EPOCHS), ("context_len", CONTEXT_LEN),
  ("hidden_dim", HIDDEN_DIM), ("n_layers", N_LAYERS),
  ("n_heads", N_HEADS), ("patience", PATIENCE),
  ("batch", BATCH), ("lr", LR), ("lambda_ent", LAMBDA_ENT),
  ("warm_hold", WARM_HOLD), ("subset", SUBSET or "all"),
  ]:
  print(f"  {name}: {val}")
  print("=" * 68)

  # ── Data ─────────────────────────────────────────────────────────────
  print("\n[data] loading corpus...")
  records = load_corpus(CORPUS_PATH, subset=SUBSET)
  print(f"[data] {len(records)} Q&A pairs loaded")

  stoi, itos = build_vocab(records)
  vocab_size = len(stoi)
  print(f"[vocab] size={vocab_size}  (special: PAD={repr(_PAD_TOKEN)}, END={repr(_END_TOKEN)})")

  # 90/10 split
  split_idx = int(len(records) * 0.9)
  train_texts = records[:split_idx]
  val_texts  = records[split_idx:]

  print(f"[data] building dataset (context_len={CONTEXT_LEN})...")
  t0 = time.perf_counter()
  X_train, Y_train = make_causal_dataset(train_texts, stoi, CONTEXT_LEN)
  X_val,  Y_val  = make_causal_dataset(val_texts,  stoi, CONTEXT_LEN)
  print(f"[data] train={len(X_train):,}  val={len(X_val):,}  "
  f"({time.perf_counter()-t0:.1f}s)")

  # ── Model ─────────────────────────────────────────────────────────────
  model = ChatHBA(
  vocab_size=vocab_size,
  context_len=CONTEXT_LEN,
  hidden_dim=HIDDEN_DIM,
  n_layers=N_LAYERS,
  n_heads=N_HEADS,
  )
  print(f"\n[model] ChatHBA params: {model.param_count():,}")
  print(f"  hidden_dim={HIDDEN_DIM}  n_layers={N_LAYERS}  n_heads={N_HEADS}")
  print(f"  head: spectral_norm (Nada's note: inherited)")
  print(f"  entropy reg: lambda_ent={LAMBDA_ENT} (Chiba: never remove)")
  print(f"  grad_clip: max_norm=1.0 (Nada's note: inherited)")

  # ── Temperature scheduler ─────────────────────────────────────────────
  temp_sched = TemperatureScheduler(
  t_start=1.0,
  t_end=0.1,
  total_epochs=EPOCHS,
  warm_hold_frac=WARM_HOLD,
  )

  # ── Training ──────────────────────────────────────────────────────────
  print(f"\n{'=' * 68}")
  print(f"  ChatHBA training  epochs={EPOCHS}  batch={BATCH}  lr={LR}")
  print(f"  Temperature: 1.0 -> 0.1  (warm_hold={WARM_HOLD*100:.0f}%)")
  print(f"  Early stopping: patience={PATIENCE} log-epochs")
  print(f"{'=' * 68}")

  t0 = time.perf_counter()
  ppl_history, best_ppl, best_epoch = train_chat_hba(
  model, X_train, Y_train, X_val, Y_val,
  epochs=EPOCHS,
  batch_size=BATCH,
  lr=LR,
  lambda_ent=LAMBDA_ENT,
  temp_scheduler=temp_sched,
  patience=PATIENCE,
  log_every=LOG_EVERY,
  checkpoint_path=CHECKPOINT_PATH,
  )
  train_time = time.perf_counter() - t0

  print(f"\n  Training time: {train_time:.1f}s ({train_time/60:.1f}min)")
  print(f"  Best epoch: {best_epoch}  Best val PPL: {best_ppl:.4f}")
  print(f"  Checkpoint saved: {CHECKPOINT_PATH}")

  # ── Final evaluation ──────────────────────────────────────────────────
  final_tau = temp_sched.temperature
  final_ppl = compute_ppl(model, X_val, Y_val, tau=final_tau)

  with torch.no_grad():
  _ = model(X_val[: min(256, len(X_val))], tau=final_tau)
  avg_entropy = torch.stack(model.get_mask_entropies()).mean().item()

  # ── Generation examples ───────────────────────────────────────────────
  test_questions = [
  "Who painted the Mona Lisa?",
  "What is the capital of France?",
  "Who wrote Romeo and Juliet?",
  ]

  print(f"\n{'=' * 68}")
  print("  Generation examples (temperature=0.8)")
  print(f"{'=' * 68}")
  for q in test_questions:
  ans = generate_answer(
  model, q, stoi, itos,
  context_len=CONTEXT_LEN,
  max_new_chars=80,
  temperature=0.8,
  tau=final_tau,
  )
  print(f"  Q: {q}")
  print(f"  A: {ans}")
  print()

  # ── Summary ───────────────────────────────────────────────────────────
  print(f"{'=' * 68}")
  print("  ChatHBA Summary")
  print(f"{'=' * 68}")
  print(f"  params  : {model.param_count():,}")
  print(f"  vocab_size  : {vocab_size}")
  print(f"  best val PPL : {best_ppl:.4f}  (ep{best_epoch})")
  print(f"  final PPL  : {final_ppl:.4f}")
  print(f"  mask entropy : {avg_entropy:.6f}")
  print(f"  train time  : {train_time:.1f}s")
  print(f"  checkpoint  : {CHECKPOINT_PATH}")
  print()

  # PPL curve
  print(f"  {'Epoch':>6} {'ValPPL':>10} {'Note':>8}")
  print(f"  {'-' * 28}")
  for ep, ppl in ppl_history:
  note = " <- BEST" if ep == best_epoch else ""
  print(f"  {ep:>6} {ppl:>10.4f}{note}")
  print()

if __name__ == "__main__":
  main()
