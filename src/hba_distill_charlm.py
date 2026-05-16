"""
HBA Distillation Char-LM
=========================
Teacher (TinyTransformerLM) -> Student (HBALanguageModel) Knowledge Distillation

参照: hba_charlm.py (HBA v2), dlgn_charlm.py (TinyTransformerLM)

蒸留方式:
  KL(student || teacher) + CE(student, true_y) のハイブリッド損失
  Hinton et al. (2015) 標準 — T^2 スケーリング付き KL

フロー:
  Stage 1: TinyTransformerLM を標準学習 (Teacher)
  Stage 2: HBALanguageModel を蒸留学習 (Student)
  - Teacher の soft target (温度 T の softmax) を KL で模倣
  - CE(student, hard label) で精度も担保
  - entropy 正則化 + gradient clip

期待値:
  Student soft PPL が TF PPL (4.86) に近づく (目標: 4.5〜4.8)
"""

import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Teacher: TinyTransformerLM を dlgn_charlm から import
from dlgn_charlm import TinyTransformerLM

# Student: HBALanguageModel + 補助クラスを hba_charlm から import
from hba_charlm import (
  HBALanguageModel,
  collapse_hba,
  calibrate_hard_threshold,
  compute_perplexity,
  measure_tokens_per_sec,
)

from dynamic_dlgn import TemperatureScheduler

# ---------------------------------------------------------------------------
# 0. Data preparation (hba_charlm と同一)
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
# 1. Distillation loss
# ---------------------------------------------------------------------------

def distillation_loss(
  student_logits: torch.Tensor,
  teacher_logits: torch.Tensor,
  true_y: torch.Tensor,
  T: float = 4.0,
  alpha: float = 0.5,
) -> torch.Tensor:
  """
  KL(student || teacher) + CE(student, true_y) のハイブリッド。

  KL term: T^2 スケーリング (Hinton 2015 標準)
  - 温度 T で softmax をなだらかにして暗知識を伝達
  - T^2 は勾配スケールを CE と揃えるための補正

  Args:
  student_logits: (B, V) — Student の生 logits
  teacher_logits: (B, V) — Teacher の生 logits (no_grad で取得済み)
  true_y:  (B,)  — ハードラベル
  T:  蒸留温度 (3〜5 が目安、高いほど soft target がなだらか)
  alpha: KL の重み (1-alpha が CE の重み)

  Returns:
  スカラー損失
  """
  # KL: log_softmax(student/T) vs softmax(teacher/T)
  kl = F.kl_div(
  F.log_softmax(student_logits / T, dim=-1),
  F.softmax(teacher_logits / T, dim=-1),
  reduction='batchmean',
  ) * (T * T)

  # CE: hard label
  ce = F.cross_entropy(student_logits, true_y)

  return alpha * kl + (1.0 - alpha) * ce

# ---------------------------------------------------------------------------
# 2. Teacher training loop
# ---------------------------------------------------------------------------

def train_teacher(
  model: TinyTransformerLM,
  X_train: torch.Tensor, Y_train: torch.Tensor,
  X_val: torch.Tensor, Y_val: torch.Tensor,
  epochs: int = 40,
  batch_size: int = 256,
  lr: float = 3e-3,
  log_every: int = 4,
) -> tuple:
  """
  TinyTransformerLM の標準学習ループ。
  dlgn_charlm.train_model と同等だが、best checkpoint + early stopping 付き。

  Returns:
  ppl_history, best_ppl, best_epoch
  """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  ppl_history = 
  best_ppl = float('inf')
  best_state = None
  best_epoch = -1

  print(f"\n  {'Ep':>4} {'TrainLoss':>10} {'ValPPL':>9} {'Best?':>6}")
  print(f"  {'-' * 35}")

  for epoch in range(epochs):
  model.train()
  perm = torch.randperm(len(X_train))
  X_s, Y_s = X_train[perm], Y_train[perm]

  ep_loss, n_b = 0.0, 0
  for i in range(0, len(X_train), batch_size):
  xb = X_s[i:i + batch_size]
  yb = Y_s[i:i + batch_size]
  optimizer.zero_grad()
  logits = model(xb)
  loss = F.cross_entropy(logits, yb)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  ep_loss += loss.item()
  n_b += 1

  lr_sched.step()

  if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
  ppl = compute_perplexity(model, X_val, Y_val)
  ppl_history.append((epoch + 1, ppl))

  is_best = ppl < best_ppl
  if is_best:
  best_ppl = ppl
  best_epoch = epoch + 1
  best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

  best_mark = " *" if is_best else ""
  print(
  f"  {epoch+1:>4} {ep_loss/n_b:>10.4f} {ppl:>9.4f}{best_mark}",
  flush=True,
  )

  if best_state is not None:
  model.load_state_dict(best_state)
  print(f"\n  [teacher best] restored ep{best_epoch}  PPL={best_ppl:.4f}")
  else:
  print("\n  [teacher best] no checkpoint saved")

  return ppl_history, best_ppl, best_epoch

# ---------------------------------------------------------------------------
# 3. Student distillation training loop
# ---------------------------------------------------------------------------

def train_student_distill(
  student: HBALanguageModel,
  teacher: TinyTransformerLM,
  X_train: torch.Tensor, Y_train: torch.Tensor,
  X_val: torch.Tensor, Y_val: torch.Tensor,
  epochs: int = 60,
  batch_size: int = 256,
  lr: float = 3e-3,
  patience: int = 5,
  warm_hold: float = 0.2,
  distill_T: float = 4.0,
  distill_alpha: float = 0.5,
  lambda_ent: float = 0.01,
  log_every: int = 2,
  best_after_anneal: bool = False,
) -> tuple:
  """
  HBALanguageModel の蒸留学習ループ。

  損失:
  loss = distillation_loss(student_logits, teacher_logits, y, T, alpha)
  - lambda_ent * mean_router_entropy  <- entropy 正則化

  Best checkpoint + Early stopping は hba_charlm.train_hba と同等。

  Args:
  best_after_anneal: True のとき、tau が t_end * 1.5 以下になった後の
  epoch のみ best 更新候補とする。warm + アニーリング
  期間中は best 更新しない。硬化済み状態での best を
  確実に得たい場合に使用。

  Returns:
  ppl_history, best_ppl, best_epoch
  """
  teacher.eval()  # Teacher は frozen (gradient なし)

  optimizer = torch.optim.Adam(student.parameters(), lr=lr)
  lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  temp_sched = TemperatureScheduler(
  t_start=1.0, t_end=0.1,
  total_epochs=epochs,
  warm_hold_frac=warm_hold,
  )

  ppl_history = 
  best_ppl = float('inf')
  best_state = None
  best_epoch = -1
  no_improve_count = 0

  # best_after_anneal: tau が t_end * 1.5 以下になった後のみ best 更新
  anneal_thresh = temp_sched.t_end * 1.5

  print(f"\n  {'Ep':>4} {'tau':>6} {'DistLoss':>9} {'EntReg':>9} {'TotLoss':>9} "
  f"{'ValPPL':>9} {'AvgMaskH':>10} {'Best?':>6}")
  if best_after_anneal:
  print(f"  [best_after_anneal=ON] best 更新は tau <= {anneal_thresh:.3f} 以降")
  print(f"  {'-' * 76}")

  for epoch in range(epochs):
  tau = temp_sched.temperature
  student.train()
  perm = torch.randperm(len(X_train))
  X_s, Y_s = X_train[perm], Y_train[perm]

  ep_dist, ep_ent_reg, ep_total, n_b = 0.0, 0.0, 0.0, 0

  for i in range(0, len(X_train), batch_size):
  xb = X_s[i:i + batch_size]
  yb = Y_s[i:i + batch_size]
  optimizer.zero_grad()

  # Teacher の soft target (frozen, no grad)
  with torch.no_grad():
  teacher_logits = teacher(xb)  # (B, V)

  # Student の logits
  student_logits = student(xb, tau=tau)  # (B, V)

  # 蒸留損失
  dist = distillation_loss(
  student_logits, teacher_logits, yb,
  T=distill_T, alpha=distill_alpha,
  )

  # Entropy 正則化
  entropies = student.get_mask_entropies()
  mean_entropy = torch.stack(entropies).mean()
  ent_reg = -lambda_ent * mean_entropy

  loss = dist + ent_reg

  loss.backward()
  # gradient clip
  torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
  optimizer.step()

  ep_dist  += dist.item()
  ep_ent_reg += ent_reg.item()
  ep_total  += loss.item()
  n_b  += 1

  lr_sched.step()
  temp_sched.step()

  if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
  # 学習中 eval: 現在の温度で評価（温度整合）
  tau_disp = temp_sched.temperature
  ppl = compute_perplexity(student, X_val, Y_val, temperature=tau_disp)
  ppl_history.append((epoch + 1, tau_disp, ppl))

  with torch.no_grad():
  _ = student(X_val[:min(256, len(X_val))], tau=tau_disp)
  entropies = student.get_mask_entropies()
  avg_mask_h = torch.stack(entropies).mean().item()

  # best_after_anneal: アニーリング後半 (tau <= t_end * 1.5) のみ best 候補
  eligible_for_best = (not best_after_anneal) or (tau_disp <= anneal_thresh)

  is_best = eligible_for_best and (ppl < best_ppl)
  if is_best:
  best_ppl = ppl
  best_epoch = epoch + 1
  best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
  no_improve_count = 0
  elif eligible_for_best:
  # best 候補 epoch で改善なし → patience カウント
  no_improve_count += 1

  if best_after_anneal and not eligible_for_best:
  best_mark = " (warm)"
  else:
  best_mark = " *" if is_best else ""
  print(
  f"  {epoch+1:>4} {tau_disp:>6.3f} "
  f"{ep_dist/n_b:>9.4f} {ep_ent_reg/n_b:>9.4f} "
  f"{ep_total/n_b:>9.4f} {ppl:>9.4f} {avg_mask_h:>10.6f}{best_mark}",
  flush=True,
  )

  if patience > 0 and eligible_for_best and no_improve_count >= patience:
  print(
  f"\n  [early stop] no improvement for {patience} log-epochs "
  f"(best={best_ppl:.4f} @ ep{best_epoch}). Stopping."
  )
  break

  if best_state is not None:
  student.load_state_dict(best_state)
  print(f"\n  [student best] restored ep{best_epoch}  PPL={best_ppl:.4f}")
  else:
  print("\n  [student best] no checkpoint saved")

  return ppl_history, best_ppl, best_epoch

# ---------------------------------------------------------------------------
# 4. Text generation (hba_charlm と同一インタフェース)
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
# 5. Main
# ---------------------------------------------------------------------------

def main():
  torch.manual_seed(42)

  print("=" * 72)
  print("  HBA Distillation Char-LM")
  print("  Teacher: TinyTransformerLM -> Student: HBALanguageModel")
  print(f"  Device: CPU  PyTorch {torch.__version__}")
  print("=" * 72)

  # ── CLI 引数 ────────────────────────────────────────────────────────────
  EPOCHS_TEACHER  = 40
  EPOCHS_STUDENT  = 60
  PATIENCE  = 5
  DISTILL_T  = 4.0
  DISTILL_ALPHA  = 0.5
  N_LAYERS  = 3
  HIDDEN_DIM  = 96
  N_HEADS  = 8
  LAMBDA_ENT  = 0.01
  WARM_HOLD  = 0.2
  AUTO_CALIBRATE  = True
  BEST_AFTER_ANNEAL = False

  for arg in sys.argv[1:]:
  if arg.startswith('--epochs-teacher='):
  EPOCHS_TEACHER = int(arg.split('=', 1)[1])
  print(f"[CLI] epochs_teacher: {EPOCHS_TEACHER}")
  elif arg.startswith('--epochs-student='):
  EPOCHS_STUDENT = int(arg.split('=', 1)[1])
  print(f"[CLI] epochs_student: {EPOCHS_STUDENT}")
  elif arg.startswith('--patience='):
  PATIENCE = int(arg.split('=', 1)[1])
  print(f"[CLI] patience: {PATIENCE}")
  elif arg.startswith('--distill-temp='):
  DISTILL_T = float(arg.split('=', 1)[1])
  print(f"[CLI] distill_temp: {DISTILL_T}")
  elif arg.startswith('--distill-alpha='):
  DISTILL_ALPHA = float(arg.split('=', 1)[1])
  print(f"[CLI] distill_alpha: {DISTILL_ALPHA}")
  elif arg.startswith('--n-layers='):
  N_LAYERS = int(arg.split('=', 1)[1])
  print(f"[CLI] n_layers: {N_LAYERS}")
  elif arg.startswith('--hidden-dim='):
  HIDDEN_DIM = int(arg.split('=', 1)[1])
  print(f"[CLI] hidden_dim: {HIDDEN_DIM}")
  elif arg.startswith('--n-heads='):
  N_HEADS = int(arg.split('=', 1)[1])
  print(f"[CLI] n_heads: {N_HEADS}")
  elif arg.startswith('--lambda-ent='):
  LAMBDA_ENT = float(arg.split('=', 1)[1])
  print(f"[CLI] lambda_ent: {LAMBDA_ENT}")
  elif arg.startswith('--warm-hold='):
  WARM_HOLD = float(arg.split('=', 1)[1])
  print(f"[CLI] warm_hold: {WARM_HOLD}")
  elif arg == '--no-calibrate':
  AUTO_CALIBRATE = False
  print("[CLI] auto_calibrate: OFF")
  elif arg == '--auto-calibrate':
  AUTO_CALIBRATE = True
  print("[CLI] auto_calibrate: ON")
  elif arg == '--best-after-anneal':
  BEST_AFTER_ANNEAL = True
  print("[CLI] best_after_anneal: ON")

  BATCH  = 256
  LR  = 3e-3
  CONTEXT_LEN = 16

  # ── Data ─────────────────────────────────────────────────────────────────
  text = load_text()
  stoi, itos = build_vocab(text)
  vocab_size = len(stoi)
  print(f"\n[vocab] size={vocab_size}  text_len={len(text):,}")

  split = int(len(text) * 0.9)
  X_train, Y_train = make_dataset(text[:split], stoi, CONTEXT_LEN)
  X_val,  Y_val  = make_dataset(text[split:],  stoi, CONTEXT_LEN)
  print(f"[data] train={len(X_train):,}  val={len(X_val):,}  "
  f"context_len={CONTEXT_LEN}")

  # =========================================================================
  # Stage 1: Teacher (TinyTransformerLM) 学習
  # =========================================================================
  print(f"\n{'=' * 72}")
  print(f"  [Stage 1] Teacher (TinyTransformerLM) training")
  print(f"  epochs={EPOCHS_TEACHER}  batch={BATCH}  lr={LR}")
  print(f"{'=' * 72}")

  teacher = TinyTransformerLM(
  vocab_size=vocab_size,
  context_len=CONTEXT_LEN,
  d_model=64,
  n_heads=2,
  n_layers=2,
  dropout=0.1,
  )
  print(f"[Teacher] params: {teacher.param_count():,}  "
  f"(d_model=64, n_heads=2, n_layers=2)")

  t0_teacher = time.perf_counter()
  _, teacher_best_ppl, teacher_best_epoch = train_teacher(
  teacher, X_train, Y_train, X_val, Y_val,
  epochs=EPOCHS_TEACHER,
  batch_size=BATCH,
  lr=LR,
  )
  teacher_train_time = time.perf_counter() - t0_teacher
  teacher.eval()

  # best restored モデルで最終 PPL 確定
  teacher_final_ppl = compute_perplexity(teacher, X_val, Y_val)
  print(f"\n  Teacher training time: {teacher_train_time:.1f}s "
  f"({teacher_train_time/60:.1f}min)")
  print(f"  Teacher PPL: {teacher_final_ppl:.4f}  "
  f"(best ep{teacher_best_epoch})")

  # =========================================================================
  # Stage 2: Student (HBALanguageModel) 蒸留学習
  # =========================================================================
  print(f"\n{'=' * 72}")
  print(f"  [Stage 2] Student (HBALanguageModel) distillation")
  print(f"  epochs={EPOCHS_STUDENT}  patience={PATIENCE}  lr={LR}")
  print(f"  distill_T={DISTILL_T}  distill_alpha={DISTILL_ALPHA}")
  print(f"  lambda_ent={LAMBDA_ENT}  warm_hold={WARM_HOLD}")
  print(f"  best_after_anneal={BEST_AFTER_ANNEAL}")
  print(f"  Student arch: hidden_dim={HIDDEN_DIM}  "
  f"n_layers={N_LAYERS}  n_heads={N_HEADS}  (HBA v2 仕様)")
  print(f"{'=' * 72}")

  student = HBALanguageModel(
  vocab_size=vocab_size,
  context_len=CONTEXT_LEN,
  hidden_dim=HIDDEN_DIM,
  n_layers=N_LAYERS,
  n_heads=N_HEADS,
  )
  print(f"[Student] params: {student.param_count():,}")

  t0_student = time.perf_counter()
  student_ppl_history, student_best_ppl, student_best_epoch = train_student_distill(
  student, teacher,
  X_train, Y_train, X_val, Y_val,
  epochs=EPOCHS_STUDENT,
  batch_size=BATCH,
  lr=LR,
  patience=PATIENCE,
  warm_hold=WARM_HOLD,
  distill_T=DISTILL_T,
  distill_alpha=DISTILL_ALPHA,
  lambda_ent=LAMBDA_ENT,
  best_after_anneal=BEST_AFTER_ANNEAL,
  )
  student_train_time = time.perf_counter() - t0_student
  print(f"\n  Student training time: {student_train_time:.1f}s "
  f"({student_train_time/60:.1f}min)")

  # best restored モデルで最終 soft PPL
  # 修正: final_tau は best_epoch 時点の実際の温度を使用（固定 0.1 は廃止）
  # ppl_history から best_epoch の tau を逆引き
  best_tau = 1.0  # fallback
  for ep_h, tau_h, _ppl_h in student_ppl_history:
  if ep_h == student_best_epoch:
  best_tau = tau_h
  break
  final_tau = best_tau
  print(f"\n  [final eval] best_epoch={student_best_epoch}  "
  f"best_tau={final_tau:.4f}  (fixed 0.1 廃止: 実温度で再評価)")
  final_ppl_soft = compute_perplexity(student, X_val, Y_val, temperature=final_tau)

  # ── ルーター entropy 確認 ────────────────────────────────────────────────
  with torch.no_grad():
  _ = student(X_val[:256], tau=final_tau)
  entropies = student.get_mask_entropies()
  final_avg_entropy = torch.stack(entropies).mean().item()

  print(f"\n[Router Stats]")
  print(f"  Final mask entropy (student best): {final_avg_entropy:.6f}")
  collapse_thresh = 0.01
  if final_avg_entropy < collapse_thresh:
  print(f"  WARNING: ルーター collapse の疑い (entropy < {collapse_thresh})")
  else:
  print(f"  OK: ルーター entropy は健全範囲")

  # ── Inference speed ──────────────────────────────────────────────────────
  sample_x = X_val[:512]
  speed_teacher = measure_tokens_per_sec(teacher, sample_x, use_temp=False)
  speed_student_soft = measure_tokens_per_sec(
  student, sample_x, use_temp=True, tau=final_tau
  )

  # ── Hard threshold calibration ───────────────────────────────────────────
  if AUTO_CALIBRATE:
  best_threshold, best_hard_ppl, _ = calibrate_hard_threshold(
  student, X_val, Y_val, batch_size=BATCH
  )
  print(f"  [calibrate] Adopted threshold: {best_threshold:.2f}  "
  f"Hard PPL: {best_hard_ppl:.4f}")
  else:
  best_threshold = 0.0
  print(f"  [calibrate] Skipped (--no-calibrate). Using threshold=0.0")

  # ── Hard collapse ────────────────────────────────────────────────────────
  print(f"\n[collapse] Student -> HardHBA "
  f"(threshold={best_threshold:.2f})...")
  hard_student = collapse_hba(student, threshold=best_threshold)
  hard_student.eval()

  speed_student_hard = measure_tokens_per_sec(hard_student, sample_x, use_temp=False)

  total_loss, n_b = 0.0, 0
  with torch.no_grad():
  for i in range(0, len(X_val), BATCH):
  xb = X_val[i:i + BATCH]
  yb = Y_val[i:i + BATCH]
  logits = hard_student(xb)
  total_loss += F.cross_entropy(logits, yb).item()
  n_b += 1
  ppl_hard = math.exp(total_loss / max(1, n_b))

  if AUTO_CALIBRATE:
  best_hard_ppl = ppl_hard  # 最終確認値で上書き

  # ── Text generation ──────────────────────────────────────────────────────
  seeds = ["First ", "To be ", "The ki"]

  print(f"\n{'=' * 72}")
  print("  Generated text: Student (HBA distilled) soft - greedy")
  print(f"{'=' * 72}")
  for seed in seeds:
  gen = generate_text(
  student, seed, stoi, itos, CONTEXT_LEN,
  n_chars=60, temperature=0.0, use_model_tau=True, tau=final_tau,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 72}")
  print("  Generated text: Student (HBA distilled) soft - sampled (temp=0.8)")
  print(f"{'=' * 72}")
  for seed in seeds:
  gen = generate_text(
  student, seed, stoi, itos, CONTEXT_LEN,
  n_chars=60, temperature=0.8, use_model_tau=True, tau=final_tau,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 72}")
  print(f"  Generated text: Student Hard (threshold={best_threshold:.2f})")
  print(f"{'=' * 72}")
  for seed in seeds:
  gen = generate_text(
  hard_student, seed, stoi, itos, CONTEXT_LEN,
  n_chars=60, temperature=0.8, use_model_tau=False,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  print(f"\n{'=' * 72}")
  print("  Generated text: Teacher (TF) - sampled (temp=0.8)")
  print(f"{'=' * 72}")
  for seed in seeds:
  gen = generate_text(
  teacher, seed, stoi, itos, CONTEXT_LEN,
  n_chars=60, temperature=0.8, use_model_tau=False,
  )
  print(f"  [{seed!r:8}] -> {gen!r}")

  # ── 比較表 ───────────────────────────────────────────────────────────────
  print(f"\n{'=' * 72}")
  print("  HBA v2 vs HBA distilled vs Transformer (Teacher) - 比較表")
  print(f"{'=' * 72}")
  fmt_h = f"  {'Model':<36} {'Params':>8} {'TeacherPPL':>11} " \
  f"{'StudentSoftPPL':>15} {'StudentHardPPL':>15} {'Speed(tok/s)':>13}"
  print(fmt_h)
  print(f"  {'-' * 103}")

  # Teacher (TF)
  print(
  f"  {'Teacher (TF)':<36} {teacher.param_count():>8,} "
  f"{teacher_final_ppl:>11.4f} "
  f"{'-':>15} {'-':>15} "
  f"{speed_teacher:>13,.0f}"
  )
  # HBA distilled (soft)
  print(
  f"  {'HBA distilled (soft/best)':<36} {student.param_count():>8,} "
  f"{'-':>11} "
  f"{final_ppl_soft:>15.4f} {'-':>15} "
  f"{speed_student_soft:>13,.0f}"
  )
  # HBA distilled (hard)
  print(
  f"  {'HBA distilled (hard/calib thresh=' + f'{best_threshold:.2f})':<36} "
  f"{student.param_count():>8,} "
  f"{'-':>11} "
  f"{'-':>15} {ppl_hard:>15.4f} "
  f"{speed_student_hard:>13,.0f}"
  )
  # HBA v2 前回 (参考値)
  print(
  f"  {'HBA v2 [前回]':<36} {'296,000':>8} "
  f"{'-':>11} "
  f"{'5.3200':>15} {'6.5400':>15} "
  f"{'-':>13}"
  )
  # Transformer 前回
  print(
  f"  {'Transformer [前回参考]':<36} {'108,861':>8} "
  f"{'4.8600':>11} "
  f"{'-':>15} {'-':>15} "
  f"{'51,291':>13}"
  )

  # ── Perplexity curve (Student) ───────────────────────────────────────────
  print(f"\n{'=' * 72}")
  print("  Perplexity curve (Student distillation)")
  print(f"{'=' * 72}")
  print(f"  {'Epoch':>6} {'tau':>6} {'ValPPL':>10} {'Note':>8}")
  print(f"  {'-' * 35}")
  for ep, tau_h, ppl in student_ppl_history:
  note = " <- BEST" if ep == student_best_epoch else ""
  print(f"  {ep:>6} {tau_h:>6.3f} {ppl:>10.4f}{note}")

  # ── 分析 ─────────────────────────────────────────────────────────────────
  print(f"\n{'=' * 72}")
  print(f"{'=' * 72}")

  tf_ref_ppl = 4.86  # 前回の Teacher PPL 参考値
  if final_ppl_soft < tf_ref_ppl:
  verdict = "蒸留 Student が TF baseline (4.86) を超えた！知識移転成功"
  elif final_ppl_soft < 5.00:
  verdict = "蒸留 Student が TF に極めて近い (< 5.0)  蒸留効果あり"
  elif final_ppl_soft < 5.32:
  verdict = "蒸留 Student が HBA v2 (5.32) を超えた  蒸留で精度向上確認"
  elif final_ppl_soft < 5.40:
  verdict = "HBA v2 と同等  蒸留効果は限定的"
  else:
  verdict = "蒸留効果が出ていない  T/alpha チューニングが必要"

  print(f"  判定: {verdict}")
  print()
  print(f"  Teacher PPL (best restored):  {teacher_final_ppl:.4f}")
  print(f"  Student soft PPL (best restored):  {final_ppl_soft:.4f}")
  print(f"  Student hard PPL (thresh={best_threshold:.2f}):  {ppl_hard:.4f}")
  print()
  print(f"  HBA v2 soft PPL [前回]:  5.3200")
  print(f"  HBA v2 hard PPL [前回]:  6.5400")
  print(f"  TF (Teacher) PPL [前回参考]:  4.8600")
  print()
  print(f"  Teacher best epoch: {teacher_best_epoch}")
  print(f"  Student best epoch: {student_best_epoch}  "
  f"(patience={PATIENCE}  best_after_anneal={BEST_AFTER_ANNEAL})")
  print(f"  Student best tau:  {final_tau:.4f}  "
  f"(warm完了={'Yes' if final_tau <= 0.5 else 'No (warm 期間内 early stop)'})")
  print(f"  Final mask entropy: {final_avg_entropy:.6f}  "
  f"({'collapse あり' if final_avg_entropy < 0.01 else 'collapse なし'})")
  print(f"  Adopted hard threshold: {best_threshold:.2f}")
  print()
  print(f"  Teacher train time: {teacher_train_time:.1f}s "
  f"({teacher_train_time/60:.1f}min)")
  print(f"  Student train time: {student_train_time:.1f}s "
  f"({student_train_time/60:.1f}min)")
  print(f"  Total time:  "
  f"{(teacher_train_time+student_train_time):.1f}s "
  f"({(teacher_train_time+student_train_time)/60:.1f}min)")
  print()
  print(f"  Student soft speed:  {speed_student_soft:,.0f} tok/s")
  print(f"  Student hard speed:  {speed_student_hard:,.0f} tok/s  "
  f"({speed_student_hard/max(1,speed_student_soft):.2f}x soft)")
  print(f"  Teacher speed:  {speed_teacher:,.0f} tok/s")
  print()
  print("  [蒸留パラメータ]")
  print(f"  - distill_T={DISTILL_T}  distill_alpha={DISTILL_ALPHA}")
  print(f"  - lambda_ent={LAMBDA_ENT}  warm_hold={WARM_HOLD}")
  print(f"  - patience={PATIENCE}  best_after_anneal={BEST_AFTER_ANNEAL}")
  print(f"  - best checkpoint: 保存 + restore 済 (ep{student_best_epoch}  tau={final_tau:.4f})")
  print(f"  - eval 温度整合: 学習中 eval = 現在の tau (固定 1.0 廃止)")
  print()

if __name__ == "__main__":
  main()
