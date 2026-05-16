"""
Dynamic DLGN with Gumbel-Softmax input pairing selection.

# LLMへの転用メモ:
# 最終的にはfloat埋め込み→[0,1]正規化→Boolean化→このDLGNでトークン特徴抽出を想定。
# 動的ペアリングにより埋め込み空間の任意次元ペアを捕捉できる。
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 既存コアの再利用 ──────────────────────────────────────────────────────────
from dlgn import _soft_gates
from arithmetic_mdl import ArithmeticWeightManager

# ────────────────────────────────────────────────────────────────────────────
# 1. Gumbel-Softmax ユーティリティ
# ────────────────────────────────────────────────────────────────────────────

def gumbel_softmax_sample(logits: torch.Tensor, temperature: float,
  hard: bool = False,
  use_gumbel_noise: bool = True) -> torch.Tensor:
  """
  logits: (..., N)
  hard=False → soft one-hot (微分可能)
  hard=True  → STE (straight-through estimator)
  use_gumbel_noise=False → 純粋温度 softmax + STE (ペアリング学習向き)
  ランダムGumbelノイズを使うと毎バッチ違う入力を選んでしまい、
  信号が平均化されてペアリング勾配が消える。
  """
  if temperature <= 0:
  raise ValueError("temperature must be > 0")

  if use_gumbel_noise:
  gumbels = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
  y = (logits + gumbels) / temperature
  else:
  y = logits / temperature

  y_soft = F.softmax(y, dim=-1)

  if hard:
  # straight-through: forward=hard one-hot, backward=soft勾配
  idx = y_soft.argmax(dim=-1, keepdim=True)
  y_hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
  return y_hard - y_soft.detach() + y_soft
  return y_soft

# ────────────────────────────────────────────────────────────────────────────
# 2. DynamicGateLayer
# ────────────────────────────────────────────────────────────────────────────

class DynamicGateLayer(nn.Module):
  """
  各ゲートが「どの2入力を選ぶか」を Gumbel-Softmax で確率的に学習する層。

  Parameters
  ----------
  in_features : 前層の次元数 N
  num_neurons : このレイヤのゲート数 K
  temperature : Gumbel-Softmax 温度 (外部からアニーリング)
  init_scale  : gate_logits 初期化スケール
  init_pairing: 'random_hard' → 初期から1つを強くピーク化
  'uniform'  → 旧来の一様初期化
  seed  : 再現性用シード
  """

  def __init__(self, in_features: int, num_neurons: int,
  init_scale: float = 0.1,
  init_pairing: str = 'random_hard',
  seed: int = 0):
  super().__init__()
  self.in_features = in_features
  self.num_neurons = num_neurons

  # ゲート種類の選択 logit (K, 16)
  self.gate_logits = nn.Parameter(
  torch.randn(num_neurons, 16) * init_scale
  )

  # 入力 A/B 選択 logit (K, N)
  # 'random_hard': 各ニューロンに対してランダムな1インデックスを強くピーク化
  #  → softmax がその入力を強く選んだ状態から学習開始
  #  → 対称性が初期に破れているので勾配が流れやすい
  rng = torch.Generator()
  rng.manual_seed(seed)

  if init_pairing == 'random_hard':
  # 各ニューロンにランダムな1入力を強くピーク化
  peak = 6.0  # 強い初期ピーク → 初期から明確な選択
  logits_a = torch.full((num_neurons, in_features), -peak * 0.1)
  idx_a_init = torch.randint(0, in_features, (num_neurons,), generator=rng)
  logits_a.scatter_(1, idx_a_init.unsqueeze(1), peak)
  logits_a += torch.randn(num_neurons, in_features, generator=rng) * 0.05

  logits_b = torch.full((num_neurons, in_features), -peak * 0.1)
  idx_b_init = (idx_a_init + 1) % in_features
  logits_b.scatter_(1, idx_b_init.unsqueeze(1), peak)
  logits_b += torch.randn(num_neurons, in_features, generator=rng) * 0.05

  elif init_pairing == 'guided_tournament':
  # XOR二分木の構造を初期化のバイアスとして仕込む。
  # ニューロンkは (2k mod N, 2k+1 mod N) を初期の強選択とする。
  # 但し完全固定ではなく周辺にノイズを入れ学習の余地を残す。
  peak = 6.0
  pairs_per = max(1, in_features // 2)
  logits_a = torch.full((num_neurons, in_features), -peak * 0.1)
  logits_b = torch.full((num_neurons, in_features), -peak * 0.1)
  for k in range(num_neurons):
  pair = k % pairs_per
  ia = (pair * 2) % in_features
  ib = (pair * 2 + 1) % in_features
  logits_a[k, ia] = peak
  logits_b[k, ib] = peak
  logits_a += torch.randn(num_neurons, in_features, generator=rng) * 0.05
  logits_b += torch.randn(num_neurons, in_features, generator=rng) * 0.05

  else:
  logits_a = torch.randn(num_neurons, in_features, generator=rng) * init_scale
  logits_b = torch.randn(num_neurons, in_features, generator=rng) * init_scale

  self.input_a_logits = nn.Parameter(logits_a)
  self.input_b_logits = nn.Parameter(logits_b)

  def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
  """
  x  : (batch, in_features) ∈ [0,1]
  temperature : Gumbel-Softmax 温度 τ

  入力選択は Straight-Through Estimator (STE) を使う。
  [なぜ加重平均ではダメか]
  w_a @ x で全入力の平均になると a ≈ E[x] ≈ 0.5 に収束し、
  XOR(a,b) ≈ 0.5 が常に成立してペアリング勾配が消える。
  STE なら forward は離散選択 → 真のバイナリ値が伝わる。
  backward は softmax の勾配 → ペアリング重みへの有効な勾配が流れる。

  Returns: (batch, num_neurons) ∈ [0,1]
  """
  # STE で入力選択: Gumbelノイズなし (ペアリング勾配を安定化)
  # warm期間(τ高): softに近い→探索; anneal期間(τ低): argmaxに近い→収束
  sel_a = gumbel_softmax_sample(self.input_a_logits, temperature,
  hard=True, use_gumbel_noise=False)
  sel_b = gumbel_softmax_sample(self.input_b_logits, temperature,
  hard=True, use_gumbel_noise=False)

  # STE で選ばれた入力値を取得
  # x: (B, N), sel_a: (K, N) → a[b,k] = x[b, argmax_a[k]]
  # = sel_a @ x.T を転置: (B, K)
  a = (sel_a @ x.t()).t()  # (B, K)
  b = (sel_b @ x.t()).t()  # (B, K)

  # 16種ゲートの soft 出力 (16, B, K)
  gate_out = _soft_gates(a, b)  # (16, B, K)
  gate_out = gate_out.permute(1, 2, 0)  # (B, K, 16)

  # ゲート種類の soft 重み (K, 16)
  p_gate = F.softmax(self.gate_logits, dim=-1)  # (K, 16)

  # y_k = Σ_g p_g * f_g(a, b)
  y = (gate_out * p_gate.unsqueeze(0)).sum(dim=-1)  # (B, K)
  return y

  def collapse(self) -> 'HardDynamicGateLayer':
  """
  argmax で入力ペアとゲート種類を固定し、推論専用 layer を返す。
  """
  with torch.no_grad():
  idx_a = self.input_a_logits.argmax(dim=-1)  # (K,)
  idx_b = self.input_b_logits.argmax(dim=-1)  # (K,)
  gate_ids = self.gate_logits.argmax(dim=-1)  # (K,)
  return HardDynamicGateLayer(
  self.in_features, self.num_neurons,
  idx_a.clone(), idx_b.clone(), gate_ids.clone()
  )

  def pairing_entropy(self) -> torch.Tensor:
  """入力選択分布のエントロピー (ビット)。0に近いほどペアが確定している。"""
  pa = F.softmax(self.input_a_logits, dim=-1)
  pb = F.softmax(self.input_b_logits, dim=-1)
  ha = -(pa * torch.log(pa + 1e-10)).sum(-1) / math.log(2)
  hb = -(pb * torch.log(pb + 1e-10)).sum(-1) / math.log(2)
  return (ha + hb).mean()

# ────────────────────────────────────────────────────────────────────────────
# 3. HardDynamicGateLayer (collapse後)
# ────────────────────────────────────────────────────────────────────────────

class HardDynamicGateLayer(nn.Module):
  """collapse後の固定ペア・固定ゲートによるハード推論層"""

  def __init__(self, in_features, num_neurons, idx_a, idx_b, gate_ids):
  super().__init__()
  self.in_features = in_features
  self.num_neurons = num_neurons
  self.register_buffer('idx_a', idx_a)
  self.register_buffer('idx_b', idx_b)
  self.register_buffer('gate_ids', gate_ids)

  @torch.no_grad()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
  a = x[:, self.idx_a]  # (B, K)
  b = x[:, self.idx_b]  # (B, K)
  gate_out = _soft_gates(a, b)  # (16, B, K)
  gate_out = gate_out.permute(1, 2, 0)  # (B, K, 16)
  idx = self.gate_ids.unsqueeze(0).unsqueeze(-1).expand(
  gate_out.shape[0], -1, 1
  )
  y = gate_out.gather(2, idx).squeeze(-1)
  return (y > 0.5).float()

# ────────────────────────────────────────────────────────────────────────────
# 4. DynamicDLGNModel
# ────────────────────────────────────────────────────────────────────────────

class DynamicDLGNModel(nn.Module):
  """
  DynamicGateLayer を複数段積んだモデル。
  温度アニーリングは外部スケジューラ (TemperatureScheduler) で管理する。
  """

  def __init__(self, in_features: int, hidden_neurons: list,
  num_classes: int = 2, init_scale: float = 0.1,
  init_pairing: str = 'random_hard', seed_base: int = 0):
  super().__init__()
  self.gate_layers = nn.ModuleList()
  prev = in_features
  for i, n in enumerate(hidden_neurons):
  self.gate_layers.append(
  DynamicGateLayer(prev, n, init_scale=init_scale,
  init_pairing=init_pairing,
  seed=seed_base + i)
  )
  prev = n
  self.head = nn.Linear(prev, num_classes)

  def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
  for layer in self.gate_layers:
  x = layer(x, temperature=temperature)
  return self.head(x)

  def collapse(self) -> 'HardDynamicDLGNModel':
  hard_layers = [layer.collapse() for layer in self.gate_layers]
  return HardDynamicDLGNModel(hard_layers, self.head)

  def pairing_entropy(self) -> float:
  """全層の平均ペア選択エントロピー (ビット)"""
  total = 0.0
  for layer in self.gate_layers:
  total += layer.pairing_entropy().item()
  return total / max(1, len(self.gate_layers))

  def mdl_code_length_diff(self) -> torch.Tensor:
  """微分可能なコード長近似 (gate種類のエントロピー和)"""
  total = torch.tensor(0.0)
  for layer in self.gate_layers:
  p = F.softmax(layer.gate_logits, dim=-1)
  h = -(p * torch.log(p + 1e-10)).sum(-1) / math.log(2)
  total = total + h.sum()
  return total

class HardDynamicDLGNModel(nn.Module):
  def __init__(self, hard_layers, head):
  super().__init__()
  self.hard_layers = nn.ModuleList(hard_layers)
  self.head = head

  @torch.no_grad()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
  for layer in self.hard_layers:
  x = layer(x)
  return self.head(x)

# ────────────────────────────────────────────────────────────────────────────
# 5. 温度アニーリングスケジューラ
# ────────────────────────────────────────────────────────────────────────────

class TemperatureScheduler:
  """
  2フェーズ温度スケジューラ。
  Phase1 (warm_hold_frac): τ = t_start で固定 (ペアリング探索期間)
  Phase2 残り  : τ を t_start → t_end に指数降下 (収束期間)

  warm_hold_frac=0.25 かつ total=120 なら:
  epoch 0-29: τ=1.0 固定 → Gumbel がランダム性を持って探索
  epoch 30-119: τ 1.0→0.05 指数降下 → 選択を固める
  """

  def __init__(self, t_start: float = 1.0, t_end: float = 0.05,
  total_epochs: int = 100, warm_hold_frac: float = 0.25):
  self.t_start = t_start
  self.t_end = t_end
  self.total = total_epochs
  self.warm_epochs = int(total_epochs * warm_hold_frac)
  self._epoch = 0

  def step(self):
  self._epoch += 1

  @property
  def temperature(self) -> float:
  if self._epoch <= self.warm_epochs:
  return self.t_start
  anneal_epochs = max(1, self.total - self.warm_epochs)
  frac = min((self._epoch - self.warm_epochs) / anneal_epochs, 1.0)
  return self.t_start * (self.t_end / self.t_start) ** frac

# ────────────────────────────────────────────────────────────────────────────
# 6. データ生成
# ────────────────────────────────────────────────────────────────────────────

def generate_parity(n: int, n_bits: int, seed: int):
  rng = torch.Generator()
  rng.manual_seed(seed)
  x = torch.randint(0, 2, (n, n_bits), generator=rng).float()
  y = x.sum(dim=-1).long() % 2
  return x, y

# ────────────────────────────────────────────────────────────────────────────
# 7. 算術符号MDL (DynamicDLGNModel 対応版)
# ────────────────────────────────────────────────────────────────────────────

def sync_arithmetic_mdl(model: DynamicDLGNModel,
  wm: ArithmeticWeightManager) -> None:
  """DynamicDLGNModel の gate_logits から ArithmeticWeightManager を更新"""
  wm.reset()
  gate_idx = 0
  for layer in model.gate_layers:
  with torch.no_grad():
  probs = F.softmax(layer.gate_logits, dim=-1)  # (K, 16)
  hard = probs.argmax(dim=-1)  # (K,)
  for k in range(layer.num_neurons):
  wm.update(gate_idx, hard[k].item(), probs[k].tolist())
  gate_idx += 1

# ────────────────────────────────────────────────────────────────────────────
# 8. スループット計測
# ────────────────────────────────────────────────────────────────────────────

def measure_throughput(model, x: torch.Tensor,
  temperature: float = None, runs: int = 20) -> float:
  with torch.no_grad():
  for _ in range(3):
  if temperature is not None:
  model(x, temperature)
  else:
  model(x)
  t0 = time.perf_counter()
  for _ in range(runs):
  if temperature is not None:
  model(x, temperature)
  else:
  model(x)
  return x.shape[0] * runs / (time.perf_counter() - t0)

# ────────────────────────────────────────────────────────────────────────────
# 9. train_64bit_parity
# ────────────────────────────────────────────────────────────────────────────

def train_64bit_parity(
  n_bits: int = 64,
  hidden_neurons: list = None,
  n_train: int = 8192,
  n_test: int = 2048,
  batch_size: int = 512,
  epochs: int = 150,
  lr: float = 0.02,
  lambda_mdl: float = 1e-4,
  lambda_diversity: float = 1e-3,
  t_start: float = 1.0,
  t_end: float = 0.05,
  warm_hold_frac: float = 0.25,
  seed: int = 42,
  init_scale: float = 0.1,
  init_pairing: str = 'random_hard',
):
  """
  64-bit パリティを動的ペアリング DLGN で学習する。

  アーキテクチャ戦略:
  - log2(64)=6 の XOR 連鎖を学習するには最低 6 段必要。
  - 動的ペアリングは各層ニューロンが任意入力を選べるので
  固定ペアより自由度が高い。
  - ただし各層の幅を広くとって「チャンス」を増やす必要がある。
  - デフォルト: [128, 64, 32, 16, 8, 4] (6段ピラミッド)

  温度アニーリング:
  Phase1 (warm_hold): τ = t_start 固定 → ペアリング探索
  Phase2 (anneal)  : τ を t_start → t_end に指数降下 → 収束

  ペアリング多様性損失:
  各層内で異なるニューロンが同じ入力を選ぶのを抑制する。
  同一インデックス選択が集中するとネットワークが無意味に縮退する。
  """
  torch.manual_seed(seed)

  if hidden_neurons is None:
  # 6段ピラミッド: XOR連鎖の深さを確保しつつ幅で学習を安定化
  hidden_neurons = [128, 64, 32, 16, 8, 4]

  x_train, y_train = generate_parity(n_train, n_bits, seed=seed)
  x_test, y_test = generate_parity(n_test, n_bits, seed=seed + 1)

  model = DynamicDLGNModel(
  in_features=n_bits,
  hidden_neurons=hidden_neurons,
  num_classes=2,
  init_scale=init_scale,
  init_pairing=init_pairing,
  seed_base=seed,
  )
  total_neurons = sum(hidden_neurons)

  print(f"\n{'=' * 68}")
  print(f"  Dynamic DLGN -- {n_bits}-bit Parity")
  print(f"  arch={n_bits}->{hidden_neurons}->2")
  print(f"  total_neurons={total_neurons}  epochs={epochs}")
  print(f"  tau: {t_start}->{t_end} (warm_hold={warm_hold_frac:.0%})"
  f"  lr={lr}  lambda_mdl={lambda_mdl}  lambda_div={lambda_diversity}")
  print(f"  init_pairing={init_pairing}  init_scale={init_scale}")
  print(f"  train={n_train}  test={n_test}  "
  f"parity1_ratio={y_train.float().mean():.3f}")
  print(f"{'=' * 68}")

  # 2段階学習 (Progressive Training):
  # Phase1 (warm_hold): ペアリング固定 + ゲート種類/headのみ学習
  # Phase2 (anneal)  : 全パラメータ解放 + ペアリング高lr
  # 理由: ペアリングとゲート種類を同時に探索すると干渉する。
  # まずゲートを固定ペアで収束させ、次にペアリングを微調整する。
  scheduler_tau = TemperatureScheduler(t_start, t_end, epochs, warm_hold_frac)
  warm_epochs = scheduler_tau.warm_epochs

  pairing_params = 
  gate_params = 
  head_params = list(model.head.parameters())
  for layer in model.gate_layers:
  pairing_params += [layer.input_a_logits, layer.input_b_logits]
  gate_params.append(layer.gate_logits)

  # Phase1用: ペアリング凍結
  for p in pairing_params:
  p.requires_grad = False

  optimizer = torch.optim.Adam([
  {'params': gate_params,  'lr': lr},
  {'params': head_params,  'lr': lr},
  ])
  scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
  optimizer, T_max=epochs
  )
  _phase2_switched = False
  criterion = nn.CrossEntropyLoss()
  wm = ArithmeticWeightManager(num_gates=total_neurons)

  log_every = max(1, epochs // 12)
  code_len_history = 

  print(f"\n  {'Ep':>5} {'tau':>6} {'CE':>8} {'TotL':>8} "
  f"{'CodeLen':>10} {'PairEnt':>9} {'TrAcc':>7} {'TeAcc':>7}")
  print(f"  {'-' * 66}")

  def pairing_diversity_loss(layer: DynamicGateLayer) -> torch.Tensor:
  """
  ペアリング「行」の集中度を上げる（逆エントロピー最大化）。
  各ニューロンが確実に1つの入力を選ぶよう促す損失。
  -max_prob の最大化 = エントロピー最小化 = 選択の確定化。

  注: 以前の実装は「列」の多様化で誤って分散を促していた。
  ここでは「行」の集中度(max確率の平均)を上げることで
  ペアリングが argmax に近づくよう促す。
  """
  pa = F.softmax(layer.input_a_logits, dim=-1)  # (K, N)
  pb = F.softmax(layer.input_b_logits, dim=-1)  # (K, N)
  # -Σ max_k(p) → 最大化することでピークを育てる (lossとして返すので符号反転)
  concentration = -(pa.max(dim=-1).values.mean() + pb.max(dim=-1).values.mean())
  return concentration

  for epoch in range(epochs):
  tau = scheduler_tau.temperature
  model.train()
  perm = torch.randperm(n_train)
  xs, ys = x_train[perm], y_train[perm]
  ep_ce, ep_loss, n_b = 0.0, 0.0, 0

  # Phase2への切り替え: warm_epochs経過後にペアリングを解放
  is_warm = (epoch < warm_epochs)
  if not is_warm and not _phase2_switched:
  _phase2_switched = True
  for p in pairing_params:
  p.requires_grad = True
  # Phase2用: 全パラメータ + ペアリング高lr
  optimizer = torch.optim.Adam([
  {'params': pairing_params, 'lr': lr * 3.0},
  {'params': gate_params,  'lr': lr * 0.3},
  {'params': head_params,  'lr': lr * 0.3},
  ])
  # Phase2のスケジューラはリセット
  scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
  optimizer, T_max=epochs - warm_epochs
  )

  eff_lambda_mdl = 0.0 if is_warm else lambda_mdl
  eff_lambda_div = 0.0 if is_warm else lambda_diversity

  for i in range(0, n_train, batch_size):
  xb, yb = xs[i:i + batch_size], ys[i:i + batch_size]
  optimizer.zero_grad()
  logits = model(xb, temperature=tau)
  ce = criterion(logits, yb)
  mdl_cl = model.mdl_code_length_diff()
  div_l = sum(pairing_diversity_loss(layer)
  for layer in model.gate_layers)
  loss = ce + eff_lambda_mdl * mdl_cl + eff_lambda_div * div_l
  loss.backward()
  # 勾配クリップ: 動的ペアリングは初期に勾配が不安定になりやすい
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
  optimizer.step()
  ep_ce += ce.item()
  ep_loss += loss.item()
  n_b += 1

  scheduler_lr.step()
  scheduler_tau.step()

  if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
  model.eval()
  sync_arithmetic_mdl(model, wm)
  cl = wm.code_length()
  code_len_history.append(cl)
  p_ent = model.pairing_entropy()
  with torch.no_grad():
  tr = (model(x_train, tau).argmax(1) == y_train).float().mean().item()
  te = (model(x_test, tau).argmax(1) == y_test).float().mean().item()
  print(f"  {epoch+1:>5} {tau:>6.4f} {ep_ce/n_b:>8.4f} "
  f"{ep_loss/n_b:>8.4f} {cl:>10.1f} {p_ent:>9.4f} "
  f"{tr:>7.4f} {te:>7.4f}")

  # ── Final soft eval ────────────────────────────────────────────────────
  final_tau = scheduler_tau.temperature
  model.eval()
  with torch.no_grad():
  final_tr = (model(x_train, final_tau).argmax(1) == y_train).float().mean().item()
  final_te = (model(x_test, final_tau).argmax(1) == y_test).float().mean().item()

  soft_speed = measure_throughput(model, x_test, temperature=final_tau)
  print(f"\n  [Soft model]  TrainAcc={final_tr:.4f}  TestAcc={final_te:.4f}")
  print(f"  Soft throughput: {soft_speed:,.0f} samples/sec")

  # ── Collapse → hard circuit ────────────────────────────────────────────
  hard = model.collapse()
  hard.eval()
  with torch.no_grad():
  hard_tr = (hard(x_train).argmax(1) == y_train).float().mean().item()
  hard_te = (hard(x_test).argmax(1) == y_test).float().mean().item()
  hard_speed = measure_throughput(hard, x_test)

  print(f"  [Hard circuit] TrainAcc={hard_tr:.4f}  TestAcc={hard_te:.4f}")
  print(f"  Hard throughput: {hard_speed:,.0f} samples/sec  "
  f"({hard_speed/soft_speed:.2f}x soft)")

  # ── MDL summary ────────────────────────────────────────────────────────
  sync_arithmetic_mdl(model, wm)
  final_cl = wm.code_length()
  max_cl = total_neurons * math.log2(16)
  print(f"\n  [MDL] gates={total_neurons}  "
  f"max_code={max_cl:.1f}bits  "
  f"final_code={final_cl:.1f}bits  "
  f"compression={final_cl/max_cl:.4f}")

  if len(code_len_history) >= 2:
  drops = sum(1 for a, b in zip(code_len_history, code_len_history[1:]) if b < a)
  total_steps = len(code_len_history) - 1
  print(f"  MDL monotone: {drops}/{total_steps} "
  f"({'OK' if drops == total_steps else 'partial'})")

  # ── Pairing 確定度の確認 ───────────────────────────────────────────────
  print(f"\n  [Pairing entropy after collapse]")
  for li, layer in enumerate(model.gate_layers):
  with torch.no_grad():
  # 各ニューロンの入力A/Bの確率分布最大値（確定度の指標）
  pa_max = F.softmax(layer.input_a_logits, dim=-1).max(dim=-1).values
  pb_max = F.softmax(layer.input_b_logits, dim=-1).max(dim=-1).values
  print(f"  Layer{li}: K={layer.num_neurons}  "
  f"a_conf={pa_max.mean():.3f}  b_conf={pb_max.mean():.3f}  "
  f"(1.0=完全確定)")

  # ── Gate distribution ─────────────────────────────────────────────────
  gate_names = [
  "FALSE", "AND", "A&~B", "A", "~A&B", "B",
  "XOR", "OR", "NOR", "XNOR", "~B", "A|~B",
  "~A", "~A|B", "NAND", "TRUE"
  ]
  counts = [0] * 16
  with torch.no_grad():
  for layer in model.gate_layers:
  for g in layer.gate_logits.argmax(dim=-1).tolist():
  counts[g] += 1

  print(f"\n  Gate distribution ({total_neurons} nodes):")
  for i, (nm, c) in enumerate(zip(gate_names, counts)):
  if c > 0:
  bar = '#' * max(1, c * 40 // max(1, total_neurons))
  print(f"  {i:2d} {nm:<7} {c:4d}  {bar}")

  return {
  'soft_train': final_tr,
  'soft_test': final_te,
  'hard_train': hard_tr,
  'hard_test': hard_te,
  'soft_speed': soft_speed,
  'hard_speed': hard_speed,
  'final_code_len': final_cl,
  'max_code_len': max_cl,
  'code_len_history': code_len_history,
  }

# ────────────────────────────────────────────────────────────────────────────
# 10. main
# ────────────────────────────────────────────────────────────────────────────

def main():
  print("=" * 68)
  print("  Dynamic DLGN -- Gumbel-Softmax Dynamic Pairing")
  print("  Device: CPU  PyTorch 2.7.0")
  print("=" * 68)

  # ── Run 1: 8-bit 健全性チェック ────────────────────────────────────────
  print("\n\n[Run 1] 8-bit parity (sanity check)")
  r8 = train_64bit_parity(
  n_bits=8,
  hidden_neurons=[32, 16, 8],
  n_train=4096,
  n_test=1024,
  epochs=100,
  lr=0.02,
  lambda_mdl=1e-4,
  lambda_diversity=1e-3,
  t_start=1.0,
  t_end=0.05,
  warm_hold_frac=0.20,
  seed=42,
  init_scale=0.1,
  init_pairing='random_hard',
  )

  # ── Run 2a: 64-bit random_hard ────────────────────────────────────────
  print("\n\n[Run 2a] 64-bit parity (random_hard init, 150ep)")
  r64a = train_64bit_parity(
  n_bits=64,
  hidden_neurons=[128, 64, 32, 16, 8, 4],
  n_train=8192,
  n_test=2048,
  epochs=150,
  lr=0.02,
  lambda_mdl=1e-4,
  lambda_diversity=1e-3,
  t_start=1.0,
  t_end=0.05,
  warm_hold_frac=0.25,
  seed=42,
  init_scale=0.1,
  init_pairing='random_hard',
  )

  # ── Run 2b: 64-bit guided_tournament (XOR連鎖バイアス初期化) ─────────
  # warm_hold_frac=0.5でPhase1を長くしてゲートを十分収束させる
  print("\n\n[Run 2b] 64-bit parity (guided_tournament init, long warm)")
  r64b = train_64bit_parity(
  n_bits=64,
  hidden_neurons=[128, 64, 32, 16, 8, 4],
  n_train=8192,
  n_test=2048,
  epochs=200,
  lr=0.02,
  lambda_mdl=1e-4,
  lambda_diversity=1e-3,
  t_start=2.0,
  t_end=0.02,
  warm_hold_frac=0.5,
  seed=42,
  init_scale=0.1,
  init_pairing='guided_tournament',
  )

  # ── Summary ───────────────────────────────────────────────────────────
  print(f"\n\n{'=' * 68}")
  print("  SUMMARY")
  print(f"{'=' * 68}")
  print(f"  8-bit  soft={r8['soft_test']:.4f}  hard={r8['hard_test']:.4f}  "
  f"speed={r8['soft_speed']:,.0f}/{r8['hard_speed']:,.0f} samp/s")
  print(f"  64-bit rnd  soft={r64a['soft_test']:.4f}  hard={r64a['hard_test']:.4f}  "
  f"speed={r64a['soft_speed']:,.0f}/{r64a['hard_speed']:,.0f} samp/s")
  print(f"  64-bit guid soft={r64b['soft_test']:.4f}  hard={r64b['hard_test']:.4f}  "
  f"speed={r64b['soft_speed']:,.0f}/{r64b['hard_speed']:,.0f} samp/s")
  print(f"\n  MDL code compression:")
  print(f"  8-bit  {r8['final_code_len']:.1f}/{r8['max_code_len']:.1f} bits  "
  f"({r8['final_code_len']/r8['max_code_len']:.4f})")
  print(f"  64-bit rnd  {r64a['final_code_len']:.1f}/{r64a['max_code_len']:.1f} bits  "
  f"({r64a['final_code_len']/r64a['max_code_len']:.4f})")
  print(f"  64-bit guid {r64b['final_code_len']:.1f}/{r64b['max_code_len']:.1f} bits  "
  f"({r64b['final_code_len']/r64b['max_code_len']:.4f})")
  print()
if __name__ == "__main__":
  main()
