"""
DLGN: Differentiable Logic Gate Network
"""

import torch
import torch.nn as nn
import math

# 16種の2入力Boolean関数 (a, b ∈ [0,1] の連続緩和)
# AND近似: a*b, OR: a+b-a*b, XOR: a+b-2*a*b, etc.
def _soft_gates(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """
  Returns: (16, *shape) tensor, each slice is one gate's soft output
  a, b: (batch, neurons) tensors in [0,1]
  """
  ab = a * b
  gates = [
  torch.zeros_like(a),  # 0: FALSE
  ab,  # 1: AND
  a - ab,  # 2: A AND NOT B
  a.clone(),  # 3: A
  b - ab,  # 4: NOT A AND B
  b.clone(),  # 5: B
  a + b - 2 * ab,  # 6: XOR
  a + b - ab,  # 7: OR
  1 - (a + b - ab),  # 8: NOR
  1 - (a + b - 2 * ab),  # 9: XNOR
  1 - b,  # 10: NOT B
  1 - b + ab,  # 11: A OR NOT B
  1 - a,  # 12: NOT A
  1 - a + ab,  # 13: NOT A OR B
  1 - ab,  # 14: NAND
  torch.ones_like(a),  # 15: TRUE
  ]
  return torch.stack(gates, dim=0)  # (16, batch, neurons)

class DifferentiableGateLayer(nn.Module):
  def __init__(
  self,
  in_features: int,
  num_neurons: int,
  seed: int = 0,
  pairing: str = 'random'
  ):
  """
  pairing:
  'random'  - 全結合からランダムペア
  'stride'  - stride=in_features//2 のシフトペア（XOR連鎖向き）
  'adjacent'- 隣接ペアを周期的に割り当て
  """
  super().__init__()
  self.in_features = in_features
  self.num_neurons = num_neurons

  # 各ニューロンが16ゲートの確率分布を持つ
  # logit → softmax → p ∈ Δ^15
  self.gate_logits = nn.Parameter(torch.randn(num_neurons, 16) * 0.1)

  # 入力ペアを層ごとに固定（学習中は変えない）
  rng = torch.Generator()
  rng.manual_seed(seed)

  if pairing == 'stride':
  # stride = in_features//2 のシフトペア（XOR二分木に有利）
  stride = max(1, in_features // 2)
  base = torch.arange(num_neurons) % in_features
  idx_a = base
  idx_b = (base + stride) % in_features
  elif pairing == 'adjacent':
  base = torch.arange(num_neurons) % in_features
  idx_a = base
  idx_b = (base + 1) % in_features
  elif pairing == 'tournament':
  # 二分木トーナメント形式: ニューロンiは入力(2i, 2i+1)を担う
  # num_neurons <= in_features//2 を想定
  # num_neurons > in_features//2 の場合はmod折り返し
  pairs_per_level = max(1, in_features // 2)
  idx_a_list, idx_b_list = , 
  for n_idx in range(num_neurons):
  pair = n_idx % pairs_per_level
  idx_a_list.append((pair * 2) % in_features)
  idx_b_list.append((pair * 2 + 1) % in_features)
  idx_a = torch.tensor(idx_a_list, dtype=torch.long)
  idx_b = torch.tensor(idx_b_list, dtype=torch.long)
  else:  # random
  idx_a = torch.randint(0, in_features, (num_neurons,), generator=rng)
  idx_b = torch.randint(0, in_features, (num_neurons,), generator=rng)

  # 自己ループを避ける
  same = idx_a == idx_b
  idx_b[same] = (idx_b[same] + 1) % in_features

  self.register_buffer('idx_a', idx_a)
  self.register_buffer('idx_b', idx_b)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
  """
  x: (batch, in_features) in [0,1]
  returns: (batch, num_neurons) in [0,1]
  """
  a = x[:, self.idx_a]  # (batch, neurons)
  b = x[:, self.idx_b]  # (batch, neurons)

  p = torch.softmax(self.gate_logits, dim=-1)  # (neurons, 16)

  gate_outputs = _soft_gates(a, b)  # (16, batch, neurons)
  gate_outputs = gate_outputs.permute(1, 2, 0)  # (batch, neurons, 16)

  # y_k = Σ_g p_g * f_g(a, b)
  y = (gate_outputs * p.unsqueeze(0)).sum(dim=-1)  # (batch, neurons)
  return y

  def collapse(self) -> 'HardGateLayer':
  """argmax でハードBoolean回路に変換"""
  hard_gates = self.gate_logits.argmax(dim=-1)  # (neurons,)
  return HardGateLayer(
  self.in_features,
  self.num_neurons,
  self.idx_a.clone(),
  self.idx_b.clone(),
  hard_gates
  )

class HardGateLayer(nn.Module):
  """collapse後のハードBoolean回路層（推論専用）"""
  def __init__(self, in_features, num_neurons, idx_a, idx_b, gate_ids):
  super().__init__()
  self.register_buffer('idx_a', idx_a)
  self.register_buffer('idx_b', idx_b)
  self.register_buffer('gate_ids', gate_ids)
  self.in_features = in_features
  self.num_neurons = num_neurons

  @torch.no_grad()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
  a = x[:, self.idx_a]
  b = x[:, self.idx_b]
  gate_outputs = _soft_gates(a, b)  # (16, batch, neurons)
  gate_outputs = gate_outputs.permute(1, 2, 0)  # (batch, neurons, 16)
  # 各ニューロンの選択ゲートだけ取る
  idx = self.gate_ids.unsqueeze(0).unsqueeze(-1).expand(
  gate_outputs.shape[0], -1, 1
  )
  y = gate_outputs.gather(2, idx).squeeze(-1)
  return (y > 0.5).float()

class DLGNModel(nn.Module):
  def __init__(
  self,
  in_features: int,
  hidden_neurons: list[int],
  num_classes: int = 2,
  seed_base: int = 42,
  pairing: str = 'random'
  ):
  super().__init__()
  self.gate_layers = nn.ModuleList()
  prev = in_features
  for i, n in enumerate(hidden_neurons):
  self.gate_layers.append(
  DifferentiableGateLayer(prev, n, seed=seed_base + i, pairing=pairing)
  )
  prev = n

  # 最終層: 線形（クラス数へ）
  self.head = nn.Linear(prev, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
  """x: (batch, in_features) float32 in {0,1} or [0,1]"""
  for layer in self.gate_layers:
  x = layer(x)
  return self.head(x)

  def collapse(self) -> 'HardDLGNModel':
  hard_layers = [layer.collapse() for layer in self.gate_layers]
  return HardDLGNModel(hard_layers, self.head)

def make_xor_tree_model(n_bits: int, num_classes: int = 2,
  seed_base: int = 42) -> 'DLGNModel':
  """
  厳密二分木XOR連鎖モデル。
  各層のニューロン数 = 入力数 // 2、tournament pairingで前層の隣接ペアを担当。

  n_bits=8:  layers=[4, 2, 1]  (3段)
  n_bits=64: layers=[32,16,8,4,2,1] (6段)

  この構造はlog2(n_bits)段のXOR連鎖を学習できる最小の構造。
  各ニューロンは「前層の連続する2出力をXOR」するよう誘導される。
  """
  import math
  n_layers = math.ceil(math.log2(n_bits))
  hidden = 
  cur = n_bits
  for _ in range(n_layers):
  cur = max(1, cur // 2)
  hidden.append(cur)

  model = DLGNModel(
  in_features=n_bits,
  hidden_neurons=hidden,
  num_classes=num_classes,
  seed_base=seed_base,
  pairing='tournament',
  )
  return model

class HardDLGNModel(nn.Module):
  """collapse後のフルハード推論モデル"""
  def __init__(self, hard_layers: list[HardGateLayer], head: nn.Linear):
  super().__init__()
  self.hard_layers = nn.ModuleList(hard_layers)
  self.head = head

  @torch.no_grad()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
  for layer in self.hard_layers:
  x = layer(x)
  return self.head(x)
