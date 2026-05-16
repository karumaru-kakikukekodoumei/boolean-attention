"""
算術符号区間マネージャー + MDL損失
"""

import math
import torch
import torch.nn as nn
from dlgn import DifferentiableGateLayer, DLGNModel

RANGE = 2 ** 32  # 区間の全域 [0, 2^32)

class ArithmeticWeightManager:
  """
  各ゲートノードの確率分布を整数区間 [L, H) ⊂ [0, 2^32) で管理する。

  算術符号の原理:
  初期区間 [0, 2^32)
  シンボル g を選択したとき (累積分布 C[g]〜C[g+1]):
  new_L = L + (H - L) * C[g]  // integer
  new_H = L + (H - L) * C[g+1]  // integer
  コード長 = Σ log2(2^32 / (H - L))

  ここでは各ゲートを独立シンボルとして扱う（連鎖ではなく独立区間の和）。
  """

  def __init__(self, num_gates: int, num_gate_types: int = 16):
  self.num_gates = num_gates
  self.num_gate_types = num_gate_types
  # 各ゲートの区間 [L, H)、初期は全域
  self.L = [0] * num_gates
  self.H = [RANGE] * num_gates

  def update(self, gate_idx: int, active_gate: int, probs: list[float]) -> None:
  """
  gate_idx: 更新対象のゲートノードindex
  active_gate: 選択されたゲートのindex (0〜15)
  probs: 16種類のゲート確率（合計1.0）

  算術符号的に区間を狭める
  """
  L, H = self.L[gate_idx], self.H[gate_idx]
  width = H - L
  if width <= 0:
  return

  # 累積分布を整数区間に変換
  cum = 0.0
  for g, p in enumerate(probs):
  cum_next = cum + p
  if g == active_gate:
  new_L = L + int(width * cum)
  new_H = L + int(width * cum_next)
  # 最低1の幅を保証（数値誤差対策）
  if new_H <= new_L:
  new_H = new_L + 1
  self.L[gate_idx] = new_L
  self.H[gate_idx] = min(new_H, RANGE)
  return
  cum = cum_next

  def code_length(self) -> float:
  """
  全ゲートのコード長 Σ log2(2^32 / (H - L)) を返す（ビット数）
  区間が狭いほど = 確信度が高いほど = コード長が短い
  """
  total = 0.0
  for L, H in zip(self.L, self.H):
  width = H - L
  if width <= 0:
  width = 1
  total += math.log2(RANGE / width)
  return total

  def reset(self) -> None:
  self.L = [0] * self.num_gates
  self.H = [RANGE] * self.num_gates

  def sync_from_model(self, model: DLGNModel) -> None:
  """
  モデルの現在のgatelogitsからsoftmax確率を取り出し、
  argmaxゲートで全ノードの区間を一括更新する。
  学習中に定期的に呼び出して区間を最新状態に保つ。
  """
  self.reset()
  gate_idx = 0
  for layer in model.gate_layers:
  with torch.no_grad():
  probs = torch.softmax(layer.gate_logits, dim=-1)  # (neurons, 16)
  hard = probs.argmax(dim=-1)  # (neurons,)
  for n in range(layer.num_neurons):
  p_list = probs[n].tolist()
  g = hard[n].item()
  self.update(gate_idx, g, p_list)
  gate_idx += 1

def _code_length_differentiable(model: DLGNModel) -> torch.Tensor:
  """
  勾配が流れる形でのコード長近似。
  各ゲートの確率分布のエントロピーの負 = 最小期待コード長。
  H(p) = -Σ p_g log2(p_g)
  コード長の上界: Σ_neurons H(p_n) ビット
  (これを最小化 = 確信度を最大化 = MDL最小化)
  """
  total = torch.tensor(0.0)
  for layer in model.gate_layers:
  p = torch.softmax(layer.gate_logits, dim=-1)  # (neurons, 16)
  # エントロピー = -Σ p log2(p)、nats→bits変換
  entropy = -(p * torch.log(p + 1e-10)).sum(dim=-1) / math.log(2)
  total = total + entropy.sum()
  return total

def mdl_loss(
  model: DLGNModel,
  ce_loss: torch.Tensor,
  lambda_mdl: float = 1e-3
) -> torch.Tensor:
  """
  MDL正則化損失: CrossEntropy + lambda * code_length

  code_length は微分可能なエントロピー近似を使用。
  lambda_mdlでデータ項とモデル複雑度のトレードオフを制御。
  """
  cl = _code_length_differentiable(model)
  return ce_loss + lambda_mdl * cl
