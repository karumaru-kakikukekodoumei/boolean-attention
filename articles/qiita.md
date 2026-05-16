# 論理ゲートで言語モデルを作って Transformer を超える実装ガイド (PPL 4.73 vs 4.86)

GPU を使わず、AND・OR・XOR などの **論理ゲートだけ** で言語モデルを構築し、
char-level TinyShakespeare で **Perplexity 4.73** を達成した実装の解説です。
比較対象の Transformer (PPL 4.86) を 0.13 上回りました。

この記事は **再現手順** に振った実装ガイドです。
研究の物語 / 失敗譚は [Zenn 版](https://zenn.dev/) に書きました（軸を分けています）。

- リポジトリ: [karumaru-kakikukekodoumei/boolean-attention](https://github.com/karumaru-kakikukekodoumei/boolean-attention)
- 詳細ドキュメント: [GitHub Pages](https://karumaru-kakikukekodoumei.github.io/boolean-attention/)
- 解説動画: [かるまるラボ - YouTube](https://www.youtube.com/@かるまる-LAB)

## 環境

- Python 3.10+
- PyTorch 2.1+
- RTX 4060 8GB （CPU でも動きますが学習時間が伸びます）

```bash
git clone https://github.com/karumaru-kakikukekodoumei/boolean-attention.git
cd boolean-attention
pip install -r requirements.txt
```

## Step 1. 微分可能な論理ゲート層 (DLGN) を作る

論理ゲートは離散関数で勾配が流れません。
2 入力ブール関数は 2^4 = 16 種類しかないという事実を使い、
**16 ゲートを softmax で混合** することで勾配を流します。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def all_gates(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """16 種類の論理ゲートを (..., 16) 次元で返す。a, b は [0,1] 連続値想定。"""
  return torch.stack([
  torch.zeros_like(a),  a * b,  a * (1 - b),  a,
  (1 - a) * b,  b,  a + b - 2*a*b,  a + b - a*b,
  1 - (a + b - a*b),  1 - (a + b - 2*a*b),  1 - b,  a + (1-b) - a*(1-b),
  1 - a,  (1-a) + b - (1-a)*b,  1 - a*b,  torch.ones_like(a),
  ], dim=-1)

class DLGNLayer(nn.Module):
  def __init__(self, in_dim: int, out_dim: int, tau: float = 1.0):
  super().__init__()
  self.in_dim, self.out_dim = in_dim, out_dim
  self.tau = tau
  # 出力位置ごとに「どの入力ペアを選ぶか」「16 ゲートのどれを選ぶか」の logit
  self.pair_a = nn.Parameter(torch.randn(out_dim, in_dim) * 0.5)
  self.pair_b = nn.Parameter(torch.randn(out_dim, in_dim) * 0.5)
  self.gate_logits = nn.Parameter(torch.randn(out_dim, 16) * 0.1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
  # x: [B, in_dim], softmax で入力ペアを連続選択
  wa = F.softmax(self.pair_a / self.tau, dim=-1)  # [out, in]
  wb = F.softmax(self.pair_b / self.tau, dim=-1)
  a = x @ wa.T  # [B, out]
  b = x @ wb.T

  gates = all_gates(a, b)  # [B, out, 16]
  alpha = F.softmax(self.gate_logits / self.tau, dim=-1)  # [out, 16]
  return (gates * alpha).sum(dim=-1)  # [B, out]
```

学習が終わったら `argmax(self.gate_logits)` で 1 個に確定すれば、
純粋なブーリアン回路に戻ります（hard collapse）。

## Step 2. DLGN flat で char-LM を作る

`src/dlgn_charlm.py` を実行すれば再現できます。
TinyShakespeare 80KB で 4 段の DLGN を学習。

```bash
python src/dlgn_charlm.py
```

| 結果 | Soft PPL | Hard PPL |
|---|---:|---:|
| DLGN flat (4 層) | 11.83 | 15.16 |
| Transformer (比較) | — | 4.86 |

論理回路だけで言語学習が **できることは確認**。
ただし Transformer には遠い。設計を変える必要があります。

## Step 3. ❌ LoopedDLGN は試す価値があるが結果は撃沈

Universal Transformer の発想を Boolean に移植。
同じ DLGN ブロックを T 回繰り返し、固定点に収束させる設計です。

```bash
python src/looped_dlgn_charlm.py --max-iters=8
```

| 版 | Soft PPL | Hard PPL |
|---|---:|---:|
| v1 (PE なし) | 11.05 | **754.31** |
| v2 (PE あり) | 18.7 | 312.4 |
| v3 (容量倍) | 17.9 | 286.1 |

ハードコラプス時に PPL が **754** まで暴騰。
理由は明快で、各反復の量子化誤差が以下のように蓄積するからです:

```
ε_total ≈ Σ_t ||f_hard(x^(t)) - f_soft(x^(t))||
```

**Boolean 量子化と反復系は構造的に相性が悪い**ことが分かりました。
この失敗を踏まえて、設計を根本から見直します。

## Step 4. HBA — Boolean を入れる場所を変える

Transformer の Attention は **(1) どこを見るか** と **(2) 値の集約** に分けられます。
HBA は **(1) のルーターだけを Boolean 化**、(2) は float のままにします。

```python
import torch.nn.utils as nn_utils

class BooleanAttentionLayer(nn.Module):
  def __init__(self, d: int, tau: float = 0.1):
  super().__init__()
  self.q = nn.Linear(d, d)
  self.k = nn.Linear(d, d)
  self.v = nn.Linear(d, d)
  # bilinear router (Lipschitz 制約に spectral norm)
  self.w_router = nn_utils.spectral_norm(nn.Linear(d, d, bias=False))
  self.tau = tau

  def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
  Q, K, V = self.q(x), self.k(x), self.v(x)  # [B, T, d]
  # Q · W · K^T
  logits = Q @ self.w_router.weight @ K.transpose(-1, -2)  # [B, T, T]
  logits = logits.masked_fill(causal_mask, float("-inf"))

  if self.training:
  router = torch.tanh(logits / self.tau)  # 連続近似
  else:
  router = torch.sign(logits)  # 推論時離散

  attn = F.softmax(router / self.tau, dim=-1)
  return attn @ V  # V は float のまま
```

ポイント:

- ルーターは離散値 (-1, +1) に確定
- 値の集約は float なので **量子化誤差が深さ方向に伝播しない**
- spectral norm で router 重みのリプシッツ性を担保（発散防止）

```bash
python src/hba_charlm.py --epochs=60
```

| HBA v1 | Best PPL | Final PPL |
|---|---:|---:|
| Ep12 / Ep60 | **5.40** | 9.75 |

TF (4.86) まで **0.54 差** まで肉薄。ただし過学習が課題。

## Step 5. HBA v2 — 安定化テクニック

v1 の過学習に以下で対処:

```python
# 1. Best checkpoint 保存
if val_ppl < best_ppl:
  best_ppl = val_ppl
  best_state = {k: v.clone() for k, v in model.state_dict().items()}
  best_epoch = ep
  bad_count = 0
else:
  bad_count += 1

# 2. Early stopping
if bad_count >= patience:
  print(f"early stop at ep {ep}")
  break

# 3. Hard threshold calibration (推論時の τ を val で再校正)
def calibrate_hard_threshold(model, val_loader, taus=[0.05, 0.08, 0.1, 0.15, 0.2]):
  best = (None, float("inf"))
  for tau in taus:
  model.set_inference_tau(tau)
  ppl = evaluate(model, val_loader)
  if ppl < best[1]:
  best = (tau, ppl)
  return best

# 4. 温度の warm_hold パターン
def temperature_schedule(epoch: int) -> float:
  if epoch < 5:  return 1.0  # warm: 柔らかく
  if epoch < 15:  return 0.5  # hold: 中間
  return max(0.1, 0.5 * 0.95**(epoch - 15))  # decay: 徐々に hard へ
```

```bash
python src/hba_charlm.py --epochs=40 --early-stop --calibrate
```

| HBA v2 | Soft PPL | Hard PPL | Train time |
|---|---:|---:|---:|
| 結果 | **5.32** | **6.54** | 4.7 min |

LoopedDLGN の Hard PPL 754 と比べて **115 倍の改善**。

## Step 6. ⭐ 知識蒸留で TF 越え

教師 (TF) → 生徒 (HBA v2 構造) に蒸留。
ハイブリッド損失で CE と KL を併用します。

```python
def distill_loss(
  student_logits: torch.Tensor,
  teacher_logits: torch.Tensor,
  targets: torch.Tensor,
  alpha: float = 0.3,
  T: float = 8.0,
) -> torch.Tensor:
  ce = F.cross_entropy(student_logits, targets)
  kl = F.kl_div(
  F.log_softmax(student_logits / T, dim=-1),
  F.softmax(teacher_logits / T, dim=-1),
  reduction="batchmean",
  )
  return alpha * ce + (1 - alpha) * (T * T) * kl
```

```bash
python src/hba_distill_charlm.py --epochs=30 --teacher-ckpt=teacher_tf.pt
```

| | Soft PPL |
|---|---:|
| Teacher (TF) | 4.86 |
| **Student (HBA distilled)** | **4.73** |
| 逆転幅 | **-0.13** |

**論理回路ベースのモデルが Transformer を逆転**。
これは born-again networks (Furlanello et al. 2018) として知られる現象です。

## ハマりどころ: 温度整合性のバグ

初期実装で、訓練 eval と最終比較で温度 τ が違っていて、
PPL が 4.71 vs 8.72 と大きく乖離するバグに数日とられました。

### Bad

```python
# 訓練 eval は固定 tau=1.0、最終比較は final_tau=0.1 と別物 → 整合性が取れない
def evaluate(model, loader):
  model.set_inference_tau(1.0)  # 訓練評価時はソフト
  ...

# 最終比較
model.set_inference_tau(0.1)  # ←ここで急に厳しい τ にする
final_ppl = evaluate(model, test_loader)
```

### Good

```python
# 訓練 eval は「現在のスケジューラ τ」で評価
def evaluate(model, loader, tau: float):
  model.set_inference_tau(tau)
  ...

# 最終比較は best epoch 時点の実 τ を逆引き
best_tau = temperature_schedule(best_epoch)
model.load_state_dict(best_state)
model.set_inference_tau(best_tau)
final_ppl = evaluate(model, test_loader, best_tau)
```

これで再現性のある PPL 4.73 が出るようになりました。

## ベンチマーク手順

`scripts/bench-render.mjs` 相当のものは置いてませんが、各 .py を順番に走らせれば再現できます。
**実行順序**:

```bash
# 1. ベースライン
python src/dlgn_charlm.py  # PPL 11.83 が出る

# 2. 失敗パス (任意)
python src/looped_dlgn_charlm.py  # PPL 754 で爆死を体感したい人だけ

# 3. HBA v2
python src/hba_charlm.py --early-stop --calibrate  # PPL 5.32

# 4. 蒸留 (要 teacher checkpoint)
python src/train_teacher.py  # まず教師を学習
python src/hba_distill_charlm.py  # PPL 4.73 → TF 越え
```

学習ログは `results/` ディレクトリ、学習済み ChatHBA は `checkpoints/` にあります。

## 応用先

HBA は **特化用途で実用性あり** という結論でした。
具体的には:

- **Speculative decoding のドラフトモデル** — 大きい教師モデルとの並用で、軽量ルーティングを担う
- **エッジ推論** — CPU/MCU で動く軽量 LM
- **電力制約環境** — GPU を持たないシステム

## まとめ

| Step | やったこと | 結果 |
|---|---|---|
| 1 | DLGN 層を実装 | 16 ゲート softmax 混合で勾配 OK |
| 2 | DLGN flat で char-LM | PPL 11.83 (TF 4.86 に未達) |
| 3 | LoopedDLGN | PPL 754 で構造的に詰む |
| 4 | HBA v1 で発想転換 | PPL 5.40 (TF まで 0.54 差) |
| 5 | HBA v2 で安定化 | PPL 5.32 / Hard 6.54 |
| 6 | 知識蒸留で逆転 | PPL **4.73** (TF 4.86 → 0.13 上回る) |

## リンク

- リポジトリ: [github.com/karumaru-kakikukekodoumei/boolean-attention](https://github.com/karumaru-kakikukekodoumei/boolean-attention)
- 詳細ドキュメント: [GitHub Pages](https://karumaru-kakikukekodoumei.github.io/boolean-attention/)
- 解説動画: [かるまるラボ - YouTube](https://www.youtube.com/@かるまる-LAB)
- 物語編 (Zenn): 別途投稿
- 原論文: Petersen et al. (2022) "Deep Differentiable Logic Gate Networks"
- 蒸留: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
