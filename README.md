# Boolean Attention — 論理ゲートだけで Transformer を超える研究

> GPU を使わず、AND・OR・XOR などの論理ゲートだけで言語モデルを作ったら、
> Perplexity で Transformer を 0.13 上回った（4.73 vs 4.86）。

個人で進めている研究記録です。論文を読むだけでは満たされず、
**自分で実装してベンチを取って、失敗まで含めて公開する** ことを方針にしています。

YouTube に解説動画も投稿しています → [かるまるラボ](https://www.youtube.com/@かるまる-LAB)

## 結論

| モデル | Soft PPL | Hard PPL | 備考 |
|---|---:|---:|---|
| Transformer (baseline) | — | **4.86** | 比較対象 |
| DLGN flat | 11.83 | 15.16 | 純粋な論理ゲートで言語学習に成功 |
| LoopedDLGN v1 | 11.05 | **754.31** | 量子化誤差が反復で爆発し撃沈 |
| HBA v1 | 5.40 → 9.75 | — | TF まで 0.54 差まで肉薄、後半過学習 |
| HBA v2 | **5.32** | **6.54** | Best ckpt / Early stop / 閾値校正で安定化 |
| **HBA Distilled (Student)** | **4.73** | 6.31 | **Transformer を 0.13 上回る** |
| ChatHBA | 6.46 | — | 数値は出るが会話崩壊（容量限界） |

データセット: TinyShakespeare 80KB / char-level / vocab=61

## なぜこんなことをしたか

現在の AI は GPU の大量並列計算に強く依存しています。
もし AND・OR・XOR といった **論理ゲートだけで言語モデルが作れたら**、
CPU で爆速、消費電力も激減し、組み込み機器でも巨大言語モデルが動かせます。

「ニューラルネットの中身は、本当に float の積和が必要なのか？」
この問いに、論理回路の側から答えを出すことを目指しました。

## 主要アイデア

### 1. DLGN (Differentiable Logic Gate Networks)
本来、論理ゲートは離散関数で勾配が流れません。
そこで **16 種類のゲートを softmax で混ぜて、勾配を流せる** ように拡張します。
学習が終わったら確率の最大値で 1 個に確定し、純粋なブーリアン回路に戻します。

### 2. LoopedDLGN（失敗）
同じブーリアンブロックを入力依存で何度も繰り返し、固定点に収束させる設計。
バナッハの不動点定理に立脚し、Universal Transformer の発想を Boolean で再現しようとしました。
**結果は撃沈**。反復ごとの量子化誤差が爆発し、Hard PPL が 754 まで暴騰しました。

### 3. HBA (Hierarchical Boolean Attention)
Transformer の Attention を **ルーター（どこを見るか）** と **値の集約** に分けて考えます。
- ルーターのみ Boolean で決定
- 値の集約は float のまま

これでハードコラプス時の誤差累積問題が **構造的に** 解消されました。

### 4. 知識蒸留
教師役の Transformer を学習させ、その知識を生徒の HBA に蒸留。
結果、Student PPL 4.73 を達成し、教師 4.86 を 0.13 上回りました。
論理回路の言語モデルが、Transformer を超えた瞬間です。

## ディレクトリ構成

```
boolean-attention/
├── src/  # 実装本体
│  ├── dlgn.py  # DLGN 基本層
│  ├── dlgn_charlm.py  # DLGN flat (PPL 11.83)
│  ├── looped_dlgn_charlm.py  # LoopedDLGN (撃沈)
│  ├── hba_charlm.py  # HBA v1/v2 (PPL 5.32)
│  ├── hba_distill_charlm.py  # 蒸留 (PPL 4.73, 本研究の主役)
│  └── chat/  # チャット試行 (会話は崩壊)
│  ├── chat_hba.py
│  └── chat_repl.py
├── data/
│  └── tinyshakespeare.txt
├── qa-corpus/  # ChatHBA 用に自作した 5,377 件 QA
├── results/  # 学習ログ
├── checkpoints/  # 学習済みモデル (ChatHBA)
├── docs/  # GitHub Pages 用詳細ドキュメント
└── articles/  # Qiita / Zenn 用記事ドラフト
```

## 動かし方

```bash
pip install -r requirements.txt

# DLGN flat (約 5 分、PPL 11.83 が出ます)
python src/dlgn_charlm.py

# HBA v2 (約 5 分、PPL 5.32 が出ます)
python src/hba_charlm.py

# 蒸留で TF 越え (約 10 分、PPL 4.73)
python src/hba_distill_charlm.py
```

## さらに詳しく

- 理論と式: [docs/theory.md](docs/theory.md)
- 全実験ログと考察: [docs/experiments.md](docs/experiments.md)
- ChatHBA の崩壊と容量限界: [docs/chat.md](docs/chat.md)

GitHub Pages 版 → `https://karumaru-kakikukekodoumei.github.io/boolean-attention/`

## 動画

YouTube に約 5 分の研究解説動画があります。
ナレーション: ずんだもん (VOICEVOX) / 編集: FrameScript（自作の動画エディタ）

→ [動画リンクは公開後ここに追記]

## ライセンス

MIT
