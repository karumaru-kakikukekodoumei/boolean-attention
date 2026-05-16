---
title: "論理ゲートだけで言語モデルを作って Transformer を超えるまで、3 度散った話"
emoji: "🧨"
type: "idea"
topics: ["機械学習", "transformer", "失敗談", "個人開発", "ポエム"]
published: false
---

## こんなことをしました

GPU を使わず、AND・OR・XOR などの論理ゲートだけで言語モデルを作って、
Transformer (Perplexity 4.86) を 0.13 上回る **PPL 4.73** を達成しました。

ただし、ここに辿り着くまでに **3 回散ってます**。

この記事は、ありがちな「やってみた」ではなく、
**どう失敗して、何を諦めて、どこで思考を変えたか** を順番に書く物語編です。
コード重視の実装ガイドは [Qiita 版](https://qiita.com/) に書きました（軸を分けています）。

- リポジトリ: [karumaru-kakikukekodoumei/boolean-attention](https://github.com/karumaru-kakikukekodoumei/boolean-attention)
- 詳細ドキュメント: [GitHub Pages](https://karumaru-kakikukekodoumei.github.io/boolean-attention/)
- 解説動画: [かるまるラボ - YouTube](https://www.youtube.com/@かるまる-LAB)

## 第 0 章: そもそも何で論理ゲートで AI 作りたいのか

今の AI は、結局 GPU の大量並列 float 積和で動いています。
学習はもちろん、推論もまともな速度を出すには GPU 必須。

でも、ニューラルネットの中身を覗くと、やってるのは大体「重みを掛けて、足して、潰す」。
**float が本当に必要なんでしょうか？**

もし AND・OR・XOR だけで言語モデルが作れたら:

- CPU でも爆速で動く
- 消費電力が激減する
- マイコンや組み込み機器でも巨大 LM が動く
- 何より、AI のブラックボックス感が薄れる（ゲートを見れば回路が読める）

「重み」「埋め込み」「アテンション」って、結局のところ何の情報処理をしてるんでしょうか？
**それが論理回路で書けるなら、書きたい。** 動機はそれです。

## 第 1 章: 一度目の挑戦 — DLGN flat で素直にやる

論理ゲートを微分可能にする手法として、Petersen et al. (2022) の **DLGN** があります。
やっていることはシンプルで:

- 2 入力のブール関数は 2^4 = 16 種類しかない
- 各位置で、16 ゲートを softmax で混ぜて勾配を流す
- 学習後は argmax で 1 個に確定して、純粋なブーリアン回路に戻す

これを 4 段積んで、TinyShakespeare 80KB を char-LM として学習させました。

結果:

| | PPL |
|---|---:|
| DLGN flat | **11.83** |
| Transformer (比較) | 4.86 |

**学習はできた**。論理回路で言語が学べるという、当時の自分にとっては結構な事実が確認できました。
ただし TF には遠い。「あー、これは何か工夫しないとダメだな」と思って次に進みます。

このときの気分は「順調な研究の出だし」。**まだ希望があった頃です。**

## 第 2 章: 二度目の挑戦 — LoopedDLGN で散る

DLGN flat の限界は表現力。じゃあ何度も繰り返せばいい。
**Universal Transformer の発想を Boolean でやろう** と考えました。

具体的には:

- 同じ DLGN ブロックを T 回繰り返し
- 入力依存で何度もループ
- 固定点 x* に収束させる
- バナッハの contraction mapping theorem で理論武装

数学的にも美しい設計です。実装も、書いてて気持ちよかった。
「これは絶対いける」と思いました。

そして:

| 版 | Hard PPL |
|---|---:|
| v1 | **754.31** |
| v2 | 312.4 |
| v3 | 286.1 |

**756 倍に膨らみました。** TF 4.86 比。

数字を見て笑いました。「いや、これはバグじゃないか」と。
でも何回チェックしても再現する。容量を倍にしても、Positional Encoding 入れても、
結果は同じ方向。

**原因は明白で、構造的でした。** 反復ごとに量子化誤差が:

$$\epsilon_{\text{total}} \gtrsim \sum_t \|f_{\text{hard}}(x^{(t)}) - f_{\text{soft}}(x^{(t)})\|$$

として爆発的に積み重なります。深さに対して線形以上で増える。
**反復系と Boolean 量子化は、相性が最悪。**

この瞬間が一番つらかった。1 週間分の作業がほぼ無に近い結果になって、
「もう Boolean は無理なんじゃないか」とちょっと諦めかけました。

## 第 3 章: 三度目の挑戦 — Boolean を入れる場所を変える

諦めかけて、コーヒー飲みながら、ふと思いました。

**「Boolean は深さ方向に入れちゃダメだ。じゃあどこに入れる？」**

Transformer の Attention をよく見ると、抽象的に 2 ステップに分けられます:

1. **ルーティング**: クエリ Q とキー K から「どこを見るか」を決める
2. **値の集約**: 注意重みでバリュー V を加重平均

このうち、**(1) だけを Boolean にしたら？** (2) は float のまま残せばいい。

ルーターは離散値 (-1, +1) でも、値の集約は float なので、
**量子化誤差が深さ方向に伝播しない**。

これが HBA (Hierarchical Boolean Attention) の発想です。

| HBA v1 | Best PPL |
|---|---:|
| Ep12 | **5.40** |

来た。**TF 4.86 まで 0.54 差。** LoopedDLGN の 754 と比べたら別世界です。

ただし後半で過学習。Best ckpt + Early stop + Hard 閾値校正 + warm_hold 温度の 4 点セットを入れた v2 で:

| HBA v2 | Soft PPL | Hard PPL |
|---|---:|---:|
| 結果 | **5.32** | **6.54** |

Hard PPL 6.54。LoopedDLGN との **115 倍の改善**。
数字が出た瞬間、声出ました。「あ、勝てるかもしれない」と。

## 第 4 章: 蒸留で TF を逆転

TF まで残り 0.46 差。これを埋めるには「中身を変える」より「学習方法を変える」方が早い。

Hinton 流の知識蒸留を試しました。教師は通常の Transformer、生徒は HBA v2 構造。

```
L = α · CE(p_student, y_true) + (1-α) · T² · KL(p_teacher_T || p_student_T)
```

`α=0.3`, `T=8`。10 分くらいで学習終わって、PPL を測定。

| | Soft PPL |
|---|---:|
| Teacher (TF) | 4.86 |
| **Student (HBA distilled)** | **4.73** |
| 逆転幅 | **-0.13** |

**論理回路ベースの生徒が、Transformer の教師を 0.13 上回りました。**

これは「born-again networks」(Furlanello et al. 2018) として知られる現象で、
教師のソフトラベルが実質的にデータオーグメンテーションとして働くため、
生徒が教師を逆転することがあります。

理屈は知っていたけど、**自分の手で実際に起きるのを見るのは別の体験** でした。

## 第 5 章: 油断して挑んだ ChatHBA、跡形もなく崩壊

研究で TF 越えできたので、調子に乗りました。

「これで会話モデル作れるんじゃね？」

英語 Q&A を 5,377 件、自分の手で作って、HBA を fine-tune。
50 epoch 回して、Perplexity も 6.46 と悪くない値。

REPL を起動して、おそるおそる入力:

```
Q: What is the capital of France?
A: It is Otewkia.

Q: How are you today?
A: ::: h.

Q: Tell me a joke.
A: The is the is the is the is the...
```

**Otewkia って何の国だよ。**

完全に崩壊しました。**数値ベンチマークの数字と、実用の品質は別物** という、
研究者なら誰でも知っている当たり前を、自分の手で再確認した瞬間です。

原因は 3 つほど考えられて:

1. char-LM は context 64 文字で長距離関係を学べない（`France→Paris` のような連鎖が難しい）
2. 5,377 件は HBA の容量に対してデータ規模が中途半端で、汎化ではなく丸暗記
3. 蒸留なしの fine-tune は HBA 単体の表現力に依存しすぎる

研究編で TF 越えできたのは蒸留があったから。fine-tune は別の戦いだと、ちゃんと書いてある参考文献を、改めて読み直しました。

## 第 6 章: それで、何が分かったのか

整理すると:

| 主張 | 自分で確認したこと |
|---|---|
| Boolean 回路で言語モデルが作れる | DLGN flat で PPL 11.83 |
| 反復 Boolean は誤差が爆発する | LoopedDLGN で PPL 754 を体験 |
| Router だけ Boolean なら誤差は伝播しない | HBA v2 で Hard PPL 6.54 |
| 蒸留で Boolean モデルが TF を超える | Student PPL 4.73 (TF 4.86) |
| char-LM 単体で会話は無理 | ChatHBA で Otewkia 出力 |

これは個人研究の良いところで、**全部失敗まで含めて公開できる** から、
誰かが同じ罠を踏まずに済むかもしれません。

応用としては、HBA は **Speculative Decoding のドラフトモデル** など、
軽量・高速ルーティングが必要な特化用途で実用の見込みがあります。
そのへんは次の研究で。

## 第 7 章: なぜ個人研究を公開するか

これは余談ですが、書いておきたいことなので。

論文を読んでいて感じるのは、「成功例ばかりが整然と書かれている」こと。
それは正しいんですが、**実装の途中で何度散ったのかは、論文には書かれていない**。

個人研究の良さは、Twitter とかブログで失敗まで書ける自由度だと思っています。
PPL 754 で爆死した話、ChatHBA で Otewkia を生んだ話、こういうのが
誰かの研究の役に立つこともあるかも、と思って書いています。

GitHub に全データ・コード・実験ログを置いておきました。気になる方は見てください。

## リンク

- [GitHub: boolean-attention](https://github.com/karumaru-kakikukekodoumei/boolean-attention)
- [GitHub Pages: 詳細ドキュメント](https://karumaru-kakikukekodoumei.github.io/boolean-attention/)
- [かるまるラボ YouTube](https://www.youtube.com/@かるまる-LAB)
- 実装ガイド (Qiita): 別途投稿
- 原論文: Petersen et al. (2022) "Deep Differentiable Logic Gate Networks"
- Born-again: Furlanello et al. (2018) "Born Again Neural Networks"
