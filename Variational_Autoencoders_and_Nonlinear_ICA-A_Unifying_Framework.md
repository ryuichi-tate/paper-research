# Variational Autoencoders and Nonlinear ICA: A Unifying Framework

## 概要
VAEとは：「深層 隠れ変数モデル」 の学習を深層学習で効果的に行うフレームワーク<br>　→　結果としてモデル内の観測変数の分布$p(\bf x)$はよく真のデータ分布とfitする。

でも本当は観測変数$\bf x$と隠れ変数$\bf z$の同時分布$p(\bf x, \bf z)$を推定したい。（できれば隠れ変数の事前分布と事後分布も。）<br>→　これは不可能と言われている。なぜならモデルが識別可能ではない（＝ 一意でない ＝ unidentifiable）から。

しかし我々はあるシンプルな変換を用いることで、<strong>隠れ変数と観測変数の同時分布の特定が可能であることを示した。</strong>これはdisentanglementにも繋がる。

これは隠れ変数の、「付加的な観測変数（クラスラベルなど）による条件付き事前確率分布」によって達成される。

↓よく分からない<br>
<cite>We build on recent developments in nonlinear ICA, which we extend to the case with noisy, undercomplete or discrete observations, integrated in a maximum likelihood framework. The result also trivially contains identifiable flow-based generative models as a special case.</cite>

## 序論
VAEは隠れ変数（＝未観測変数or潜在変数）のある確率モデルとその推論を学習する技術の枠組みである。<br>
原則として、学習済のVAEにて隠れ変数を周辺化したデータの分布は、（未知の）真のデータ分布を良く近似する。なので人工的なデータ生成も可能になる。

ここでは更に観測変数と隠れ変数の同時分布を学習したい。でも観測変数は観測できるが隠れ変数は観測できないのでこれは不可能。<br>
仮になんとかして同時分布が学習できれば、隠れ変数の事前分布と事後分布も近似することもできる。<br>
応用先：データの背後にある隠れ変数の構造を捉えたり、ある観測データの元となる隠れ変数を推定することもできる。

<strong>モデルが<font color="Red">識別可能（= 一意 = indentifiable）</font>（後述）でなければ同時分布の学習はできない。</strong>VAEの元論文では、隠れ変数を周辺化した分布（＝データ分布を近似する分布）を推定するようなパラメタの学習方法に<strong>のみ</strong>言及している。

VAE論文の参考文献をみると、方向性としてはdisentanglementがやりたいみたい。特にβ-VAEなどはこの典型例だ。でもモデルの識別可能性やモデルの隠れ変数についての理論的な証明には言及していない。<br>
GANを用いて最も独立となるような成分を抜き出す研究もある（Brakel and Bengio）。<br>
しかしこれらの先行研究でのモデルは識別可能ではない。それは<cite>non-conditional latent priors<cite>のせいだ。（どゆこと？）←あとで示す。

最新の非線形独立成分分析(ICA)の理論研究において、初めて「深層 隠れ変数モデル」の識別可能性についての言及があった（Hyvarinen and Morioka）。<br>
<strong>非線形ICAは、反転可能な非線形変換によって観測変数へ変換されるお互いに独立な隠れ変数を、観測変数から推定する厳密なフレームワークを提供する。</strong><br>
ただし少し条件が必要になる。なぜなら一般的には隠れ変数から観測変数への変換に非線形関数を想定する場合、隠れ変数の推定は不可能と言われているためだ。

<strong>この論文では、比較的緩い条件を入れてやれば、VAEにおける隠れ変数$\bf z$と観測変数$\bf x$の同時分布$p(\bf x,\bf z)$は識別可能で学習可能であることを示す。そしてVAEとICAの橋渡しをする。</strong><br>
この二つの補完的な手法の、教師なし表現学習における見解を述べる。<br>
そのために、<strong>付加的に観測された変数での条件付き分布の積で表現される隠れ変数の事前分布</strong>を用いる。（付加的に観測された変数：ラベルや時間indexなど）←教師なしじゃねぇな(笑)<br>
これはVAEだけじゃなく、一般的な隠れ変数モデルに適用できる理論だ。しかしVAEは、は巨大なデータセットとリッチなモデル（＝DNN）を用いることができる点で、隠れ変数の推定を効果的に行えるモデルである。<br>
真の確率モデルが分かっている人工データ実験において、識別可能なVAEによる隠れ変数の観測変数の同時分布の推定結果が真の同時分布を良く近似できていることを確認した。

## 深層隠れ変数モデルの識別不可能性

## 深層隠れ変数モデルって？
観測変数$\bf x\in\mathcal{R}^d$と隠れ変数$\bf z\in\mathbb{R}^n$がある。確率モデルは
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Ap_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+x%2C%5Cbf+z%29%3Dp_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+x%7C%5Cbf+z%29p_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+z%29%0A%5Cend%7Balign%2A%7D%0A">

