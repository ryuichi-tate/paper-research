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

### 深層隠れ変数モデルって？
観測変数$\bf x\in\mathcal{R}^d$と隠れ変数$\bf z\in\mathbb{R}^n$がある。確率モデルは<bf>

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%7D%0Ap_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+x%2C%5Cbf+z%29%3Dp_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+x%7C%5Cbf+z%29p_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+z%29%5Ctag%7B1%7D%0A%5Cend%7Balign%7D%0A" alt="\begin{align}p_{\boldsymbol\theta}(\bf x,\bf z)=p_{\boldsymbol\theta}(\bf x|\bf z)p_{\boldsymbol\theta}(\bf z)\tag{1}\end{align}">

という構造。$\boldsymbol\theta\in\Theta$はモデルのパラメタで、$p_{\boldsymbol\theta}(\bf z)$は隠れ変数の事前分布。隠れ変数$\bf z$の値によって決まる観測変数$\bf x$の確率分布$p_{\boldsymbol\theta}(\bf x|\bf z)$は、decoderというニューラルネットでparametarizeされる。<bf>
データの経験分布$p_{\boldsymbol\theta}(\bf x)$はこうなる。<br>
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Ap_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+x%29%3D%5Cint+p_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+x%2C%5Cbf+z%29%0A%5Cend%7Balign%2A%7D+" alt="\begin{align*}p_{\boldsymbol\theta}(\bf x)=\int p_{\boldsymbol\theta}(\bf x,\bf z)\end{align*} ">

$p_{\boldsymbol\theta}(\bf x|\bf z)$はニューラルネットでモデリングされるので、リッチなデータ分布$p_{\boldsymbol\theta}(\bf x)$でもモデリングできる。

観測データはある真の同時分布$p_{\boldsymbol\theta^{\ast}}(\bf x,\bf z)=p_{\boldsymbol\theta^{\ast}}(\bf x|\bf z)p_{\boldsymbol\theta^{\ast}}(\bf z)$から生成されているとする（$\boldsymbol{\theta}^{\ast}$は真のパラメタ）。データセット$\mathcal{D}$は<br>
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%26%5Cmathcal%7BD%7D%3D%5C%7B%5Cbf+x%5E%7B%281%29%7D%2C+%5Cbf+x%5E%7B%282%29%7D%2C%5Ccdots%2C%5Cbf+x%5E%7B%28N%29%7D%5C%7D%5C%5C%0A%26%5Cbf+x%5E%7B%28i%29%7D%5Csim+p_%7B%5Cboldsymbol%5Ctheta%5E%7B%5Cast%7D%7D%28%5Cbf+x%7C%5Cbf+z%5E%7B%5Cast%28i%29%7D%29%5C%5C%0A%26%5Cbf+z%5E%7B%5Cast%28i%29%7D%5Csim+p_%7B%5Cboldsymbol%5Ctheta%5E%7B%5Cast%7D%7D%28%5Cbf+z%29%0A%5Cend%7Balign%2A%7D" alt="\begin{align*}&\mathcal{D}=\{\bf x^{(1)}, \bf x^{(2)},\cdots,\bf x^{(N)}\}\\&\bf x^{(i)}\sim p_{\boldsymbol\theta^{\ast}}(\bf x|\bf z^{\ast(i)})\\&\bf z^{\ast(i)}\sim p_{\boldsymbol\theta^{\ast}}(\bf z)end{align*}">

で構成。<br>
ICAの文脈では$\bf z$をsourceという言葉で表現している。<br>
あと$\bf x^{(i)}\sim p_{\boldsymbol\theta^{\ast}}(\bf x)$も忘れずに。

VAEのフレームワークを用いると、周辺尤度（$p_{\boldsymbol{\theta}}(\bf x)$のこと?）を最大化するようにパラメタ$\boldsymbol{\theta}$は学習される。

### パラメタ空間 VS 関数空間(Function Space)
この論文では少々特殊なnotationをする。<br>
- $\boldsymbol{\theta}\in\Theta$は関数空間におけるモデルパラメタ
- $\bf w\in W$はニューラルネットのパラメタ

### 識別可能性とは
VAEモデルは
- 完全な生成モデル$p_{\boldsymbol{\theta}}(\bf x,\bf z)=p_{\boldsymbol{\theta}}(\bf x|\bf z)p_{\boldsymbol{\theta}}(\bf z)$
- 推定モデル$q_{\boldsymbol{\phi}}(\bf z|\bf x)$（$\bf z$の事後分布$p_{\boldsymbol\theta}(\bf z|\bf x)$を近似するもの）

を学習する。問題点は学習で得られたこの二つが実際はなんなのかは分からないことだ。分かるのは$\bf x$の周辺分布$p_{\boldsymbol\theta}(\bf x)$に意味はあるが、他の分布はあんまり意味がないということだ。

<strong>モデルの識別可能性とはこれ↓。</strong>

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cforall+%28%5Cboldsymbol%5Ctheta%2C+%5Cboldsymbol%5Ctheta%27%29%5C+%3A%5C+p_%7B%5Cboldsymbol%5Ctheta%7D%28%5Cbf+x%29%3Dp_%7B%5Cboldsymbol%5Ctheta%27%7D%28%5Cbf+x%29%5C++%5CRightarrow%5C+%5Cboldsymbol%5Ctheta%3D%5Cboldsymbol%5Ctheta%27%0A" alt="\forall (\boldsymbol\theta, \boldsymbol\theta')\ :\ p_{\boldsymbol\theta}(\bf x)=p_{\boldsymbol\theta'}(\bf x)\  \Rightarrow\ \boldsymbol\theta=\boldsymbol\theta'">

こいつが全ての$(\bf x,\bf z)$で成り立つようなモデルが欲しい。これは、学習によってデータに適合した$\boldsymbol{\theta}$を獲得した時、
- 周辺分布$p_{\boldsymbol{\theta}}(\bf x)$
- 同時分布$p_{\boldsymbol{\theta}}(\bf x,\bf z)$
- 事前分布$p_{\boldsymbol{\theta}}(\bf z)$
- 事後分布$p_{\boldsymbol{\theta}}(\bf z|\bf x)$

が全てそれぞれの真の確率分布（つまり$\boldsymbol{\theta}'$の時の場合）と一致することを表す。<br>
VAEの場合は得られた推論モデル$q_{\boldsymbol{\phi}}(\bf z|\bf x)$を用いてあるデータを生成した元のsource$\bf z^{\ast}$の推論もできる。

深層隠れ変数モデルは識別可能な保証はない。なぜなら条件付けのない隠れ変数の分布$p_{\boldsymbol{\theta}}(\bf z)$は識別不可能であるからだ。

    例えば$p_{\boldsymbol{\theta}}(\bf z)$を二次元の標準正規分布とする。回転行列$M$で$\bf z'=M\bf z$と変換した$\bf z'$を考えると、$\bf z'$は$\bf z$とは違う値になるが、$\bf z'$が従う確率分布も二次元の標準正規分布になる。