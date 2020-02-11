# Multiple Kernel Learning with Gaussianity Measures

# Abstract
カーネル法は非線形多変量解析に有効であることが知られている。

でもどんなカーネルを選べばいいんだろう？<br>
→多重カーネル学習(MKL)で行くか。これは有望なカーネル最適化手法の一つ。(たくさんのカーネルを凸結合したものを使う？)

### Fisher判別分析(FDA)
FDAは、特徴空間における各クラスのデータ分布が共有共分散構造を持つガウス分布である場合、Bayes最適な判別境界?を与える。(FDAってクラス間分散を大きく、クラス内分散を小さくするするような空間にデータを射影すること。)<br>
この事実に基づいて、ガウス性の概念に基づくMKLフレームワークを提案した。具体的な実装として、カーネル関数の凸結合に関連する特徴空間におけるガウス性を測定するために経験的特性関数を採用し、2つのMKLアルゴリズムを導出した。

# 第2章　ガウス性のMultiple Kernel Learning

FDAでは二クラスのデータの、特徴量空間における分布がどちらも正規分布で、かつ共分散行列が等しくなって欲しい。(ベイズ最適だから。)

そういう損失関数を考えましょ。

# 第3章　技術的な下準備(Technical Preliminary)

## 経験特性関数

### 特性関数
ってこういう関数。

$$c(\mathbf{t})=\mathbb{E}_{p_X}[e^{i\mathbf{t}^TX}]$$

$p_X$は確率変数$X$の密度関数。特性関数は、一つの分布に対して特有の関数になる。<br>
経験特性関数(ecf)はこれ。

$$c_{\mathcal{D}}(\mathbf{t})=\mathbb{E}_{p_D}[e^{i\mathbf{t}^TX}]=\frac{1}{n}\sum_{x_j\in \mathcal{D}}e^{i\mathbf{t}^Tx_j}$$

$\mathcal{D}$はデータ集合。$\mathcal{D}=\{x_j\}_{j=1,2,3,\cdots n}$

### 経験特性関数の性質

- いくつかの一般的な制限の下で、$c_{\mathcal{D}}(\mathbf{t})$は、特性関数$c(\mathbf{t})$にほぼ確実に均一に収束する。<br>[Feuerverger, A., & Mureika, R. (1977). The empirical characteristic function and its
applications. ](https://projecteuclid.org/download/pdf_1/euclid.aos/1176343742)<br>
これは任意次元$d$に対して、経験的特性関数の特性関数へ収束することが、大数の法則とGlivenko‐Cantelli定理(バプニク、1998)を用いて証明された。<br>
(Glivenko‐Cantelli定理)<br>
    $X$の真の分布関数を$F_X(x)$, 経験分布関数を$F_n(x)$とした時、$n\rightarrow\infty$で
    $$\sup_{x}|F_X(x)-F_N(x)|\rightarrow 0$$
    概収束する。<br>
確率変数を有限次元の特徴量空間に変換しても分布固有の特性関数の性質は変化しないらしい。適合度検定や分布形状検定、ICAにも使われる。

ガウス分布を考える時は、特性関数の尺度(絶対値とか)を計算すればいい。(偏角はいらない？)


- ある確率変数の特性関数の対数が二次型式(quadratic form)になったら、その確率変数はガウス分布である。

$$ \log{|c(\mathbf{t})|^2} = \mathbf{t}^T\Sigma \mathbf{t} $$

多変量ガウス分布の特性関数を

$$c^{\ast}(\mathbf{t})=e^{i\mathbf{t}^T\mathbf{\mu}}e^{\frac{1}{2}{\mathbf{t}^T\Sigma{\mathbf{t}}}}$$

とする。他変量ガウス分布からのサンプルから作られるecfを$c^{\ast}_{\mathcal{D}}(\mathbf{t})$とすると、これは標本平均と標本分散を$c^{\ast}(\mathbf{t})$に代入すると得られる。

### カーネル化された経験特性関数

経験特性関数の式とか、ガウス分布の特性関数とか、みんな内積onlyじゃん。<br>
→カーネル$k_{\beta}$で計算できるのでは？<br>
とりあえず特徴量空間$\mathcal{H}_{\beta}$への写像を考える。

$$\phi_{\beta}:\mathbf{X}\longmapsto\mathbf{\phi}_{\beta}(\mathbf{X})$$

つまり関数$\phi_{\beta}$で確率変数を特徴量へ変換したあとのことを考える。
<!-- 特徴量の空間(双対空間？つまり一次元の空間？)での特性関数は、

$$c(\boldsymbol{\tau})=\mathbb{E}_{p_{\phi_{\beta}}(X)}\left[e^{i\boldsymbol{\tau}\phi_{\beta}(X)}\right]$$ -->

一般的な空間(つまり多次元の特徴量？)へ写像した時の特性関数は

$$c(\boldsymbol{\tau})=\mathbb{E}_{p_{\phi_{\beta}}(X)}\left[e^{i<\boldsymbol{\tau},\phi_{\beta}(X)>}\right]$$

$\boldsymbol{\tau}$はベクトルね。<br>
経験特性関数はこうだよね。

$$c_{\phi_{\beta}(\mathcal{D})}(\boldsymbol{\tau})=\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}e^{i<\boldsymbol{\tau},\phi_{\beta}(\mathbf{x}_j)>}$$
<!-- <blockquote> -->
特性関数の性質として、特性関数の引数$\boldsymbol{\tau}$の定義域($\mathcal{H}_{\beta}$)全ての点において特性関数が一致しているなら、二つの確率変数の分布は一致する、と言える。つまり、原理的に経験特性関数も全ての点での値を調べる必要がある。<br>しかし特徴量空間における経験特性関数$c_{\phi_{\beta}(\mathcal{D})}(\boldsymbol{\tau})$をカーネル関数で表現したいなら、

$$\boldsymbol{\tau}\in\phi_{\beta}(\mathcal{X})$$

すなわち元の生データの定義域$\mathcal{X}$を関数$\phi_{\beta}(\cdot)$によって写像した空間についてのみ調べれば良い。つまり元のデータ空間の点$\mathbf{t}$について

$$\phi_{\beta}({\mathbf{t}})=\boldsymbol{\tau}$$

となる$\mathbf{t}$が存在するような$\boldsymbol{\tau}$について調べれば良い。<br>
なんかこうしてみると、$\mathbf{t}$も確率変数$\mathbf{X}$も同じ空間の一点なんやなぁ。
<!-- </blockquote> -->


$$c_{\phi_{\beta}(\mathcal{D})}(\boldsymbol{\tau})=\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}e^{i<\boldsymbol{\tau},\phi_{\beta}(\mathbf{x}_j)>}$$
$$=\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}e^{i<\phi_{\beta}({\mathbf{t}}),\phi_{\beta}(\mathbf{x}_j)>}$$
$$=\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}e^{ik_{\beta}(\mathbf{t},\mathbf{x}_j)}$$
$$=\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}\cos\left\{k_{\beta}(\mathbf{t},\mathbf{x}_j)\right\}+i\times\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}\sin\left\{k_{\beta}(\mathbf{t},\mathbf{x}_j)\right\}$$

元のデータ空間における$\mathbf{t}$を特徴量空間に写した$\boldsymbol{\tau}$に対する経験特性関数の値$c_{\phi_{\beta}(\mathcal{D})}(\boldsymbol{\tau})$を$c_{\mathcal{D}}(\mathbf{t};\beta)$とかく。

$$|c_{\mathcal{D}}(\mathbf{t};\beta)|^2=\left(\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}\cos\left\{k_{\beta}(\mathbf{t},\mathbf{x}_j)\right\}\right)^2+\left(\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}\sin\left\{k_{\beta}(\mathbf{t},\mathbf{x}_j)\right\}\right)^2$$


とりあえずこの話はここでおいておいて、次にガウス分布のecfのお話に移る。

特徴量空間における確率分布の分散の推定量$\hat\Sigma_{\mathcal{D}}$は、データ${\mathcal{D}}$を$\phi_{\beta}$で写した点の集合に対して標本分散を取ったもので表せる。


$$\hat\Sigma_{\mathcal{D}}=\frac{1}{n}\sum_{\mathbf{x}_j\in\mathcal{D}}\left\{\phi_{\beta}({\mathbf{x}_{i}})-\overline{\phi_{\beta}({\mathbf{x}}})\right\}\left\{\phi_{\beta}({\mathbf{x}_{i}})-\overline{\phi_{\beta}({\mathbf{x}}})\right\}^T$$


$k_{\beta}(\mathbf{x},\mathbf{y})=<\phi_{\beta}({\mathbf{x}}),\phi_{\beta}(\mathbf{y})>$と定義したが、これを複数のカーネルの凸結合で近似することを考える。
$$k_{\beta}(\mathbf{x},\mathbf{y})=\sum_{s=1}^S\beta_sk_s(\mathbf{x},\mathbf{y})$$
これが多分第二章とかでやってたお話。

.

.

.


なんかごちゃごちゃすると、<strong>特徴量空間においてガウス分布に従う確率変数の、特徴量空間におけるecf</strong>は、

$$c^{\ast}_{\mathcal{D}}(\mathbf{t};\beta)=e^{i\hat\mu_{\beta}\mathbf(t)} \exp\left\{-\frac{1}{2n}\sum_{\mathbf{x}_j\in\mathcal{D}}(k_{\beta}(\mathbf{t},\mathbf{x}_j)-\hat\mu_{\beta}(\mathbf{t}))^2\right\}$$
$$\hat\boldsymbol{\mu}_{\beta}(\mathbf{t}) = \frac{1}{n}\sum_{x_j\in\mathcal{D}}k_{\beta}(\mathbf{t},x_j)$$

となるらしい。

結局、関数が同じってことをいうためにどのポイントを比較すりゃいいんじゃ...
理論的には全ての点を調べる必要がある！と言われても...<br>
というわけでEriksson & Koivunenさんたちがやったのは、データ点に相当する$\mathbf{t}\in\mathcal{D}$についてのみ調べればいい、というICAのための検定統計量。えーでもデータ点莫大だし、計算したくない...<br>
そこで我々が、そんなデータ点全て調べなくても、一点だけ調べれば良い方法を見つけました！

与えられたデータから任意の点$\mathbf{t}$を選ぶ。損失を

$$M_G(\phi_{\beta}(\mathcal{D})) = \frac{|\mathcal{D}|}{n}(S^{\ast}_{\mathcal{D}}(\mathbf{t};\beta)-S_{\mathcal{D}}(\mathbf{t};\beta))^2$$

$$S^{\ast}_{\mathcal{D}}(\mathbf{t};\beta)=-\log|c^{\ast}_{\mathcal{D}}(\mathbf{t};\beta)|^2$$

$$S_{\mathcal{D}}(\mathbf{t};\beta)=-\log|c_{\mathcal{D}}(\mathbf{t};\beta)|^2$$

で定義することを提案する。

#  我々がやらなければならないこと
kernelだけでなく射影の学習せねばならん。