# Independent Components Analysis by Direct Entropy Minimization

Erik G. Miller, John W. Fisher III

http://vis-www.cs.umass.edu/papers/ICA.pdf

RADICAL ：  Robust, Accurate, Direct ICA aLgorithm の略<br>
（最後の無理矢理な感じ嫌いじゃない😎）

## 概要
効率的なエントロピー推定に基づく独立成分分析を行う。確率変数の同時分布と個々の確率分布（周辺分布）の関の分布とのKL距離を推定した物を小さくしたいのは他のICAと一緒。このKL距離の推定を、統計的なエントロピー推定で使われる手法を用いて行う。

## 序論
RADICAL開発の根底にある原理
1. 統計的な独立性の指標の代替値ではなく、直接こいつを最適化しよう。
2. 密度関数の明示的な推定は避けたい？（だって必要ないから）
3. 統計分野の方で一次元のエントロピーの推定量としてロバストでいい感じの奴があるらしい
4. 局所解がたくさんあるからSGDは使えないけど、エントロピーの推定量が完璧だからglobal minimamが探せるぜ

- 第1章では問題の整理とcontrast functionの議論とかする
- 第2章ではorder統計に基づく一次元エントロピー推定量の話
- 第３章ではsourceが二次元の場合のICAで確認。あとsourceがどんな分布でもRADICALが頑健に仕事してくれることを示すよ
- 高次元(16次元)の場合でやる

$$\bf Y=F_{W}(X)$$

与えられたデータ$\bf X$に対し小さくしたいのは確率変数列$\{Y_i\}_{i=1}^n$のお互いの従属具合。
つまり、

$p({\bf Y})$と$p(Y_1)p(Y_2)\cdots p(Y_n)$のKL距離が小さくなればいい。

$$
Loss = KL[p({\bf Y})||p(Y_1)p(Y_2)\cdots p(Y_n)]\\
=\int p({\bf y})\log\frac{p({\bf y})}{p(y_1)p(y_2)\cdots p(y_n)}d{\bf y}\\
=\sum_{i=1}^nH(Y_i)-H({\bf Y})
$$
ただし、
$$
H({\bf X})=-\int p({\bf x})\log p({\bf x})d{\bf x}
$$
よって
$$
\hat{\bf W} = \argmin_{\bf W}\left\{ \sum_{i=1}^nH(Y_i)-H({\bf Y})\right\}
$$
を解く。そのためのエントロピー$H$の推定方法と最適化の方法について考えましょう。

Vasicek (1976)が提案しているエントロピーの推定方法とよく似た推定方法を提案します。


## 第2章：連続確率変数のためのエントロピー推定量
RADICALにおけるエントロピーの推定方法は、順序統計量を用いる。

確率変数$Z$からのサンプル$Z^{1},Z^{2},\cdots,Z^{n}$とその順序統計量$Z^{(1)},Z^{(2)},\cdots,Z^{(n)}$を考える。Vasicek (1976)によるエントロピーの推定量は
$$
\hat H(Z^{1},Z^{2},\cdots,Z^{n})=\frac{1}{N}\sum_{i=1}^{N-m_N}\log\left\{\frac{N}{m_n}(Z^{(i+m_N)}-Z^{(i)})\right\}
$$
となる。ただし$m_N\in\mathbb{N}$は$N$で決まる関数。

RADICALで用いる推定量は、ズバリ
$$
\hat H_{RADICAL}(Z^{1},Z^{2},\cdots,Z^{n})=\frac{1}{N-m}\sum_{i=1}^{N-m}\log\left\{\frac{N+1}{m}(Z^{(i+m_N)}-Z^{(i)})\right\}
$$
これ。密度を推定しなくてもいけるというね(笑)

## 第３章：二次元のRADICAL
こいつを用いてKL距離を導出して最適化したいんだけど、問題が2点。
- まず局所解に陥る。これについては網羅的に探さないといけない(泣)。（出力$Y$を回転させたものを網羅調査すればいいのかな？白色化されてれば$90$°で済みますね）
- もう一つは擬似最小値に陥る可能性。これは観測系列$X$に対して正規ノイズを付与し、代理データ$X'$を作成（データ数$R$倍）し、これで$\hat H_{RADICAL}$を計算することで対処できるっぽい。

