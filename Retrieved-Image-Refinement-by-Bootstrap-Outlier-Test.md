# 概要
異常検知の話。<br>
異常度（外れ値スコア）の分布をパラメトリックに仮定せずに仮説検定を行い、異常検知を行う手法を提案する。<br>
仮説検定に必要な検定統計量の分布は，ブートストラップ法を用いて与えられたデータから推定する。<br>
画像検索の質を向上させた。

# イントロ
Hawkins [16]によると、外れ値とは、他のデータポイントから離れているため、別のメカニズムによって引き起こされたと疑われるポイントと定義されている。
- 仮説検定を用いた方法[2,4,15]
- k-近傍距離に基づく方法[5,28]

がよく使われる。<br>
仮説検定を使うというのは、観測系列のある時点が外れ値か否かを検定する、ということである。ここにデータ生成のモデリングは必ずしも必要ではない。<br>
k-近傍距離や角度(程度？)ベースの手法[20]、isolation forest[22]など多くの異常検知問題では、直接外れ値検知をするのではなく、異常度 (外れ値スコア)を計算することが多いが、スコアを算出する（つまりどれほど異常なのかを数値で出す）必要は必ずしもない。

ｋ-最近傍距離に基づく外れ値検出の場合は、異常スコアの上位k点を異常点と判断するが、普通は異常スコアを閾値で切る。しかし
- 閾値の決め方は経験則
- 単一のスコアを単一の閾値で切るだけでは検知できない異常もある

ここでは仮説検定の客観性とk-近傍法の柔軟性を兼ねた手法を提案。<br>
k-近傍法に基づく任意の異常スコア関数を用い、与えられた時点が異常か否かを検定する。
- 統計的に適切な閾値を算出することが可能となる。
- ブートストラップ法を用いて統計量の分布を作成するので、従来の正規分布を仮定した異常検知の手法より柔軟。

という利点がある。

# 定式化
$\mathcal{X}$が観測データを出力するドメインとする。観測は$D=\{x_1,x_2,\cdots,x_n\}\subset\mathcal{X}$。ある検査点$x$の異常スコアを$s_k(x;D)$とする。検査点が異常かどうかを判定したい。（教師なし学習？）

ここで$s_k:\mathcal{X}\rightarrow\mathbb{R}$はパラメータ$k$の関数。検査点の外れ具合の程度を与える。従来はこのスコアに対しユーザが閾値を設定していた。

閾値をユーザが設定するデメリットは、
- 客観性に欠ける
- 解釈性がない
- データが変わった時に適切な閾値も変わる
- 単一の閾値だけでは検出できない場合がある<br>（あるタイミングでこの値が出るのは別に異常じゃないけど、別のタイミングで同じ値が出たらそれは異常として欲しい、みたいな。）

つまり閾値を固定してしまうのはよくない。

データセット$D\subset\mathcal{X}$と異常スコア関数$s_k(\cdot;D)$が与えられた時、ある検査点$x$が異常点か否かを統計検定したい。

# 異常検知の統計検定
適応的(流動的？)に閾値を決定したい。

仮説１:ある検査点$x$が外れ値でないならば、その異常スコア$s_k(x)$は付近の観測点の異常スコアの平均値とほぼ一致する。

$\mathcal{N}_k(x)$を検査点$x$の$k$近傍のデータ点集合とする。検査点の近傍の異常スコアの平均は、
$$
\bar s_k(x;D) = \frac{1}{k}\sum_{y\in\mathcal{N}_k}s_k(y;D)\tag{1}
$$

検査点$x$に対し、ある未知の確率分布上での$s_k(x;D)$の平均を$\mu$、$\bar s_k(x;D)$の平均を$\bar\mu$とする。仮説1に基づいた統計検定は
$$
H_0:\mu=\bar\mu\\
H_1:\mu>\bar\mu\tag{2}
$$
その検定統計量は、
$$
t=s_k(x;D) - \bar s_k(x;D)\tag{3}
$$
を用いる。

検定を行うには検定統計量$t$の確率分布を知る必要があるが、これおbootstrap-samplingによって近似する。

bootstrap-samplingを用いると、データセットのみから統計量の性質を評価できる。ここでは検定統計量$t$のbootstrap検定統計量として、
$$
\hat t^{\ast} = s_k(x;\mathcal{N}_k^{\ast}(x))-\bar s_k(x;\mathcal{N}_k^{\ast}(x))\tag{4}
$$
を考える。ここで$\mathcal{N}_k^{\ast}(x)$は、検査点$x$の$k$近傍のデータ集合$\mathcal{N}_k(x)$から$k$回bootstrap-samplingした集合を表す。一つの$\mathcal{N}_k(x)$からbootstrap-samplingは何度でも実行可能なので、これを$B$回行ったとする。検定統計量の分布はこの$B$回のサンプリング結果$\{\hat t^{\ast1},\hat t^{\ast2},\cdots,\hat t^{\ast B}\}$を用いてその性質を評価する。<br>
方法としてはパーセンタイル法、正規近似、$BC_a$法など。（[11]参照）中でもパーセンタイル法が良さそうなので採用。<br>
bootstrap-samplingにより$B$個の検定統計量$\hat t^{\ast b}$の実現値を計算し、ソートし、パーセント点を計算する。（順序統計量）<br>
帰無仮説の元(つまり検査てんは異常ではない場合)では検定統計量$t$は$0$付近に散らばる。検定における$p$値は、$B$個の実現値のうち$0$以下になった数の割合とする。($p$値とは帰無仮説の元で検定統計量が実現するはずの値を実際に実現する確率であり、これを計算した時にある水準より小さい場合は帰無仮説を棄却する、というもの。)つまり$p$値がある水準より小さかったら、その検査点を外れ値と判定する。<br>
実験では$B=10^5$