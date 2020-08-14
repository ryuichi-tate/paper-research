# Wasserstein GAN

もうこのスライドでいいんじゃないかな。
https://www.slideshare.net/DeepLearningJP2016/dlwasserstein-gantowards-principled-methods-for-training-generative-adversarial-networks


本物のデータ分布$\mathbb{P}_r$を作りたい。
つまり、駆動するごとに$\mathbb{P}_r$からサンプリングした値を出力してくれる機械(モデル)が欲しい。

使って良いのは分布$\mathbb{P}_r$からのサンプリングデータのみ。

最尤推定法は、パラメトリックな分布$\mathbb{P}_{\theta}$を用意する。$\theta$を動かして一番$\mathbb{P}_r$に**近い**$\mathbb{P}_{\theta}$を探すぜ！と言う感じ。
ここで言う「近い」とは、分布どうしの距離$\rho(\mathbb{P}_{\theta},\mathbb{P}_r)$を定義して、この距離が小さいと言うこと。
(例えば$KL[\mathbb{P}_r||\mathbb{P}_{\theta}]$)

ニューラルネットを使う場合のパラメトリックな分布$\mathbb{P}_{\theta}$とは：一様分布の出力をパラメータ群$\theta$のニューラルネットで変換した値たちを$\mathbb{P}_{\theta}$としましょ

GANは分布の距離を計算する機構(Discriminatorもしくはcritic)と、一様分布から$\mathbb{P}_{\theta}$へ変換する機構(generator)の二つのニューラルネットからなる。
この二つを同時に学習させるようなLossの設計が大事。
つまり、Lossは分布間距離を求めるためのLossと、データ分布$\mathbb{P}_r$とパラメトリック分布$\mathbb{P}_{\theta}$の分布間距離をなるべく小さくするLossの二つの項からなる。

この論文はいい分布間距離について議論している。

「いい」って何？
→ 第1章では、この分布間距離が$\theta$の変化に対して連続でなければならないと言った。

>If $\rho$ is our notion of distance between two distributions, we would like to have a loss function $\theta \mapsto\rho(\mathbb{P}_{\theta},\mathbb{P}_r)$ that is continuous, and this is equivalent to having the mapping $\theta \mapsto\mathbb{P}_{\theta}$ be continuous when using the distance between distributions $\rho$.
>$\rho$が２つの分布間の距離で、すなわち誤差関数である。こいつが連続になって欲しい。



第二章ではいろいろな分布間距離について紹介している。

- Total Variation (TV) 距離$$\delta(\mathbb{P}_r,\mathbb{P}_g)=\sup_{A\in\Sigma}|\mathbb{P}_r(A)-\mathbb{P}_g(A)|$$
- Kullback-Leibler (KL) 距離$$KL(\mathbb{P}_r||\mathbb{P}_g)=\int P_r(x)\log\left(\frac{P_r(x)}{P_g(x)}\right)dx$$
- Jensen-Shannon(JS) 距離$$JSD[\mathbb{P}_r||\mathbb{P}_{\theta}]=KL(\mathbb{P}_r||\frac{\mathbb{P}_r+\mathbb{P}_g}{2})+KL(\mathbb{P}_g||\frac{\mathbb{P}_r+\mathbb{P}_g}{2})$$
- Earth-Mover(EM) 距離$$W(\mathbb{P}_r,\mathbb{P}_g)=\inf_{\gamma\in\prod(\mathbb{P}_r,\mathbb{P}_g)}\mathbb{E}_{(X,Y)\sim\gamma}[|X-Y|_2]\tag{1}$$


でもKL距離って、$\mathbb{P}_{\theta}$が$0$に限りなく近い時発散しちゃったり、逆に$\mathbb{P}_r$が$0$になると(例え分布が似てなくても)距離も$0$になっちゃうことがあるよね、数式的に。

実際に$Z\sim U(0,1)$、$\mathbb{P}_r\sim (Z,0)$、$\mathbb{P}_{\theta}\sim (Z,\theta)$で距離を計算してみたヨ

Earth-Mover(EM) 距離の実現ついて、実際は$\mathbb{P}_r$と$\mathbb{P}_g$の同時分布$\gamma$の中で$\mathbb{E}_{(X,Y)\sim\gamma}[|X-Y|_2]$を最小にする$\gamma$を探す。
この最小化問題の双対問題として、
>$$\sup_{f\in F}\left(\mathbb{E}_{X\sim\mathbb{P}_r}[f(X)]-\mathbb{E}_{X\sim\mathbb{P}_g}[f(X)]\right)$$

と言うのがあるので、この$f$をニューラルネットで近似する。
つまり、Discriminatorのニューラルネットのパラメータを$w$とすると、損失は
$$-\left(\mathbb{E}_{X\sim\mathbb{P}_r}[f_w(X)]-\mathbb{E}_{X\sim\mathbb{P}_g}[f_w(X)]\right)$$
実際に期待値を計算するときはバッチの平均をとる。