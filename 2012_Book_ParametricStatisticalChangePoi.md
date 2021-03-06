# 前置き
- ここで以降挙げていく特定のモデルの変化点を検出する体系的な方法を記載。
- 変化点とは，観測系列がある時点まではある分布に従い，その時点以降は別の分布に従うような時間の点である。
- 変化点検知の歴史は1950年台にまで遡る。
  - 以降40年くらいは主に以下の研究。
    - 正規分布からのi.i.d.サンプルの系列の平均の変化に起因する変化点検知
    - 線形回帰モデル、もしくは線形自己回帰モデルを元にした変化点検知
  - 手法は尤度比検定、ノンパラ、ベイズが多かった。
  - ガンマ分布や指数分布に着目した研究は少数。
- ここでは以下の内容を尤度比検定、ベイズ、情報量基準の観点から議論する。
  - 正規乱数の変化（平均の変化、分散共分散構造の変化、両方の変化）
  - 回帰モデル、ガンマモデル、指数モデル、離散値をとる系列の変化点

# 内容
## 目次
  1. 前置き
     1. 導入
     2. 問題の定式化
     3. 背後のモデルと方法論
  2. １変数正規分布
  3. 多変量正規分布
  4. 回帰モデル
     1. 文献のレビュー
     2. 単純な線形回帰モデル
        1. 情報量基準からのアプローチ
        2. ベイズからのアプローチ
        3. 株価データへの応用
     3. 多変量の線形回帰モデル
  5. ガンマモデル
  6. 指数モデル
  7. ハザード関数のための変化点検知？
  8. 離散値のモデル
  9.  その他の変化点検知

## 前置き
### 導入
変化点の応用先はたくさんあり、実現すればより不利益を回避し、利益を享受できる。例えば、
1. 株価の変動：投資家にとって変化点を事前に察知できると嬉しい。あとはあるイベントが株価へ有意な変化をもたらしたかを定量的に議論できる。
2. 品質制御：工場での生産物の品質の変化は避けたい。
3. 交通死亡事故率：制限速度の緩和は事故死亡率に変化をもたらしたのか？
4. 地質データ
5. 遺伝子系列：遺伝子発現の変化点は知りたい（癌研究とか）

2.の例のみオンライン(監視)変化点検知、他はオフラインつまり後から解析すれば言いタイプ。
オンライン変化点検知は品質管理、公衆衛生監視、信号処理の分野で発展(Mei, 2006)。Fearnhead and Liu (2007)はオンライン多次元変化点検知の研究を、Wu(2007)は平均過程に起こるスパースな変化点の逐次検知手法を開発？

**でもここではオフライン変化点検知を扱うよ！**<br>
オフライン変化点検知の二つの側面
1. 観測系列に変化点があるかどうかを検知
2. 観測系列内の変化点の数とその時点を検知

### 定式化
${\bf x}_1,{\bf x}_2,\cdots,{\bf x}_n$を**独立な**確率変数(ベクトル)とし、それぞれの確率分布関数を$F_1,F_2,\cdots,F_n$とする。<br>変化点検知における帰無仮説は
$$
H_0:F_1=F_2=\cdots=F_n
$$
対立仮説は
$$
H_1:F_1=\cdots=F_{k_1}\neq F_{k_1+1}=\cdots=F_{K_2}\neq F_{k_2+1}=\cdots=F_{K_q}\neq F_{k_q+1}==\cdots=F_{k_n}
$$
ここで$1<k_q<k_2<\cdots<k_q$で、$q$は未知の変化点数、$k_i$は推定したい変化点の場所である。<br>
もし$F_1,F_2,\cdots.F_n$が同一のパラメトリック分布族$F(\theta), \theta\in\mathcal{R}^p$に属するなら、変化点検知は以下のように変換できる。<br>
帰無仮説は
$$
H_0:\theta_1=\theta_2=\cdots=\theta_n=\theta(未知)\tag{1.1}
$$
対立仮説は
$$
H_1:\theta_1=\cdots=\theta_{k_1}\neq \theta_{k_1+1}=\cdots=\theta_{K_2}\neq \theta_{k_2+1}=\cdots=\theta_{K_q}\neq \theta_{k_q+1}=\cdots=\theta_{k_n}\tag{1.2}
$$
また、対立仮説を以下のようにした、**流行性変化点検知**というのもある。
$$
H_1:\theta_1=\cdots=\theta_{k}=\alpha\neq \theta_{k+1}=\cdots=\theta_{t}=\beta\neq \theta_{t+1}=\cdots=\theta_{n}=\alpha
$$
流行性変化点検知の詳細はLevin and Kline (1985)、Ramanayake (1998)を参照。

変化点検知はパラメトリック、ノンパラメトリック、回帰、時系列、逐次、ベイズなどいろんな想定が可能。

### 背後のモデルと方法論
- 最尤比検定
- ベイズ検定
- ノンパラメトリック検定
- 確率過程
- 情報理論的アプローチ
  
などの方法が取られることが多い。以下、代表的な研究例
- Chernoff & Zacks(1964)：二次損失関数を用いた事前一様分布の平均を推定するベイズ推定量の作成。(?)
- Sen & Srivastava(1975a,b)：正規分布に従う観測系列の平均の単一変化点の存在を検定するための検定統計量の厳密分布と漸近分布を導出。
- Later (1973, 1980)：上記を多変量化。
- Hawkins(1977) & Worsley(1979)：上記の場合における分散既知&分散未知の場合の帰無分布の導出
- Srivastava ＆ Worsley(1986)：多変量正規平均の多重変化を研究。ボンフェローニ不等式に基づいて尤度比検定統計量の帰無分布を近似。
- Guan (2004)：セミパラメトリックな変化点検知？
- Gurevich & Vexler(2005)：ロジスティック回帰における変化点検知。
- Horuath et al.(2004)：線形モデルの変化点問題を研究。
- Ramanayake (2004)：ガンマ分布の形状パラメータの変化点を検出するための検定を提供
- Goldenshluyer & Tsbakov & Zeev(2006)：間接的なノイズの多い観測系列に対する最適なノンパラメトリック変化点推定を提供。
- Osorio & Galea(2006)：Student-t分布の線形回帰モデルの変化点問題を調査
- Kirch & Steinebach(2006)：変化点の時点を推定する検定統計量の分布? permitation method?
- Juruskova (2007)：3つのパラメータからなるワイブル分布の変化点検知
- Wu (2008)：回帰問題における変化点分析？
- Vexler et al. (2009)：変化点分析の文脈での分類問題の研究？
- Ramanayake & Gupta(2010)：指数分布の変化点検知

異常は**ランダム系列の中に単一の変化点がある場合**が多い。しかし複数の変化点の問題を扱ったのは少ない。
- Incl´an and Tiao (1994)：CUSUM法を用いた複数の変化点問題（多重変化点検知）の検定と推定
- Chen and Gupta(1995)：正規分布の系列の平均ベクトルと共分散の同時変化に対する順序統計量の尤度（Lehmann, 1986参照）の漸近的帰無分布を導出。後に多重変化点検知。

#### 二値分割法
多次元の系列における変化点の数とその位置を検出する手法として**二値分割法** (**the binary segmentation procedure**)と呼ばれる方法がある。<br>
複数の変化点の検出に広く利用されており、変化点の数とその位置を同時に検出でき、計算時間を大幅に節約できるというメリットがある。

${\bf x}_1,{\bf x}_2,\cdots,{\bf x}_n$を独立な確率変数(ベクトル)とし、それぞれの確率分布関数を$F_1(\theta_1),F_2{\theta_2},\cdots,F_n(\theta_n)$とする。<br>
もし当てたい変化点が(1.1)式と(1.2)式で定式化できるなら、この二値分割法は有効。<br>
一般的なステップは以下の通り
##### Step1 
単一の変化点VS変化点なし
$$
H_0:\theta_1=\theta_2=\cdots=\theta_n=\theta(未知)\tag{1.1}
$$
$$
H_1:\theta_1=\theta_2=\cdots=\theta_k\neq\theta_{k+1}=\cdots=\theta_n\tag{1.3}
$$
の検定を行う。もし$H_0$が受容されれば終わり。棄却されれば、変化点が一つあるということで次のステップへ。
##### Step2 
変化点の前後の系列で分割し、それぞれでStep1を行う
##### Step3 
変化点がなくなるまでリピート
##### Step4 
以上の作業によって得られた変化点を$\{\hat k_1,\hat k_2,\cdots\hat k_q\}$、変化点の数の推定量を$q$とする。

正規分布の系列に対する議論が多いのは、これが最も一般的だからである。


## 回帰モデル
### 文献
変化点検知における回帰モデルとは、説明変数$(x,y)$が別の分布から得られた場合に、どうやって回帰モデルを設計するか、というお話。