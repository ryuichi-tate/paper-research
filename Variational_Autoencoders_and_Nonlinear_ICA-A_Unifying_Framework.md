# Variational Autoencoders and Nonlinear ICA: A Unifying Framework

## 概要
VAEとは：「深層 隠れ変数モデル」 の学習を深層学習で効果的に行うフレームワーク<br>　→　結果としてモデル内の観測変数の分布$p(\bf x)$はよく真のデータ分布とfitする。

でも本当は観測変数$\bf x$と隠れ変数$\bf z$の同時分布$p(\bf x, \bf z)$を推定したい。（できれば隠れ変数の事前分布と事後分布も。）<br>→　これは不可能と言われている。なぜならモデルが一意でない（unidentifiable）から。

しかし我々はあるシンプルな変換を用いることで、<strong>隠れ変数と観測変数の同時分布の特定が可能であることを示した。</strong>これはdisentanglementにも繋がる。

これは隠れ変数の、付加的な観測変数（クラスラベルなど）による条件付き事前確率分布　によって達成される。

↓良く分からない<br>
<cite>We build on recent developments in nonlinear ICA, which we extend to the case with noisy, undercomplete or discrete observations, integrated in a maximum likelihood framework. The result also trivially contains identifiable flow-based generative models as a special case.</cite>

## 序論
VAEは隠れ変数（＝未観測変数or潜在変数）のある確率モデルとその推論を学習する技術の枠組みである。<br>
原則として、学習済のVAEにて隠れ変数を周辺化したデータの分布は、（未知の）真のデータ分布を良く近似する。なので人工的なデータ生成も可能になる。

更に観測変数と隠れ変数の同時分布を学習したい。でも観測変数は観測できるが隠れ変数は観測できないのでこれは不可能。<br>
仮になんとかして同時分布が学習できれば、隠れ変数の事前分布と事後分布も近似することもできる。<br>
応用先：データの背後にある隠れ変数の構造を捉えたり、ある観測データの元となる隠れ変数を推定することもできる。

モデルが<font color="Red">識別可能</font>（後述）でなければ同時分布の学習はできない。VAEの元論文では、隠れ変数を周辺化した分布（＝データ分布を近似する分布）を推定するようなパラメタの学習方法に<strong>のみ</strong>言及している。