# 概要
二つの分布間距離であるEMDとは、一方の分布を他方の分布に変換するために支払わなければならない最小コストに基づく。<br>
Peleg, Werman, Romによって視覚の問題として取り上げられたのが最初？<br>
我々は画像検索に応用した。<br>
EMDは線形最適化問題の一つである輸送問題に基づくため便利なツールがたくさんある。<br>
ヒストグラムを直接比較するのと違って、定義域が一致している必要もないし、bin幅を決める必要もない。**面積が同じならEMDは真の距離と言える。**

# イントロ
画像の性質を評価する際に、多次元分布(ヒストグラム)はよく用いられる。<br>
輝度値の一次元ヒストグラムは画像の明るさを表し、三次元ならカラー画像を表す。<br>周波数の分布は画像のテクスチャ(文脈?)を評価する？