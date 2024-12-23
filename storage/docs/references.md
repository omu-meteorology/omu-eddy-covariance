# リファレンス

このドキュメントでは、`omu-eddy-covariance`パッケージで提供されている機能について説明します。

## フットプリント解析

- `FluxFootprintAnalyzer`: フラックスフットプリントを計算し、可視化するためのクラスです。
- `GeoFluxFootprintAnalyzer`: 地理情報を考慮したフラックスフットプリントを計算・可視化するためのクラスです。

## 移動観測解析

- `MobileSpatialAnalyzer`: 移動観測データを解析するためのクラスです。
- `MSAInputConfig`: 移動観測データの入力設定を管理するクラスです。

## 伝達関数解析

- `FftFileReorganizer`: FFTファイルを整理するためのクラスです。
- `TransferFunctionCalculator`: 伝達関数を計算し、可視化するためのクラスです。

## データ前処理

- `EddyDataPreprocessor`: 渦相関法で得られたデータの前処理を行うクラスです。
- `SpectrumCalculator`: スペクトル解析を行うためのクラスです。

## 注意事項

- 本パッケージは大阪公立大学生態気象学研究グループの構成員、または著作権者から明示的な許可を得た第三者のみが使用できます。
- データの品質管理や解析手法については、研究グループの方針に従ってください。
