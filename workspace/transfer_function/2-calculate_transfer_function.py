import os
from omu_eddy_covariance import TransferFunctionCalculator

# 変数定義
base_path = "/home/connect0459/workspace/labo/omu-eddy-covariance/workspace/transfer_function/private/2024.08.06"
tf_dir_name: str = "tf"

# tf_file_name: str = "TF_Ultra.2024.08.06.csv"
tf_file_name: str = "TF_Ultra.2024.08.06-detrend.csv"

show_cospectra_plot: bool = True
# show_cospectra_plot: bool = False
show_tf_plot: bool = True
# show_tf_plot: bool = False

# UltraのFFTファイルで使用されるキー名(スペース込み)
key_wt: str = "  f*cospec_wt/wt"
key_wch4: str = " f*cospec_wc/wc closed"
key_wc2h6: str = " f*cospec_wq/wq closed"

# メイン処理
try:
    file_path: str = os.path.join(base_path, tf_dir_name, tf_file_name)
    calculator = TransferFunctionCalculator(file_path, " f", 0.01, 1)

    # コスペクトルのプロット
    if show_cospectra_plot:
        # calculator.plot_cospectra(
        #     key_wt,
        #     key_wch4,
        #     label1=r"$fC_{wTv}$ / $\\overline{w^\\prime Tv^\\prime}$",
        #     label2=r"$fC_{wCH_{4}}$ / $\\overline{w^\\prime CH_{4}^\\prime}$",
        #     color2="red",
        # )
        # calculator.plot_cospectra(
        #     key_wt,
        #     key_wc2h6,
        #     label1=r"$fC_{wTv}$ / $\\overline{w^\\prime Tv^\\prime}$",
        #     label2=r"$fC_{wC_{2}H_{6}}$ / $\\overline{w^\\prime C_{2}H_{6}^\\prime}$",
        #     color2="orange",
        # )
        calculator.plot_cospectra(
            key_wt,
            key_wch4,
            label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
            label2=r"$fC_{wCH_{4}}$ / $\overline{w^\prime CH_{4}^\prime}$",
            color2="red",
            subplot_label="(a)"
        )
        calculator.plot_cospectra(
            key_wt,
            key_wc2h6,
            label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
            label2=r"$fC_{wC_{2}H_{6}}$ / $\overline{w^\prime C_{2}H_{6}^\prime}$",
            color2="orange",
            subplot_label="(b)"
        )

    # 伝達関数の計算
    print("伝達関数を分析中...")
    a_wch4: float = calculator.analyze_transfer_function(
        key_wt, "wTv", key_wch4, "wCH4", show_tf_plot
    )
    a_wc2h6: float = calculator.analyze_transfer_function(
        key_wt, "wTv", key_wc2h6, "wC2H6", show_tf_plot
    )
    print(f"wCH4の係数 a: {a_wch4}")
    print(f"wC2H6の係数 a: {a_wc2h6}")
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
