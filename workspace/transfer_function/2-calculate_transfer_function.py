import os
import matplotlib.pyplot as plt
from omu_eddy_covariance import TransferFunctionCalculator

# 変数定義
base_path = "/home/connect0459/labo/omu-eddy-covariance/workspace/transfer_function/private/2024.08.06"
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
    tfc = TransferFunctionCalculator(file_path, " f", 0.01, 1)

    # コスペクトルのプロット
    fig = tfc.create_plot_cospectra(
        key_wt,
        key_wch4,
        label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
        label2=r"$fC_{wCH_{4}}$ / $\overline{w^\prime CH_{4}^\prime}$",
        color2="red",
        subplot_label="(a)",
    )

    if show_cospectra_plot:
        plt.show()  # GUIで表示する場合

    plt.close(fig)  # メモリ解放

    fig = tfc.create_plot_cospectra(
        key_wt,
        key_wc2h6,
        label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
        label2=r"$fC_{wC_{2}H_{6}}$ / $\overline{w^\prime C_{2}H_{6}^\prime}$",
        color2="orange",
        subplot_label="(b)",
    )

    if show_cospectra_plot:
        plt.show()  # GUIで表示する場合

    plt.close(fig)  # メモリ解放

    # 伝達関数の計算
    print("伝達関数を分析中...")
    df_wt_wch4 = tfc.process_data(reference_key=key_wt, target_key=key_wch4)
    df_wt_wc2h6 = tfc.process_data(reference_key=key_wt, target_key=key_wc2h6)
    # self.plot_ratio(df_processed, target_name, reference_name)
    a_wch4, _ = tfc.calculate_transfer_function(df_wt_wch4)
    a_wc2h6, _ = tfc.calculate_transfer_function(df_wt_wc2h6)

    fig_wch4 = tfc.create_plot_transfer_function(
        a=a_wch4,
        df_processed=df_wt_wch4,
        reference_name="wTv",
        target_name="wCH4",
    )
    fig_wc2h6 = tfc.create_plot_transfer_function(
        a=a_wc2h6,
        df_processed=df_wt_wc2h6,
        reference_name="wTv",
        target_name="wC2H6",
    )

    # a_wch4: float = tfc.analyze_transfer_function(
    #     key_wt, "wTv", key_wch4, "wCH4", show_tf_plot
    # )
    # a_wc2h6: float = tfc.analyze_transfer_function(
    #     key_wt, "wTv", key_wc2h6, "wC2H6", show_tf_plot
    # )
    print(f"wCH4の係数 a: {a_wch4}")
    print(f"wC2H6の係数 a: {a_wc2h6}")
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
