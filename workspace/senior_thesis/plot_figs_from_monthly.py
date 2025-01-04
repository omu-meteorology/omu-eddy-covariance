import os
import re  # 正規表現を使用するためのインポート
import glob
import numpy as np
import pandas as pd
from omu_eddy_covariance import (
    MonthlyConverter,
    MonthlyFiguresGenerator,
    EddyDataPreprocessor,
)
from tqdm import tqdm  # プログレスバー用
import matplotlib.font_manager as fm

# フォントファイルを登録
font_paths: list[str] = [
    "/home/connect0459/.local/share/fonts/arial.ttf",  # 英語のデフォルト
    "/home/connect0459/.local/share/fonts/msgothic.ttc",  # 日本語のデフォルト
]
for path in font_paths:
    fm.fontManager.addfont(path)
# プロットの書式を設定
MonthlyFiguresGenerator.setup_plot_params(
    font_family=["Arial", "MS Gothic"], font_size=24, tick_size=24
)

include_end_date: bool = True
start_date, end_date = "2024-05-15", "2024-11-30"  # yyyy-MM-ddで指定
months_two: list[str] = [
    "05_06",
    "07_08",
    "09_10",
    "11_12",
]
months: list[int] = [5, 6, 7, 8, 9, 10, 11]
subplot_labels: list[list[str | None]] = [
    # ["(a1)", "(a2)"],
    # ["(b1)", "(b2)"],
    # ["(c1)", "(c2)"],
    # ["(d1)", "(d2)"],
    # ["(e1)", "(e2)"],
    # ["(f1)", "(f2)"],
    # ["(g1)", "(g2)"],
    ["(a)", None],
    ["(b)", None],
    ["(c)", None],
    ["(d)", None],
    ["(e)", None],
    ["(f)", None],
    ["(g)", None],
]
lags_list: list[int] = [9.2, 10.0, 10.0, 10.0, 11.7, 13.2, 15.5]
output_dir = (
    "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/outputs"
)

# フラグ
plot_turbulences: bool = False
plot_spectra: bool = False
plot_spectra_two: bool = False
plot_timeseries: bool = False
plot_diurnals: bool = False
plot_seasonal: bool = False
diurnal_subplot_fontsize: float = 36
plot_scatter: bool = True
plot_sources: bool = False

if __name__ == "__main__":
    # Ultra
    with MonthlyConverter(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/monthly",
        file_pattern="SA.Ultra.*.xlsx",
    ) as converter:
        df_ultra = converter.read_sheets(
            # sheet_names=["Final", "Final.SA"],
            # columns=["Fch4_ultra", "Fc2h6_ultra", "Fch4"],
            sheet_names=["Final", "Final.SA"],
            columns=[
                "Fch4_ultra",
                "Fc2h6_ultra",
                "CH4_ultra",
                "C2H6_ultra",
                "Fch4_open",
                "slope",
                "intercept",
                "r_value",
                "p_value",
                "stderr",
                "RSSI",
                "Wind direction_x",
                "WS vector_x",
            ],
            start_date=start_date,
            end_date=end_date,
            include_end_date=include_end_date,
        )
        df_ultra["Wind direction"] = df_ultra["Wind direction_x"]
        df_ultra["WS vector"] = df_ultra["WS vector_x"]

    # Picarro
    with MonthlyConverter(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/monthly",
        file_pattern="SA.Picaro.*.xlsx",
    ) as converter:
        df_picarro = converter.read_sheets(
            sheet_names=["Final"],
            columns=["Fch4_picaro"],
            start_date=start_date,
            end_date=end_date,
            include_end_date=include_end_date,
        )
        # print(df_picarro.head(10))

    # 両方を結合したDataFrameを明示的に作成
    df_combined = MonthlyConverter.merge_dataframes(df1=df_ultra, df2=df_picarro)

    # 濃度データはキャリブレーション後から使用
    # Dateカラムをインデックスに設定
    df_combined["Date"] = pd.to_datetime(df_combined["Date"])
    df_combined["Date_index"] = df_combined["Date"]
    df_combined.set_index("Date_index", inplace=True)

    # 濃度データのみを2023-09-11以降に制限（年を2023に修正）
    filter_date = pd.to_datetime("2024-09-11")
    mask = df_combined.index < filter_date
    df_combined.loc[mask, "CH4_ultra"] = np.nan
    df_combined.loc[mask, "C2H6_ultra"] = np.nan

    # RSSIが40未満のデータは信頼性が低いため、Fch4_openをnanに置換
    df_combined.loc[df_combined["RSSI"] < 40, "Fch4_open"] = np.nan

    # CH4_ultraをppb単位に直したカラムを作成
    df_combined["CH4_ultra_ppb"] = df_combined["CH4_ultra"] * 1000

    # print("------")
    # print(df_combined.head(10))  # DataFrameの先頭10行を表示

    mfg = MonthlyFiguresGenerator()

    if plot_timeseries:
        df_trimed = df_combined.copy()

        # Dateカラムをインデックスに設定
        df_trimed["Date"] = pd.to_datetime(df_trimed["Date"])
        df_trimed["Date_index"] = df_trimed["Date"]
        df_trimed.set_index("Date_index", inplace=True)

        # データの日付範囲を確認
        print("データの開始日:", df_trimed.index.min())
        print("データの終了日:", df_trimed.index.max())

        # 濃度データのみを2024-09-11以降に制限
        filter_date = pd.to_datetime("2024-09-11")
        df_trimed.loc[df_trimed.index < filter_date, "CH4_ultra"] = np.nan
        df_trimed.loc[df_trimed.index < filter_date, "C2H6_ultra"] = np.nan

        print("\nフィルタリング後のデータ件数:", len(df_trimed))
        print("\nフィルタリング後の最初の10件:")
        print(df_trimed.head(10))

        mfg.plot_c1c2_concentrations_and_fluxes_timeseries(
            df=df_trimed,
            ch4_conc_key="CH4_ultra",
            ch4_flux_key="Fch4_ultra",
            c2h6_conc_key="C2H6_ultra",
            c2h6_flux_key="Fc2h6_ultra",
            output_dir=(os.path.join(output_dir, "timeseries")),
        )
        mfg.logger.info("'timeseries'を作成しました。")

    if plot_turbulences:
        # データディレクトリのパスを定義
        # target_tag: str = "0605_0608"
        # data_dir = f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-{target_tag}"
        target_tag: str = "1008_1012"
        data_dir = f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-{target_tag}"

        # ディレクトリ内の全てのCSVファイルを取得
        filepaths = glob.glob(os.path.join(data_dir, "*-resampled.csv"))

        # 各ファイルに対して処理を実行
        for filepath in tqdm(filepaths, desc="乱流データの処理"):
            # ファイル名から日時を抽出
            filename = os.path.basename(filepath)
            try:
                # ファイル名をアンダースコアで分割し、日時部分を取得
                parts = filename.split("_")
                # 年、月、日、時刻の部分を見つける
                for i, part in enumerate(parts):
                    if part == "2024":  # 年を見つけたら、そこから4つの要素を取得
                        date = "_".join(
                            [
                                parts[i],  # 年
                                parts[i + 1],  # 月
                                parts[i + 2],  # 日
                                re.sub(
                                    r"(\+|-resampled\.csv)", "", parts[i + 3]
                                ),  # 時刻から+と-resampled.csvを削除
                            ]
                        )
                        break

                # データの読み込みと処理
                edp = EddyDataPreprocessor(10)
                df_for_turb, _ = edp.get_resampled_df(
                    filepath=filepath, is_already_resampled=True
                )
                df_for_turb = edp.add_uvw_columns(df_for_turb)

                # 図の作成と保存
                mfg.plot_turbulence(
                    df=df_for_turb,
                    uz_key="wind_w",
                    output_dir=(os.path.join(output_dir, "turbulences", target_tag)),
                    output_filename=f"turbulence-{date}.png",
                    add_serial_labels=False,
                )
                # mfg.logger.info(f"'{date}'の'turbulences'を作成しました。")

            except (IndexError, ValueError):
                # except (IndexError, ValueError) as e:
                # mfg.logger.warning(
                #     f"ファイル名'{filename}'から日時を抽出できませんでした: {e}"
                # )
                continue

    if plot_spectra_two:
        for month in months_two:
            month_str = month
            mfg.logger.info(f"{month_str}の処理を開始します。")

            # パワースペクトルのプロット
            mfg.plot_spectra(
                input_dir=f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-two-{month_str}",
                output_dir=(os.path.join(output_dir, "spectra", "two")),
                output_basename=f"spectrum-two-{month}",
                fs=10,
                lag_second=10,
                plot_co=False,
            )
            mfg.logger.info("'spectra_two'を作成しました。")

    for month, lag_sec, subplot_label in zip(months, lags_list, subplot_labels):
        # monthを0埋めのMM形式に変換
        month_str = f"{month:02d}"
        mfg.logger.info(f"{month_str}の処理を開始します。")

        if plot_spectra:
            # パワースペクトルのプロット
            mfg.plot_spectra(
                input_dir=f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-{month_str}",
                output_dir=(os.path.join(output_dir, "spectra")),
                output_basename=f"spectrum-{month}",
                fs=10,
                lag_second=lag_sec,
            )
            mfg.logger.info("'spectra'を作成しました。")

        # 月ごとのDataFrameを作成
        df_month: pd.DataFrame = MonthlyConverter.extract_monthly_data(
            df=df_combined, target_months=[month]
        )
        if month == 11:
            df_month["Fch4_open"] = np.nan

        if plot_diurnals:
            # 日変化パターンを月ごとに作成
            mfg.plot_c1c2_fluxes_diurnal_patterns(
                df=df_month,
                y_cols_ch4=["Fch4_ultra", "Fch4_open", "Fch4_picaro"],
                y_cols_c2h6=["Fc2h6_ultra"],
                labels_ch4=["Ultra", "Open Path", "G2401"],
                labels_c2h6=["Ultra"],
                legend_only_ch4=True,
                # show_label=True,
                # show_legend=True,
                show_label=False,
                show_legend=False,
                subplot_fontsize=diurnal_subplot_fontsize,
                subplot_label_ch4=subplot_label[0],
                subplot_label_c2h6=subplot_label[1],
                colors_ch4=["red", "black", "blue"],
                colors_c2h6=["orange"],
                output_dir=(os.path.join(output_dir, "diurnal")),
                output_filename=f"diurnal-{month_str}.png",  # タグ付けしたファイル名
            )
            if month == 11:
                mfg.plot_c1c2_fluxes_diurnal_patterns(
                    df=df_month,
                    y_cols_ch4=["Fch4_ultra", "Fch4_open", "Fch4_picaro"],
                    y_cols_c2h6=["Fc2h6_ultra"],
                    labels_ch4=[r"Ultra CH$_4$", "Open Path", "G2401"],
                    labels_c2h6=[r"Ultra C$_2$H$_6$"],
                    legend_only_ch4=False,
                    show_label=False,
                    show_legend=True,
                    subplot_label_ch4=subplot_label[0],
                    subplot_label_c2h6=subplot_label[1],
                    colors_ch4=["red", "black", "blue"],
                    colors_c2h6=["orange"],
                    output_dir=(os.path.join(output_dir, "diurnal")),
                    output_filename="diurnal-legend.png",  # タグ付けしたファイル名
                )

            mfg.plot_c1c2_fluxes_diurnal_patterns_by_date(
                df=df_month,
                y_col_ch4="Fch4_ultra",
                y_col_c2h6="Fc2h6_ultra",
                plot_holiday=False,
                show_label=False,
                show_legend=False,
                subplot_fontsize=diurnal_subplot_fontsize,
                subplot_label_ch4=subplot_label[0],
                subplot_label_c2h6=subplot_label[1],
                output_dir=(os.path.join(output_dir, "diurnal_by_date")),
                output_filename=f"diurnal_by_date-{month_str}.png",
            )
            mfg.logger.info("'diurnals'を作成しました。")

        if plot_scatter:
            # c1c2 conc
            try:
                mfg.plot_scatter(
                    df=df_month,
                    x_col="CH4_ultra_ppb",
                    y_col="C2H6_ultra",
                    xlabel=r"CH$_4$ Concentration (ppb)",
                    ylabel=r"C$_2$H$_6$ Concentration (ppb)",
                    # x_col="CH4_ultra",
                    # y_col="C2H6_ultra",
                    # xlabel=r"CH$_4$ Concentration (ppm)",
                    # ylabel=r"C$_2$H$_6$ Concentration (ppb)",
                    output_dir=(os.path.join(output_dir, "scatter")),
                    output_filename=f"scatter-ultra_c1c2_c-{month_str}.png",
                    # x_axis_range=(1.8, 2.6),
                    # y_axis_range=(-21, 0),
                    # x_axis_range=None,
                    # y_axis_range=None,
                    show_fixed_slope=True,
                    # fixed_slope=0.076 * 1000,
                    fixed_slope=0.076,
                )
            except Exception as e:
                print(e)

            # c1c2 flux
            mfg.plot_scatter(
                df=df_month,
                x_col="Fch4_ultra",
                y_col="Fc2h6_ultra",
                xlabel=r"CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                ylabel=r"C$_2$H$_6$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                output_dir=(os.path.join(output_dir, "scatter")),
                output_filename=f"scatter-ultra_c1c2_f-{month_str}.png",
                x_axis_range=(-50, 400),
                y_axis_range=(-5, 25),
                show_fixed_slope=True,
            )
            try:
                # open_ultra
                mfg.plot_scatter(
                    df=df_month,
                    x_col="Fch4_open",
                    y_col="Fch4_ultra",
                    xlabel=r"Open Path CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                    ylabel=r"Ultra CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                    output_dir=(os.path.join(output_dir, "scatter")),
                    output_filename=f"scatter-open_ultra-{month_str}.png",
                    x_axis_range=(-50, 200),
                    y_axis_range=(-50, 200),
                )
            except Exception as e:
                print(e)

            # g2401_ultra
            mfg.plot_scatter(
                df=df_month,
                x_col="Fch4_picaro",
                y_col="Fch4_ultra",
                xlabel=r"G2401 CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                ylabel=r"Ultra CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                output_dir=(os.path.join(output_dir, "scatter")),
                output_filename=f"scatter-g2401_ultra-{month_str}.png",
                x_axis_range=(-50, 200),
                y_axis_range=(-50, 200),
            )
            mfg.logger.info("'scatters'を作成しました。")

    if plot_seasonal:
        seasons: list[list[int]] = [[6, 7, 8], [9, 10, 11]]
        seasons_tags: list[str] = ["summer", "fall"]
        for season, tag in zip(seasons, seasons_tags):
            # 月ごとのDataFrameを作成
            df_season: pd.DataFrame = MonthlyConverter.extract_monthly_data(
                df=df_combined, target_months=season
            )

            # 日変化パターンを月ごとに作成
            mfg.plot_c1c2_fluxes_diurnal_patterns(
                df=df_season,
                y_cols_ch4=["Fch4_ultra", "Fch4_open", "Fch4_picaro"],
                y_cols_c2h6=["Fc2h6_ultra"],
                labels_ch4=["Ultra", "Open Path", "G2401"],
                labels_c2h6=["Ultra"],
                legend_only_ch4=True,
                # show_label=True,
                # show_legend=True,
                show_label=False,
                show_legend=False,
                subplot_fontsize=diurnal_subplot_fontsize,
                colors_ch4=["black", "red", "blue"],
                colors_c2h6=["black"],
                output_dir=(os.path.join(output_dir, "diurnal")),
                output_filename=f"diurnal-{tag}.png",  # タグ付けしたファイル名
            )

            mfg.plot_c1c2_fluxes_diurnal_patterns_by_date(
                df=df_season,
                y_col_ch4="Fch4_ultra",
                y_col_c2h6="Fc2h6_ultra",
                legend_only_ch4=True,
                # show_label=True,
                # show_legend=True,
                show_label=False,
                show_legend=False,
                subplot_fontsize=diurnal_subplot_fontsize,
                plot_holiday=False,
                output_dir=(os.path.join(output_dir, "diurnal_by_date")),
                output_filename=f"diurnal_by_date-{tag}.png",  # タグ付けしたファイル名
            )
            mfg.logger.info("'diurnals-seasons'を作成しました。")

            mfg.plot_source_contributions_diurnal(
                df=df_season,
                output_dir=(os.path.join(output_dir, "tests")),
                output_filename=f"source_contributions-{tag}.png",
                ch4_flux_key="Fch4_ultra",
                c2h6_flux_key="Fc2h6_ultra",
                y_max=90,
            )

    # mfg.plot_flux_diurnal_patterns_with_std(
    #     df=df_combined,
    #     output_dir=(os.path.join(output_dir, "tests")),
    #     ch4_flux_key="Fch4_ultra",
    #     c2h6_flux_key="Fc2h6_ultra",
    # )

    if plot_sources:
        mfg.plot_source_contributions_diurnal(
            df=df_combined,
            output_dir=(os.path.join(output_dir, "tests")),
            ch4_flux_key="Fch4_ultra",
            c2h6_flux_key="Fc2h6_ultra",
            y_max=90,
        )

        mfg.plot_wind_rose_sources(
            df=df_combined,
            output_dir=(os.path.join(output_dir, "tests")),
            ch4_flux_key="Fch4_ultra",
            c2h6_flux_key="Fc2h6_ultra",
        )
