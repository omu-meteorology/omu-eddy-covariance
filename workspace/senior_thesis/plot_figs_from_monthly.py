import os
import numpy as np
import pandas as pd
from omu_eddy_covariance import (
    MonthlyConverter,
    MonthlyFiguresGenerator,
    EddyDataPreprocessor,
)

include_end_date: bool = True
start_date, end_date = "2024-05-15", "2024-11-30"  # yyyy-MM-ddで指定
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
plot_timeseries: bool = False
plot_diurnals: bool = False
plot_diurnals_seasonal: bool = False
diurnal_subplot_fontsize: float = 36
plot_scatter: bool = False

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

    # RSSIが40未満のデータは信頼性が低いため、Fch4_openをnanに置換
    df_combined.loc[df_combined["RSSI"] < 40, "Fch4_open"] = np.nan
    df_combined["CH4_ultra_ppb"] = df_combined["CH4_ultra"] * 1000

    # print("------")
    # print(df_combined.head(10))  # DataFrameの先頭10行を表示

    mfg = MonthlyFiguresGenerator()
    MonthlyFiguresGenerator.setup_plot_params(
        font_family=["IPAGothic", "Dejavu Sans"], font_size=24, tick_size=24
    )

    if plot_timeseries:
        mfg.plot_c1c2_fluxes_timeseries(
            df=df_combined, output_dir=(os.path.join(output_dir, "timeseries"))
        )
        mfg.plot_c1c2_concentrations_and_fluxes_timeseries(
            df=df_combined,
            output_dir=(os.path.join(output_dir, "timeseries")),
        )
        mfg.logger.info("'timeseries'を作成しました。")

    if plot_turbulences:
        filename: str = "TOA5_37477.SAC_Ultra.Eddy_3_2024_06_01_1400-resampled.csv"
        filepath: str = f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-06/{filename}"
        edp = EddyDataPreprocessor(10)
        df_for_turb, _ = edp.get_resampled_df(
            filepath=filepath, is_already_resampled=True
        )
        df_for_turb = edp.add_uvw_columns(df_for_turb)
        mfg.plot_turbulence(
            df=df_for_turb,
            uz_key="wind_w",
            output_dir=(os.path.join(output_dir, "turbulences")),
        )
        mfg.logger.info("'turbulences'を作成しました。")

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
        if month == 10 or month == 11:
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
            # 濃度の変動を計算
            df_month["CH4_ultra_fluc"] = (
                df_month["CH4_ultra_ppb"] - df_month["CH4_ultra_ppb"].mean()
            )
            df_month["C2H6_ultra_fluc"] = (
                df_month["C2H6_ultra"] - df_month["C2H6_ultra"].mean()
            )

            # # conc
            # mfg.plot_scatter(
            #     df=df_month,
            #     x_col="CH4_ultra_ppb",
            #     y_col="C2H6_ultra",
            #     xlabel=r"Ultra CH$_4$ Concentration (ppb)",
            #     ylabel=r"Ultra C$_2$H$_6$ Concentration (ppb)",
            #     output_dir=(os.path.join(output_dir, "scatter")),
            #     output_filename=f"scatter-ultra_conc-{month_str}.png",
            #     x_axis_range=None,
            #     y_axis_range=None,
            #     show_fixed_slope=True,
            # )
            # 濃度変動の散布図
            mfg.plot_scatter(
                df=df_month,
                x_col="CH4_ultra_fluc",
                y_col="C2H6_ultra_fluc",
                # xlabel=r"$\Delta$CH$_4$ (ppb)",
                # ylabel=r"$\Delta$C$_2$H$_6$ (ppb)",
                output_dir=(os.path.join(output_dir, "scatter")),
                output_filename=f"scatter-ultra_conc_fluc-{month_str}.png",
                x_axis_range=None,
                y_axis_range=None,
                show_fixed_slope=True,
            )

            # c1c2
            mfg.plot_scatter(
                df=df_month,
                x_col="Fch4_ultra",
                y_col="Fc2h6_ultra",
                # xlabel=r"CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                # ylabel=r"C$_2$H$_6$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                output_dir=(os.path.join(output_dir, "scatter")),
                output_filename=f"scatter-ultra_c1c2-{month_str}.png",
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
                    show_label=False,
                    output_dir=(os.path.join(output_dir, "scatter")),
                    output_filename=f"scatter-open_ultra-{month_str}.png",
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
            )
            mfg.logger.info("'scatters'を作成しました。")

    if plot_diurnals_seasonal:
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

    mfg.plot_flux_diurnal_patterns_with_std(
        df=df_combined,
        output_dir=(os.path.join(output_dir, "tests")),
        ch4_flux_key="Fch4_ultra",
        c2h6_flux_key="Fc2h6_ultra",
    )

    mfg.plot_source_contributions_diurnal(
        df=df_combined,
        output_dir=(os.path.join(output_dir, "tests")),
        ch4_flux_key="Fch4_ultra",
        c2h6_flux_key="Fc2h6_ultra",
    )

    mfg.plot_wind_rose_sources(
        df=df_combined,
        output_dir=(os.path.join(output_dir, "tests")),
        ch4_flux_key="Fch4_ultra",
        c2h6_flux_key="Fc2h6_ultra",
    )
