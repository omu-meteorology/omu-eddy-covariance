import os
import pandas as pd
from omu_eddy_covariance import (
    MonthlyConverter,
    MonthlyFiguresGenerator,
)

include_end_date: bool = True
start_date, end_date = "2024-05-15", "2024-11-30"  # yyyy-MM-ddで指定
months: list[int] = [5, 6, 7, 8, 9, 10, 11]
lags_list: list[int] = [9.2, 10.0, 10.0, 10.0, 11.7, 13.2, 15.5]
# months: list[int] = [6, 7, 8, 9, 10, 11]
# lags_list: list[int] = [10.0, 10.0, 10.0, 11.7, 13.2, 15.5]
output_dir = (
    "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/outputs"
)

# フラグ
plot_timeseries: bool = False
plot_diurnals: bool = False
plot_diurnals_seasonal: bool = True
plot_scatter: bool = False
plot_spectra: bool = False

if __name__ == "__main__":
    # Ultra
    with MonthlyConverter(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/monthly",
        file_pattern="SA.Ultra.*.xlsx",
    ) as converter:
        df_ultra = converter.read_sheets(
            # sheet_names=["Final", "Final.SA"],
            # columns=["Fch4_ultra", "Fc2h6_ultra", "Fch4"],
            sheet_names=["Final"],
            columns=["Fch4_ultra", "Fc2h6_ultra", "Fch4_open"],
            start_date=start_date,
            end_date=end_date,
            include_end_date=include_end_date,
        )

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

    # print("------")
    # print(df_combined.head(10))  # DataFrameの先頭10行を表示

    mfg = MonthlyFiguresGenerator()
    mfg.plot_c1c2_fluxes_timeseries(df=df_combined, output_dir=output_dir)

    for month, lag_sec in zip(months, lags_list):
        # monthを0埋めのMM形式に変換
        month_str = f"{month:02d}"

        if plot_spectra:
            # パワースペクトルのプロット
            mfg.plot_spectra(
                input_dir=f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-{month_str}",
                output_dir=(os.path.join(output_dir, "spectra")),
                output_basename=f"spectrum-{month}",
                fs=10,
                lag_second=lag_sec,
            )

        # 月ごとのDataFrameを作成
        df_month: pd.DataFrame = MonthlyConverter.extract_monthly_data(
            df=df_combined, target_months=[month]
        )

        if plot_diurnals:
            # 日変化パターンを月ごとに作成
            mfg.plot_c1c2_fluxes_diurnal_patterns(
                df=df_month,
                y_cols_ch4=["Fch4_ultra", "Fch4_open", "Fch4_picaro"],
                y_cols_c2h6=["Fc2h6_ultra"],
                labels_ch4=["Ultra", "Open Path", "G2401"],
                labels_c2h6=["Ultra"],
                legend_only_ch4=True,
                show_label=True,
                show_legend=True,
                # show_label=False,
                # show_legend=False,
                colors_ch4=["black", "red", "blue"],
                colors_c2h6=["black"],
                output_dir=(os.path.join(output_dir, "diurnal")),
                output_filename=f"diurnal-{month_str}.png",  # タグ付けしたファイル名
            )

            mfg.plot_c1c2_fluxes_diurnal_patterns_by_date(
                df=df_month,
                y_col_ch4="Fch4_ultra",
                y_col_c2h6="Fc2h6_ultra",
                output_dir=(os.path.join(output_dir, "diurnal_by_date")),
            )

        if plot_scatter:
            mfg.plot_c1c2_fluxes_scatter(
                df=df_month,
                x_col="Fch4_open",
                y_col="Fch4_ultra",
                xlabel=r"Open Path CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                ylabel=r"Ultra CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                output_dir=(os.path.join(output_dir, "scatter")),
                output_filename=f"scatter-open_ultra-{month_str}.png",
            )
            mfg.plot_c1c2_fluxes_scatter(
                df=df_month,
                x_col="Fch4_picaro",
                y_col="Fch4_ultra",
                xlabel=r"G2401 CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                ylabel=r"Ultra CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
                output_dir=(os.path.join(output_dir, "scatter")),
                output_filename=f"scatter-g2401_ultra-{month_str}.png",
            )

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
                show_label=True,
                show_legend=True,
                # show_label=False,
                # show_legend=False,
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
                show_label=True,
                show_legend=True,
                # show_label=False,
                # show_legend=False,
                plot_holiday=False,
                output_dir=(os.path.join(output_dir, "diurnal_by_date")),
                output_filename=f"diurnal_by_date-{tag}.png",  # タグ付けしたファイル名
            )
