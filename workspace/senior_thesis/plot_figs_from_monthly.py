import os
import pandas as pd
from omu_eddy_covariance import (
    MonthlyConverter,
    MonthlyFiguresGenerator,
)

include_end_date: bool = True
start_date, end_date = "2024-05-15", "2024-11-30"
months: list[int] = [5, 6, 7, 8, 9, 10, 11]
output_dir = (
    "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/outputs"
)

if __name__ == "__main__":
    # Ultra
    with MonthlyConverter(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/monthly",
        file_pattern="SA.Ultra.*.xlsx",
    ) as converter:
        df_ultra = converter.read_sheets(
            sheet_names=["Final", "Final.SA"],
            columns=["Fch4_ultra", "Fc2h6_ultra", "Fch4"],
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

    # 両方を結合したDataFrameを明示的に作成
    df_combined: pd.DataFrame = pd.concat([df_ultra, df_picarro], ignore_index=True)

    mfg = MonthlyFiguresGenerator()
    mfg.plot_c1c2_fluxes_timeseries(df=df_combined, output_dir=output_dir)

    months_list: list[str] = ["05", "06", "07", "08", "09", "10", "11"]
    lags_list: list[int] = [9.2, 10.0, 10.0, 10.0, 11.7, 13.2, 15.5]
    for month, lag_sec in zip(months_list, lags_list):
        # パワースペクトルのプロット
        mfg.plot_spectra(
            input_dir=f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-{month}",
            output_dir=(os.path.join(output_dir, "spectra")),
            output_basename=f"spectrum-{month}",
            fs=10,
            lag_second=lag_sec,
        )

    # 日変化パターンを月ごとに作成
    for month in months:
        df_month: pd.DataFrame = MonthlyFiguresGenerator.extract_monthly_data(
            df=df_combined, target_months=[month]
        )
        # monthを0埋めのMM形式に変換
        month_str = f"{month:02d}"
        mfg.plot_c1c2_fluxes_diurnal_patterns(
            df=df_month,
            y_cols_ch4=["Fch4_ultra", "Fch4", "Fch4_picaro"],
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
            subplot_fontsize=24,
        )
