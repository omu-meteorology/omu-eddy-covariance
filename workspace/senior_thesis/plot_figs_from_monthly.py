from omu_eddy_covariance import (
    MonthlyConverter,
    MonthlyFiguresGenerator,
)

start_date, end_date = "2024-05-15", "2024-11-30"
output_dir = (
    "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/outputs"
)

if __name__ == "__main__":
    with MonthlyConverter(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/monthly",
        file_pattern="SA.Ultra.*.xlsx",
    ) as converter:
        monthly_df = converter.read_sheets(
            sheet_names=["Final"],
            start_date=start_date,
            end_date=end_date,
            include_end_date=True,
        )

        mfg = MonthlyFiguresGenerator()
        mfg.plot_monthly_c1c2_fluxes_timeseries(
            monthly_df=monthly_df, output_dir=output_dir
        )

        months_list: list[str] = ["05"]
        for month in months_list:
            # パワースペクトルのプロット
            mfg.plot_monthly_power_spectrum(
                input_dir=f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-{month}",
                output_dir=f"{output_dir}/power",
                output_filename=f"monthly_power-{month}.png",
                lag_second=10,
            )
