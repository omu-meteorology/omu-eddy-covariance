from omu_eddy_covariance import MonthlyConverter

# with文を使用した基本的な使い方
with MonthlyConverter(
    "/home/connect0459/labo/omu-eddy-covariance/workspace/ultra/private/data/monthly"
) as converter:
    # 利用可能な日付の確認
    dates = converter.get_available_dates()
    print(f"利用可能な日付: {dates}")

    # 特定の期間のデータを読み込む
    df = converter.read_sheets(
        sheet_names=["Final.SA", "Final"], start_date="2024.06", end_date="2024.09"
    )

    print(df)

    # numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    # monthly_summary = df[numeric_columns].groupby(["year", "month"]).sum()
    # print(monthly_summary)
