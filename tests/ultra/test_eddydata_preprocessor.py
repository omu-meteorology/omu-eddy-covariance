import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from logging import Logger, INFO, DEBUG
from omu_eddy_covariance import EddyDataPreprocessor


@pytest.fixture
def sample_df():
    """基本的なテスト用DataFrameを提供するフィクスチャ"""
    return pd.DataFrame(
        {
            "Ux": [1.0, 2.0, 3.0],
            "Uy": [1.0, 2.0, 3.0],
            "Uz": [0.5, 1.0, 1.5],
            "Tv": [20.0, 21.0, 22.0],
            "diag_sonic": [0, 0, 0],
        }
    )


@pytest.fixture
def preprocessor():
    """EddyDataPreprocessorのインスタンスを提供するフィクスチャ"""
    return EddyDataPreprocessor(fs=10.0)


def test_init():
    """初期化のテスト"""
    # デフォルトパラメータでの初期化
    processor = EddyDataPreprocessor()
    assert processor.fs == 10.0
    assert isinstance(processor.logger, Logger)

    # カスタムパラメータでの初期化
    processor_debug = EddyDataPreprocessor(fs=20.0, logging_debug=True)
    assert processor_debug.fs == 20.0
    assert processor_debug.logger.level == DEBUG


def test_add_uvw_columns(preprocessor, sample_df):
    """uvw列の追加機能のテスト"""
    result = preprocessor.add_uvw_columns(sample_df)

    # 必要な列が追加されていることを確認
    assert "wind_u" in result.columns
    assert "wind_v" in result.columns
    assert "wind_w" in result.columns
    assert "rad_wind_dir" in result.columns
    assert "rad_wind_inc" in result.columns
    assert "degree_wind_dir" in result.columns
    assert "degree_wind_inc" in result.columns

    # 値が数値であることを確認
    assert np.issubdtype(result["wind_u"].dtype, np.number)
    assert np.issubdtype(result["wind_v"].dtype, np.number)
    assert np.issubdtype(result["wind_w"].dtype, np.number)


def test_add_uvw_columns_missing_columns():
    """必要な列が欠けている場合のエラー処理テスト"""
    invalid_df = pd.DataFrame(
        {
            "Ux": [1.0, 2.0],  # Uy, Uzが欠けている
        }
    )

    with pytest.raises(ValueError):
        EddyDataPreprocessor().add_uvw_columns(invalid_df)


def test_calculate_lag_time(preprocessor, sample_df):
    """遅れ時間計算のテスト"""
    # テストデータの作成
    test_df = sample_df.copy()
    test_df["delayed_signal"] = test_df["Tv"].shift(2)  # 2サンプル分の遅れを作る

    lags = EddyDataPreprocessor._calculate_lag_time(test_df, "Tv", ["delayed_signal"])

    assert len(lags) == 1
    assert isinstance(lags[0], int)


def test_get_sorted_files():
    """ファイルソート機能のテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # テストファイルの作成
        files = ["Eddy_1.dat", "Eddy_2.dat", "Eddy_10.dat", "other.txt"]
        for f in files:
            open(os.path.join(tmpdir, f), "w").close()

        sorted_files = EddyDataPreprocessor._get_sorted_files(
            tmpdir, r"Eddy_(\d+)", ".dat"
        )

        assert len(sorted_files) == 3
        assert sorted_files == ["Eddy_1.dat", "Eddy_2.dat", "Eddy_10.dat"]


def test_wind_direction():
    """風向計算のテスト"""
    x_array = np.array([1.0, 1.0, 1.0])
    y_array = np.array([1.0, 1.0, 1.0])

    direction = EddyDataPreprocessor._wind_direction(x_array, y_array)
    assert isinstance(direction, float)
    # arctan2の実装では、風向は-π/4となる
    expected_direction = -np.pi / 4
    assert abs(direction - expected_direction) < 1e-6


def test_get_resampled_df():
    """リサンプリング機能のテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # テスト用CSVファイルの作成
        # ヘッダー行を含む正しいフォーマットのCSVを作成
        test_file = os.path.join(tmpdir, "test.dat")
        with open(test_file, "w") as f:
            # メタデータ行を作成
            f.write(
                '"TOA5","CR3000","CR3000","12345","CR3000.Std.99.03","CPU:TestEddyCov.CR3","1234","Sonic"\n'
            )
            f.write('"TIMESTAMP","RECORD","Ux","Uy","Uz","Tv"\n')  # ヘッダー行1
            f.write('"TS","RN","m/s","m/s","m/s","C"\n')  # ヘッダー行2 (単位)
            f.write('"","","Smp","Smp","Smp","Smp"\n')  # ヘッダー行3 (処理)
            # データ行
            f.write("2024-01-01 00:00:00.0,0,1.0,1.0,0.5,20.0\n")
            f.write("2024-01-01 00:00:00.1,1,2.0,2.0,1.0,21.0\n")

        processor = EddyDataPreprocessor(fs=10.0)
        df, metadata = processor.get_resampled_df(
            filepath=test_file,
            metadata_rows=4,  # メタデータの行数を4に設定
            skiprows=[0, 2, 3],  # 適切なskiprowsを設定
            numeric_columns=["Ux", "Uy", "Uz", "Tv"],
        )

        # メタデータのテスト
        assert len(metadata) == 4
        assert isinstance(df, pd.DataFrame)
        assert "TIMESTAMP" in df.columns
        assert all(col in df.columns for col in ["Ux", "Uy", "Uz", "Tv"])

        # 数値データの型チェック
        assert df["Ux"].dtype.kind in "fc"  # float or complex
        assert df["Uy"].dtype.kind in "fc"
        assert df["Uz"].dtype.kind in "fc"
        assert df["Tv"].dtype.kind in "fc"


def test_setup_logger():
    """ロガー設定のテスト"""
    logger = EddyDataPreprocessor.setup_logger(None, INFO)
    assert isinstance(logger, Logger)
    assert logger.level == INFO

    # カスタムロガーを渡した場合
    custom_logger = Logger("test")
    result_logger = EddyDataPreprocessor.setup_logger(custom_logger)
    assert result_logger == custom_logger


def test_horizontal_wind_speed():
    """水平風速計算のテスト"""
    x_array = np.array([1.0, 2.0, 3.0])
    y_array = np.array([1.0, 2.0, 3.0])
    wind_dir = np.pi / 4  # 45度

    u, v = EddyDataPreprocessor._horizontal_wind_speed(x_array, y_array, wind_dir)

    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert len(u) == len(x_array)
    assert len(v) == len(y_array)


def test_vertical_rotation():
    """鉛直回転のテスト"""
    u_array = np.array([1.0, 2.0, 3.0])
    w_array = np.array([0.5, 1.0, 1.5])
    wind_inc = np.pi / 6  # 30度

    u_rot, w_rot = EddyDataPreprocessor._vertical_rotation(u_array, w_array, wind_inc)

    assert isinstance(u_rot, np.ndarray)
    assert isinstance(w_rot, np.ndarray)
    assert len(u_rot) == len(u_array)
    assert len(w_rot) == len(w_array)


def test_wind_inclination():
    """風の迎角計算のテスト"""
    u_array = np.array([1.0, 2.0, 3.0])
    w_array = np.array([0.5, 1.0, 1.5])

    inclination = EddyDataPreprocessor._wind_inclination(u_array, w_array)
    assert isinstance(inclination, float)