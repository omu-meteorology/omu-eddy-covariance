import pytest
from omu_eddy_covariance import __version__


def test_version():
    # バージョン番号が正しいことを確認
    assert __version__ == "0.1.0"
