import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmissionData:
    """
    ホットスポットの排出量データを格納するクラス。

    Attributes:
        source (str): データソース
        type (str): ホットスポットの種類（"bio", "gas", "comb"）
        section (Optional[str]): セクション情報
        latitude (float): 緯度
        longitude (float): 経度
        delta_ch4 (float): CH4の増加量 (ppm)
        delta_c2h6 (float): C2H6の増加量 (ppb)
        ratio (float): C2H6/CH4比
        emission_rate (float): 排出量 (L/min)
        daily_emission (float): 日排出量 (L/day)
        annual_emission (float): 年間排出量 (L/year)
    """

    source: str
    type: str
    section: Optional[str]
    latitude: float
    longitude: float
    delta_ch4: float
    delta_c2h6: float
    ratio: float
    emission_rate: float
    daily_emission: float
    annual_emission: float

    def __post_init__(self) -> None:
        """
        Initialize時のバリデーションを行います。

        Raises:
            ValueError: 入力値が不正な場合
        """
        # sourceのバリデーション
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Source must be a non-empty string")

        # typeのバリデーション
        valid_types = {"bio", "gas", "comb"}
        if self.type not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")

        # sectionのバリデーション（Noneは許可）
        if self.section is not None and not isinstance(self.section, str):
            raise ValueError("Section must be a string or None")

        # 緯度のバリデーション
        if (
            not isinstance(self.latitude, (int, float))
            or not -90 <= self.latitude <= 90
        ):
            raise ValueError("Latitude must be a number between -90 and 90")

        # 経度のバリデーション
        if (
            not isinstance(self.longitude, (int, float))
            or not -180 <= self.longitude <= 180
        ):
            raise ValueError("Longitude must be a number between -180 and 180")

        # delta_ch4のバリデーション
        if not isinstance(self.delta_ch4, (int, float)) or self.delta_ch4 < 0:
            raise ValueError("Delta CH4 must be a non-negative number")

        # delta_c2h6のバリデーション
        if not isinstance(self.delta_c2h6, (int, float)) or self.delta_c2h6 < 0:
            raise ValueError("Delta C2H6 must be a non-negative number")

        # ratioのバリデーション
        if not isinstance(self.ratio, (int, float)) or self.ratio < 0:
            raise ValueError("Ratio must be a non-negative number")

        # emission_rateのバリデーション
        if not isinstance(self.emission_rate, (int, float)) or self.emission_rate < 0:
            raise ValueError("Emission rate must be a non-negative number")

        # daily_emissionのバリデーション
        expected_daily = self.emission_rate * 60 * 24
        if not math.isclose(self.daily_emission, expected_daily, rel_tol=1e-10):
            raise ValueError(
                f"Daily emission ({self.daily_emission}) does not match "
                f"calculated value from emission rate ({expected_daily})"
            )

        # annual_emissionのバリデーション
        expected_annual = self.daily_emission * 365
        if not math.isclose(self.annual_emission, expected_annual, rel_tol=1e-10):
            raise ValueError(
                f"Annual emission ({self.annual_emission}) does not match "
                f"calculated value from daily emission ({expected_annual})"
            )

        # NaN値のチェック
        numeric_fields = [
            self.latitude,
            self.longitude,
            self.delta_ch4,
            self.delta_c2h6,
            self.ratio,
            self.emission_rate,
            self.daily_emission,
            self.annual_emission,
        ]
        if any(math.isnan(x) for x in numeric_fields):
            raise ValueError("Numeric fields cannot contain NaN values")

    def to_dict(self) -> dict:
        """
        データクラスの内容を辞書形式に変換します。

        Returns:
            dict: データクラスの属性と値を含む辞書
        """
        return {
            "source": self.source,
            "type": self.type,
            "section": self.section,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "delta_ch4": self.delta_ch4,
            "delta_c2h6": self.delta_c2h6,
            "ratio": self.ratio,
            "emission_rate": self.emission_rate,
            "daily_emission": self.daily_emission,
            "annual_emission": self.annual_emission,
        }
