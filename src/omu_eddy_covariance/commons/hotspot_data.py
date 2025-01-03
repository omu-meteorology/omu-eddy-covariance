from dataclasses import dataclass
from typing import Literal


@dataclass
class HotspotData:
    """ホットスポットの情報を保持するデータクラス"""

    angle: float  # 中心からの角度
    avg_lat: float  # 平均緯度
    avg_lon: float  # 平均経度
    correlation: float  # ΔC2H6/ΔCH4相関係数
    ratio: float  # ΔC2H6/ΔCH4の比率
    section: int  # 所属する区画番号
    source: str  # データソース
    type: Literal["bio", "gas", "comb"]  # ホットスポットの種類

    def __post_init__(self):
        """
        __post_init__で各プロパティをバリデーション
        """
        # 角度は-180~180度の範囲内であることを確認
        if not -180 <= self.angle <= 180:
            raise ValueError(
                f"'angle' must be between -180 and 180 degrees: {self.angle}"
            )

        # 緯度は-90から90度の範囲内であることを確認
        if not -90 <= self.avg_lat <= 90:
            raise ValueError(
                f"'avg_lat' must be between -90 and 90 degrees: {self.avg_lat}"
            )

        # 経度は-180から180度の範囲内であることを確認
        if not -180 <= self.avg_lon <= 180:
            raise ValueError(
                f"'avg_lon' must be between -180 and 180 degrees: {self.avg_lon}"
            )

        # 相関係数は-1から1の範囲内であることを確認
        if not -1 <= self.correlation <= 1 and str(self.correlation) != "nan":
            raise ValueError(
                f"'correlation' must be between -1 and 1: {self.correlation}"
            )

        # 比率は0または正の値であることを確認
        if self.ratio < 0:
            raise ValueError(f"'ratio' must be 0 or a positive value: {self.ratio}")

        # セクション番号は0または正の整数であることを確認
        if not isinstance(self.section, int) or self.section < 0:
            raise ValueError(
                f"'section' must be a non-negative integer: {self.section}"
            )

        # データソースが空でないことを確認
        if not self.source.strip():
            raise ValueError(f"'source' must not be empty: {self.source}")
