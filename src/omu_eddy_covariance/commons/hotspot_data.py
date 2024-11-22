from dataclasses import dataclass


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
    type: str  # ホットスポットの種類 ("bio", "gas", or "comb")
