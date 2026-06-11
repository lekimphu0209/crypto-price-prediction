"""
Interval Value Object - Immutable time interval
"""
from dataclasses import dataclass
from typing import Final
from datetime import timedelta


@dataclass(frozen=True)
class Interval:
    """Immutable time interval value object"""
    
    value: str
    
    VALID_INTERVALS = frozenset(['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'])
    
    def __post_init__(self):
        """Validate interval"""
        if self.value not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of {self.VALID_INTERVALS}")
    
    @property
    def to_timedelta(self) -> timedelta:
        """Convert interval to timedelta"""
        mapping = {
            '1m': timedelta(minutes=1),
            '3m': timedelta(minutes=3),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '6h': timedelta(hours=6),
            '8h': timedelta(hours=8),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
            '3d': timedelta(days=3),
            '1w': timedelta(weeks=1),
            '1M': timedelta(days=30),
        }
        return mapping[self.value]
    
    @property
    def minutes(self) -> int:
        """Convert interval to minutes"""
        td = self.to_timedelta
        return int(td.total_seconds() / 60)
    
    def __str__(self) -> str:
        return self.value
