"""
Symbol Value Object - Immutable trading pair symbol
"""
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Symbol:
    """Immutable trading pair symbol value object"""
    
    value: str
    
    def __post_init__(self):
        """Validate symbol format"""
        if not self.value:
            raise ValueError("Symbol cannot be empty")
        if not self.value.isupper():
            raise ValueError("Symbol must be uppercase")
        if len(self.value) < 6:
            raise ValueError("Symbol must be at least 6 characters (e.g., BTCUSDT)")
    
    @property
    def base(self) -> str:
        """Get base currency (e.g., BTC from BTCUSDT)"""
        # Common quote currencies
        quote_currencies = ['USDT', 'USD', 'EUR', 'GBP', 'BTC', 'ETH']
        for quote in quote_currencies:
            if self.value.endswith(quote):
                return self.value[:-len(quote)]
        return self.value
    
    @property
    def quote(self) -> str:
        """Get quote currency (e.g., USDT from BTCUSDT)"""
        quote_currencies = ['USDT', 'USD', 'EUR', 'GBP', 'BTC', 'ETH']
        for quote in quote_currencies:
            if self.value.endswith(quote):
                return quote
        return 'USDT'  # Default
    
    def __str__(self) -> str:
        return self.value
