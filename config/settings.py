"""Explicit modelling assumptions. Returns and thresholds use decimal units."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    min_observations: int = 250
    trading_days: int = 252
    risk_window: int = 20
    var_confidence: float = 0.95
    low_risk_limit: float = 0.15
    high_risk_limit: float = 0.30
    contamination: float = 0.03

    def __post_init__(self):
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("Train and validation fractions must leave a positive test set.")
        if self.min_observations < 100 or self.risk_window < 2 or self.trading_days < 1:
            raise ValueError("Require at least 100 observations, risk window >=2 and trading days >=1.")
        if not 0 < self.var_confidence < 1:
            raise ValueError("VaR confidence must lie between 0 and 1.")
        if not 0 < self.low_risk_limit < self.high_risk_limit:
            raise ValueError("Risk thresholds must be positive and ordered.")
        if not 0 < self.contamination <= 0.5:
            raise ValueError("Isolation Forest contamination must be in (0, 0.5].")

    def to_dict(self):
        return asdict(self)
