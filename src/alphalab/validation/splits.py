"""Walk-forward splits with an embargo between training and test windows.

Daily signals use look-back windows, and with delay 1 a position decided on the last training day is
held into the test window. An embargo of `embargo` trading days at the start of each test window keeps
returns that depend on training-period decisions out of the test score (the "purge"/embargo idea of
Lopez de Prado, simplified for non-overlapping daily labels).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def walk_forward(dates: list[date], first_test_year: int, last_test_year: int, embargo: int = 5,
                 expanding: bool = True, train_years: int = 3) -> list[Fold]:
    """One fold per calendar test year. Train = all earlier data (expanding) or the last `train_years` years."""
    ds = sorted(set(dates))
    folds = []
    for y in range(first_test_year, last_test_year + 1):
        test = [d for d in ds if d.year == y]
        if not test:
            continue
        train = [d for d in ds if d.year < y and (expanding or d.year >= y - train_years)]
        if not train:
            continue
        test = test[embargo:] if len(test) > embargo else test
        folds.append(Fold(train[0], train[-1], test[0], test[-1]))
    return folds
