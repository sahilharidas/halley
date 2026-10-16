"""Date arithmetic for load-time derived fields.

Time inside a model is an integer count of months (D10); dates are inputs. These helpers turn
date columns into the integers a model reads, with the conventions of the term assurance spec
(sections 4.2 and 5). They work on NumPy ``datetime64[D]`` arrays and return ``int64`` arrays.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from numpy.typing import NDArray

DateArray = NDArray[np.datetime64]
IntArray = NDArray[np.int64]


def as_dates(values: object) -> DateArray:
    """A ``datetime64[D]`` array from dates, strings or an existing array."""
    return np.asarray(values, dtype="datetime64[D]")


def year(d: DateArray) -> IntArray:
    return d.astype("datetime64[Y]").astype(np.int64) + 1970


def month(d: DateArray) -> IntArray:
    """Calendar month, 1 to 12."""
    return d.astype("datetime64[M]").astype(np.int64) % 12 + 1


def day(d: DateArray) -> IntArray:
    """Day of month, 1 to 31."""
    return (d - d.astype("datetime64[M]")).astype(np.int64) + 1


def month_index(d: DateArray) -> IntArray:
    """``12 * year + month - 1``: months since year 0."""
    return 12 * year(d) + month(d) - 1


def months_between(d1: DateArray, d2: DateArray) -> IntArray:
    """Whole months from ``d1`` to ``d2`` (term spec 4.2): count calendar months, then subtract
    one when the day of month of ``d2`` has not reached that of ``d1``."""
    return month_index(d2) - month_index(d1) - (day(d2) < day(d1)).astype(np.int64)


def age_last_birthday(birth: DateArray, on: DateArray) -> IntArray:
    """Age last birthday on a date (term spec section 5): years apart, less one when the
    (month, day) of ``on`` precedes the (month, day) of ``birth``."""
    before = (month(on) < month(birth)) | ((month(on) == month(birth)) & (day(on) < day(birth)))
    return year(on) - year(birth) - before.astype(np.int64)


def first_of_month(d: dt.date) -> bool:
    return d.day == 1


def is_month_end(d: dt.date) -> bool:
    return (d + dt.timedelta(days=1)).day == 1
