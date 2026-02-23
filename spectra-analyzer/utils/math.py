import math
import numpy as np

from typing import Iterable


def mean(arr: Iterable[float]) -> float:
    a = list(arr)
    if not a:
        raise ValueError("mean() received empty iterable")
    return sum(a) / len(a)


def calc_cross_correlation(x_arr: Iterable[float], y_arr: Iterable[float]) -> float:
    if isinstance(x_arr, np.ndarray) or isinstance(y_arr, np.ndarray):
        x = np.asarray(x_arr, dtype=float)
        y = np.asarray(y_arr, dtype=float)
        if x.shape != y.shape or x.size == 0:
            return 0.0
        x_mean = x.mean()
        y_mean = y.mean()
        x_diff = x - x_mean
        y_diff = y - y_mean
        denom = math.sqrt(float((x_diff * x_diff).sum() * (y_diff * y_diff).sum()))
        if denom == 0.0:
            return 0.0
        return float((x_diff * y_diff).sum() / denom)

    x_list = list(x_arr)
    y_list = list(y_arr)
    n = len(x_list)
    if n == 0 or n != len(y_list):
        return 0.0

    sum_x = sum(x_list)
    sum_y = sum(y_list)
    x_mean = sum_x / n
    y_mean = sum_y / n

    num = 0.0
    sum_x_sq_diff = 0.0
    sum_y_sq_diff = 0.0
    for x, y in zip(x_list, y_list):
        dx = x - x_mean
        dy = y - y_mean
        num += dx * dy
        sum_x_sq_diff += dx * dx
        sum_y_sq_diff += dy * dy

    denom = math.sqrt(sum_x_sq_diff * sum_y_sq_diff)
    if denom == 0.0:
        return 0.0

    return num / denom
