import math


def mean(arr: list[float]) -> float:
    return sum(arr) / len(arr)


def calc_cross_correlation(x_arr: list[float], y_arr: list[float]) -> float:
    x_mean = mean(x_arr)
    y_mean = mean(y_arr)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_arr, y_arr))

    x_sq_diff = sum((x - x_mean) ** 2 for x in x_arr)
    y_sq_diff = sum((y - y_mean) ** 2 for y in y_arr)
    denominator = math.sqrt(x_sq_diff * y_sq_diff)
    correlation = numerator / denominator

    return correlation
