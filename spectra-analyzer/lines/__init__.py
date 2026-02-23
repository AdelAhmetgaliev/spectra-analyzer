import math


def get_gaussian_intens(
    wavelength: float,
    amplitude: float,
    center: float,
    sigma: float,
    *,
    clamp: bool = True,
) -> float:
    """
    Возвращает интенсивность по гауссовому профилю:
    I = 1 - A * exp(-0.5 * ((wavelength - center) / sigma)**2)

    Параметры:
    - wavelength: значение длины волны
    - amplitude: амплитуда A (обычно >= 0)
    - center: центр профиля
    - sigma: стандартное отклонение (> 0)
    - clamp: если True, результат ограничивается в диапазоне [0.0, 1.0]

    Исключения:
    - ValueError при sigma <= 0 или амплитуде отрицательной (если это нежелательно).
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if amplitude < 0.0:
        raise ValueError("amplitude must be >= 0")

    dx: float = wavelength - center
    z: float = dx / sigma

    exponent: float = -0.5 * z * z
    value: float = 1.0 - amplitude * math.exp(exponent)

    if clamp:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0

    return value
