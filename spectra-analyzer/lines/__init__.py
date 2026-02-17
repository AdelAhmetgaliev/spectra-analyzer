import math


def get_gaussian_intens(
    wavelength: float, amplitude: float, center: float, sigma: float
) -> float:
    return 1.0 - amplitude * math.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
