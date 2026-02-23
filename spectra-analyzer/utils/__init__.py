import numpy as np

from ..utils.math import calc_cross_correlation
from ..models.spectra import Spectra


SPEED_OF_LIGHT: float = 299_792_458.0  # m/s


def get_max_corr_wl(line_spec: Spectra) -> float:
    wl_list, intens_list = line_spec.unzip()
    if not wl_list:
        raise ValueError("Input spectrum is empty")

    wl_arr = np.asarray(wl_list, dtype=float)
    intens_arr = np.asarray(intens_list, dtype=float)

    ampl = float(1.0 - intens_arr.min())
    sigma = 1.0

    result_wl: float = float(wl_arr[0])
    max_corr: float = -np.inf

    for c in wl_arr:
        z = (wl_arr - c) / sigma
        templ = 1.0 - ampl * np.exp(-0.5 * z * z)
        cross_corr = calc_cross_correlation(intens_arr, templ)
        if cross_corr > max_corr:
            max_corr = float(cross_corr)
            result_wl = float(c)

    return result_wl


def calculate_radial_velocity(line_spec: Spectra, lab_wl: float) -> float:
    line_wl = get_max_corr_wl(line_spec)
    return SPEED_OF_LIGHT * (line_wl - float(lab_wl)) / float(lab_wl)
