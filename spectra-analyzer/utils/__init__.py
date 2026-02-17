from ..utils.math import calc_cross_correlation
from ..models.spectra import Spectra, generate_gauss_spectra


def get_max_corr_wl(line_spec: Spectra) -> float:
    wl_list, intens_list = line_spec.unzip()

    ampl, sigma = 1.0 - min(intens_list), 1.0
    result_wl, max_corr = 0.0, 0.0

    for wl in wl_list:
        templ_spec = generate_gauss_spectra(wl_list, ampl, wl, sigma)
        cross_corr = calc_cross_correlation(intens_list, templ_spec.intensities())

        if max_corr < cross_corr:
            max_corr = cross_corr
            result_wl = wl

    return result_wl
