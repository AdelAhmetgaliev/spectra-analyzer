import os
from typing import Self

import numpy as np

from scipy.signal import savgol_filter

from ..lines import get_gaussian_intens


# Тип для хранения единичного элемента спектра: (длина волны, интенсивность)
type SpectraPoint = tuple[float, float]


class Spectra:
    data: list[SpectraPoint]

    def __init__(self, file_path: str | os.PathLike) -> None:
        self.data: list[SpectraPoint] = []

        with open(file=file_path, mode="r", encoding="utf-8") as data_file:
            for line in data_file:
                data_tuple = tuple(map(float, line.split()))
                if len(data_tuple) == 2:
                    self.data.append(data_tuple)

    def wavelengths(self) -> list[float]:
        wavelengths_list: list[float] = []
        for data_el in self.data:
            wavelengths_list.append(data_el[0])

        return wavelengths_list

    def intensities(self) -> list[float]:
        intensities_list: list[float] = []
        for data_el in self.data:
            intensities_list.append(data_el[1])

        return intensities_list

    def unzip(self) -> tuple[list[float], list[float]]:
        wl_list: list[float] = []
        intens_list: list[float] = []
        for data_el in self.data:
            wl_list.append(data_el[0])
            intens_list.append(data_el[1])

        return wl_list, intens_list

    def normalize_intensities(self) -> Self:
        wl_list = self.wavelengths()
        intens_list = self.intensities()
        max_intens = 1.0 - min(intens_list)

        new_data: list[SpectraPoint] = []
        for i, intens in enumerate(intens_list):
            new_data.append((wl_list[i], intens / max_intens))

        norm_spectra = Spectra.__new__(Spectra)
        norm_spectra.data = new_data
        return norm_spectra  # type: ignore

    def savgol_filter(self, window_length: int, poly_order: int) -> Self:
        wl_list, intens_list = self.unzip()
        intens_filter_list = np.array(
            savgol_filter(intens_list, window_length, poly_order)
        )

        filtered_data: list[SpectraPoint] = list(zip(wl_list, intens_filter_list))
        filtered_spectra = Spectra.__new__(Spectra)
        filtered_spectra.data = filtered_data
        return filtered_spectra  # type: ignore

    def filter_by_wavelength(self, wl_min: float, wl_max: float) -> Self:
        wl_list, intens_list = self.unzip()
        new_wl_list: list[float] = []
        new_intens_list: list[float] = []

        for i, wl in enumerate(wl_list):
            if wl < wl_min or wl > wl_max:
                continue
            new_wl_list.append(wl)
            new_intens_list.append(intens_list[i])

        new_data: list[SpectraPoint] = list(zip(new_wl_list, new_intens_list))
        result_spectra = Spectra.__new__(Spectra)
        result_spectra.data = new_data
        return result_spectra  # type: ignore


def generate_gauss_spectra(
    wavelengths: list[float], amplitude: float, center: float, sigma: float
) -> Spectra:
    result_data: list[SpectraPoint] = []
    for wl in wavelengths:
        result_data.append((wl, get_gaussian_intens(wl, amplitude, center, sigma)))

    result_spectra = Spectra.__new__(Spectra)
    result_spectra.data = result_data
    return result_spectra
