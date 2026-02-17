import os
# from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from . import DATA_DIR
from .models.spectra import Spectra, generate_gauss_spectra
from .models.observation import Observation
from .utils import get_max_corr_wl
from .utils.obs_reader import read_observations_from_file


FILTER_WINDOW_LENGTH: int = 11
FILTER_POLY_ORDER: int = 3


def plot_spectrum(observation: Observation, spectra: Spectra) -> None:
    wl_list, intens_list = spectra.unzip()

    _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wl_list, intens_list, color="blue", linewidth=2)
    ax.set_title(
        f"Спектральные данные, MJD={observation.mjd}", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Длина волны (A)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Интенсивность", fontsize=14, fontweight="bold")

    ax.set_xlim(min(wl_list), max(wl_list))
    ax.set_ylim(min(intens_list) * 0.8, max(intens_list) * 1.1)

    ax.legend(["Спектр"], loc="lower right", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)

    plt.show()


def plot_spectrum_and_filtered(
    observation: Observation, spectra: Spectra, spectra_filtered: Spectra
) -> None:
    wl_list, intens_list = spectra.unzip()
    intens_filtered_list = spectra_filtered.intensities()

    _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wl_list, intens_list, color="blue", linewidth=2)
    ax.plot(wl_list, intens_filtered_list, color="green", linewidth=2)
    ax.set_title(
        f"Спектральные данные, MJD={observation.mjd}", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Длина волны (A)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Интенсивность", fontsize=14, fontweight="bold")

    ax.set_xlim(min(wl_list), max(wl_list))
    ax.set_ylim(min(intens_list) * 0.8, max(intens_list) * 1.1)

    ax.legend(
        ["Спектр исходный", "Спектр отфильтрованный"], loc="lower right", fontsize=12
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)

    plt.show()


def plot_lines(line_spectra: Spectra, line_spectra_generated: Spectra) -> None:
    wl_list, intens_list = line_spectra.unzip()
    intens_list = 1.0 - np.array(intens_list)
    intens_generated_list = 1.0 - np.array(line_spectra_generated.intensities())

    _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wl_list, intens_list, color="blue", linewidth=2)
    ax.plot(wl_list, intens_generated_list, color="red", linewidth=2)
    ax.set_title("Сравнение линий", fontsize=16, fontweight="bold")
    ax.set_xlabel("Длина волны (A)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Интенсивность", fontsize=14, fontweight="bold")

    ax.set_xlim(min(wl_list), max(wl_list))
    ax.set_ylim(0, max(intens_list) * 1.1)

    ax.legend(
        ["Линия исходная", "Линия сгенерированная"], loc="upper right", fontsize=12
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)

    plt.show()


def calculate_radial_velocity(line_spec: Spectra, lab_wl: float) -> float:
    SPEED_OF_LIGHT = 299792458  # м/с

    line_wl = get_max_corr_wl(line_spec)

    return SPEED_OF_LIGHT * (line_wl - lab_wl) / lab_wl


def main() -> None:
    lab_wl = 4101.734
    path_to_obsdat = os.path.join(DATA_DIR, "ObsDat.txt")
    observation_list = read_observations_from_file(path_to_obsdat)
    observation_list.sort(key=lambda obs: obs.mjd)

    spectra_list: list[Spectra] = []
    for obs in observation_list:
        spec = Spectra(obs.file_path)
        spectra_list.append(spec)

    plot_spectrum_and_filtered(
        observation_list[0],
        spectra_list[0],
        spectra_list[0].savgol_filter(FILTER_WINDOW_LENGTH, FILTER_POLY_ORDER),
    )

    rad_vel_list: list[float] = []
    for spec in spectra_list:
        filtered_spectra = spec.savgol_filter(FILTER_WINDOW_LENGTH, FILTER_POLY_ORDER)
        hi_line_spectra = filtered_spectra.filter_by_wavelength(4092, 4110)

        # ###################
        # center_wl = get_max_corr_wl(hi_line_spectra)
        # ampl, sigma = 1.0 - min(hi_line_spectra.intensities()), 1.0
        # templ_spec = generate_gauss_spectra(hi_line_spectra.wavelengths(), ampl, center_wl, sigma)
        # plot_lines(hi_line_spectra, templ_spec)
        # ###################

        rad_vel = calculate_radial_velocity(hi_line_spectra, lab_wl)
        rad_vel_list.append(rad_vel)

    with open("velocity_by_HI_2.dat", mode="w+", encoding="utf-8") as output_file:
        output_file.write("MJD\tradial velocity, km/s\n")
        for i in range(len(observation_list)):
            output_file.write(
                f"{observation_list[i].mjd:.14f}\t{rad_vel_list[i] / 1000.0:+.1f}\n"
            )


if __name__ == "__main__":
    main()
