import os

# from pprint import pprint
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

from . import DATA_DIR
from .models.spectra import Spectra  # , generate_gauss_spectra
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


@dataclass
class DateAndVelocity:
    date_mjd: float
    rad_vel: float


def load_data(file_path):
    """Загрузка данных из файла."""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue

            try:
                mjd = float(parts[0])
                vel = float(parts[1])
                data.append(DateAndVelocity(mjd, vel))
            except ValueError:
                pass

    return data


def extract_times_and_velocities(data):
    """Преобразование списка объектов в массивы времени и скоростей."""
    times = np.array([d.date_mjd for d in data])
    velocities = np.array([d.rad_vel for d in data])
    return times, velocities


def fwhm_error(freqs, powers):
    """Расчёт ошибки периода через FWHM (Full Width at Half Maximum)."""
    max_power_idx = np.argmax(powers)
    half_max_power = powers[max_power_idx] * 0.5

    # Нахождение точек пересечения половинной мощности слева и справа от максимума
    left_cross_idx = np.where((powers[: max_power_idx + 1][::-1] <= half_max_power))[0][
        -1
    ]
    right_cross_idx = (
        np.where((powers[max_power_idx:] >= half_max_power))[0][-1] + max_power_idx
    )

    freq_left = freqs[left_cross_idx]
    freq_right = freqs[right_cross_idx]

    delta_freq = abs(freq_right - freq_left)
    error_in_period = delta_freq / ((freq_left + freq_right) / 2) ** 2
    return error_in_period


def calculate_radial_velocity_errors(velocity_files):
    """Вычисление стандартных ошибок радиальных скоростей."""
    all_data = {}  # Храним измерения для каждого MJD отдельно

    # Объединение данных из всех файлов
    for file_name in velocity_files:
        data = load_data(file_name)
        for item in data:
            key = round(item.date_mjd, 14)  # Округлим до шести знаков после запятой
            if key not in all_data:
                all_data[key] = {"times": [], "velocities": []}
            all_data[key]["times"].append(item.date_mjd)
            all_data[key]["velocities"].append(item.rad_vel)

    # Теперь считаем ошибки радиальных скоростей
    result = []
    for mjd, values in all_data.items():
        mean_vel = np.mean(values["velocities"])
        std_err = np.std(values["velocities"]) / np.sqrt(
            len(values["velocities"])
        )  # Стандартная ошибка
        result.append((mjd, mean_vel, std_err))

    return result


def compute_period(velocity_files):
    """Вычисление среднего спектра и определение наилучшего периода."""
    # Получаем ошибки скоростей
    errors = calculate_radial_velocity_errors(velocity_files)
    times = np.array([err[0] for err in errors])
    velocities = np.array([err[1] for err in errors])
    errors = np.array([err[2] for err in errors])

    # Используем метод Ломб-Скаргла с учтёнными ошибками
    ls = LombScargle(times, velocities)
    # ls = LombScargle(times, velocities, dy=errors)
    frequency, power = ls.autopower(method="slow", normalization="psd")

    # Поиск частоты с наибольшей мощностью (это соответствует наиболее вероятному периоду)
    best_frequency = frequency[np.argmax(power)]
    best_period = 1 / best_frequency

    # Оцениваем ошибку периода используя FWHM
    error_in_period = fwhm_error(frequency, power)

    return best_period, error_in_period, frequency, power


def main() -> None:
    velocity_files_arr = [
        "velocity_by_HeII.dat",
        "velocity_by_HeII_1.dat",
        "velocity_by_HeII_2.dat",
    ]

    best_period, error, frequencies, powers = compute_period(velocity_files_arr)
    print(f"Период {best_period:.3f} +/- {error:.3f} дней")


if __name__ == "__main__":
    main()
