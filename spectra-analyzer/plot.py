import numpy as np
import matplotlib.pyplot as plt

from .models.spectra import Spectra
from .models.observation import Observation


def plot_spectrum(
    observation: Observation, spectra: Spectra, *, show: bool = True
) -> None:
    wl_list, intens_list = spectra.unzip()
    wl = np.asarray(wl_list, dtype=float)
    inten = np.asarray(intens_list, dtype=float)

    if wl.size == 0:
        raise ValueError("Spectrum is empty")

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl, inten, color="blue", linewidth=2)
    ax.set_title(
        f"Спектральные данные, MJD={observation.mjd}", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Длина волны (A)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Интенсивность", fontsize=14, fontweight="bold")

    ax.set_xlim(float(wl.min()), float(wl.max()))
    ax.set_ylim(float(inten.min()) * 0.8, float(inten.max()) * 1.1)

    ax.legend(["Спектр"], loc="lower right", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)

    if show:
        plt.show()


def plot_spectrum_and_filtered(
    observation: Observation,
    spectra: Spectra,
    spectra_filtered: Spectra,
    *,
    show: bool = True,
) -> None:
    wl_list, intens_list = spectra.unzip()
    wl = np.asarray(wl_list, dtype=float)
    inten = np.asarray(intens_list, dtype=float)
    inten_f = np.asarray(spectra_filtered.intensities(), dtype=float)

    if wl.size == 0:
        raise ValueError("Spectrum is empty")
    if inten_f.shape != inten.shape:
        raise ValueError("Filtered spectrum length mismatch")

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl, inten, color="blue", linewidth=2)
    ax.plot(wl, inten_f, color="green", linewidth=2)
    ax.set_title(
        f"Спектральные данные, MJD={observation.mjd}", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Длина волны (A)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Интенсивность", fontsize=14, fontweight="bold")

    ax.set_xlim(float(wl.min()), float(wl.max()))
    ax.set_ylim(float(inten.min()) * 0.8, float(inten.max()) * 1.1)

    ax.legend(
        ["Спектр исходный", "Спектр отфильтрованный"], loc="lower right", fontsize=12
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)

    if show:
        plt.show()


def plot_lines(
    line_spectra: Spectra, line_spectra_generated: Spectra, *, show: bool = True
) -> None:
    wl_list, intens_list = line_spectra.unzip()
    wl = np.asarray(wl_list, dtype=float)
    inten = 1.0 - np.asarray(intens_list, dtype=float)
    inten_gen = 1.0 - np.asarray(line_spectra_generated.intensities(), dtype=float)

    if wl.size == 0:
        raise ValueError("Spectrum is empty")
    if inten_gen.shape != inten.shape:
        raise ValueError("Generated spectrum length mismatch")

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl, inten, color="blue", linewidth=2)
    ax.plot(wl, inten_gen, color="red", linewidth=2)
    ax.set_title("Сравнение линий", fontsize=16, fontweight="bold")
    ax.set_xlabel("Длина волны (A)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Интенсивность", fontsize=14, fontweight="bold")

    ax.set_xlim(float(wl.min()), float(wl.max()))
    ax.set_ylim(0.0, float(inten.max()) * 1.1)

    ax.legend(
        ["Линия исходная", "Линия сгенерированная"], loc="upper right", fontsize=12
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)

    if show:
        plt.show()
