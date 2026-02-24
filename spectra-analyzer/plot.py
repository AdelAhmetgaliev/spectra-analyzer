import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Tuple

from .models.spectra import Spectra
from .models.observation import Observation


def save_figure(
    fig,
    path: str,
    *,
    dpi: int = 300,
    transparent: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    quality: int | None = None,  # only for JPEG (1-95)
    optimize: bool = True,  # for PNG/JPEG
) -> None:
    save_kwargs = {
        "dpi": dpi,
        "transparent": transparent,
        "bbox_inches": bbox_inches,
        "pad_inches": pad_inches,
        "facecolor": fig.get_facecolor(),
        "edgecolor": "none",
    }

    ext = path.split(".")[-1].lower()
    if ext in {"png", "jpg", "jpeg", "tif", "tiff"}:
        # растровые форматы
        if ext in {"jpg", "jpeg"} and quality is not None:
            save_kwargs["quality"] = max(1, min(95, int(quality)))
        # TIFF может поддерживать dpi; для PNG/JPEG dpi уже передано
    else:
        # для векторных форматов dpi обычно не важен, но можно оставить
        # убираем опции, не поддерживаемые некоторыми векторами
        save_kwargs.pop("quality", None)

    fig.savefig(path, **save_kwargs)


def plot_spectrum(
    observation: Observation, spectra: Spectra, *, show: bool = True
) -> Tuple:
    wl_list, intens_list = spectra.unzip()
    wl = np.asarray(wl_list, dtype=float)
    inten = np.asarray(intens_list, dtype=float)

    if wl.size == 0:
        raise ValueError("Spectrum is empty")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl, inten, color="#d24e26", linewidth=2)
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

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_spectrum_and_filtered(
    observation: Observation,
    spectra: Spectra,
    spectra_filtered: Spectra,
    *,
    show: bool = True,
) -> Tuple:
    wl_list, intens_list = spectra.unzip()
    wl = np.asarray(wl_list, dtype=float)
    inten = np.asarray(intens_list, dtype=float)
    inten_f = np.asarray(spectra_filtered.intensities(), dtype=float)

    if wl.size == 0:
        raise ValueError("Spectrum is empty")
    if inten_f.shape != inten.shape:
        raise ValueError("Filtered spectrum length mismatch")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl, inten, color="#d24e26", linewidth=2)
    ax.plot(wl, inten_f, color="#8c3e3b", linewidth=2)
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

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_lines(
    line_spectra: Spectra, line_spectra_generated: Spectra, *, show: bool = True
) -> Tuple:
    wl_list, intens_list = line_spectra.unzip()
    wl = np.asarray(wl_list, dtype=float)
    inten = 1.0 - np.asarray(intens_list, dtype=float)
    inten_gen = 1.0 - np.asarray(line_spectra_generated.intensities(), dtype=float)

    if wl.size == 0:
        raise ValueError("Spectrum is empty")
    if inten_gen.shape != inten.shape:
        raise ValueError("Generated spectrum length mismatch")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl, inten, color="#d24e26", linewidth=2)
    ax.plot(wl, inten_gen, color="#8c3e3b", linewidth=2)
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

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_rv_and_phase(
    times: np.ndarray,
    rvels: np.ndarray,
    errs: np.ndarray,
    period: float,
    *,
    time_label: str = "MJD",
    rv_label: str = "Лучевая скорость, км/с",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    marker: str = "o",
    color: str = "k",
    phase_color: str = "k",
    fontsize: int = 12,
    grid_alpha: float = 0.5,
) -> None:
    if not (len(times) == len(rvels) == len(errs)):
        raise ValueError("times, rvels и errs должны иметь одинаковую длину")

    phases = (times % period) / period
    order = np.argsort(phases)
    phases_sorted = phases[order]
    rvels_sorted = rvels[order]
    errs_sorted = errs[order]

    phases_dup = np.concatenate([phases_sorted - 1.0, phases_sorted])
    rvels_dup = np.concatenate([rvels_sorted, rvels_sorted])
    errs_dup = np.concatenate([errs_sorted, errs_sorted])

    plt.rcParams.update(
        {
            "font.size": fontsize,
            "font.family": "sans-serif",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=False, gridspec_kw={"height_ratios": (1, 1.1)}
    )

    eb_kwargs = dict(
        fmt=marker, color=color, ecolor=color, elinewidth=1.0, capsize=3, markersize=4
    )

    ax1.errorbar(times, rvels, yerr=errs, **eb_kwargs)
    ax1.set_xlabel(time_label, fontsize=fontsize)
    ax1.set_ylabel(rv_label, fontsize=fontsize)
    ax1.set_ylim(-110, 110)
    ax1.set_yticks([-100, -50, 0, 50, 100])
    ax1.set_title(title or "Лучевая скорость звезды", fontsize=fontsize + 1)
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=grid_alpha)
    ax1.tick_params(which="both", top=True, right=True, labelsize=fontsize)
    ax1.minorticks_on()

    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)

    ax2.errorbar(
        phases_dup,
        rvels_dup,
        yerr=errs_dup,
        fmt=marker,
        color=phase_color,
        ecolor=phase_color,
        elinewidth=1.0,
        capsize=3,
        markersize=4,
    )
    ax2.set_xlabel(f"Фаза (период = {period:.5g} суток)", fontsize=fontsize)
    ax2.set_ylabel(rv_label, fontsize=fontsize)
    ax2.set_xlim(-1.02, 1.02)
    ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.set_ylim(-110, 110)
    ax2.set_yticks([-100, -50, 0, 50, 100])
    ax2.grid(True, linestyle="--", linewidth=0.6, alpha=grid_alpha)
    ax2.tick_params(which="both", top=True, right=True, labelsize=fontsize)
    ax2.minorticks_on()
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()
    save_figure(fig, "rv_time_phase.png")


def plot_rv_phase_with_model(
    times: np.ndarray,
    rvels: np.ndarray,
    errs: np.ndarray,
    period: float,
    model_func: Callable[..., np.ndarray],
    model_params: np.ndarray,
    *,
    time_label: str = "MJD",
    rv_label: str = "Лучевая скорость, км/с",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    marker: str = "o",
    color: str = "k",
    phase_color: str = "k",
    model_color: str = "tab:red",
    fontsize: int = 12,
    grid_alpha: float = 0.5,
    model_sampling: int = 1000,
) -> Tuple:
    if not (len(times) == len(rvels) == len(errs)):
        raise ValueError("times, rvels и errs должны иметь одинаковую длину")

    phases = (times % period) / period
    order = np.argsort(phases)
    phases_sorted = phases[order]
    rvels_sorted = rvels[order]
    errs_sorted = errs[order]

    phases_dup = np.concatenate([phases_sorted - 1.0, phases_sorted])
    rvels_dup = np.concatenate([rvels_sorted, rvels_sorted])
    errs_dup = np.concatenate([errs_sorted, errs_sorted])

    plt.rcParams.update(
        {
            "font.size": fontsize,
            "font.family": "sans-serif",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=False, gridspec_kw={"height_ratios": (1, 1.1)}
    )

    eb_kwargs = dict(
        fmt=marker, color=color, ecolor=color, elinewidth=1.0, capsize=3, markersize=4
    )

    ax1.errorbar(times, rvels, yerr=errs, **eb_kwargs)
    ax1.set_xlabel(time_label, fontsize=fontsize)
    ax1.set_ylabel(rv_label, fontsize=fontsize)
    ax1.set_ylim(-110, 110)
    ax1.set_yticks([-100, -50, 0, 50, 100])
    ax1.set_title(title or "Лучевая скорость звезды", fontsize=fontsize + 1)
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=grid_alpha)
    ax1.tick_params(which="both", top=True, right=True, labelsize=fontsize)
    ax1.minorticks_on()
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)

    ax2.errorbar(
        phases_dup,
        rvels_dup,
        yerr=errs_dup,
        fmt=marker,
        color=phase_color,
        ecolor=phase_color,
        elinewidth=1.0,
        capsize=3,
        markersize=4,
        label="Лучевая скорость",
    )

    phi_model = np.linspace(-1.0, 1.0, model_sampling)
    phi_model_mod = phi_model % 1.0
    try:
        model_vals = model_func(phi_model_mod, *tuple(model_params))
    except Exception as e:
        raise RuntimeError(f"model_func вызвал ошибку при вычислении: {e}")

    ax2.plot(
        phi_model,
        model_vals,
        color=model_color,
        linewidth=1.6,
        label="Аппроксимирующая функция",
    )

    ax2.set_xlabel(f"Фаза (период = {period:.5g} суток)", fontsize=fontsize)
    ax2.set_ylabel(rv_label, fontsize=fontsize)
    ax2.set_xlim(-1.02, 1.02)
    ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.set_ylim(-110, 110)
    ax2.set_yticks([-100, -50, 0, 50, 100])
    ax2.grid(True, linestyle="--", linewidth=0.6, alpha=grid_alpha)
    ax2.tick_params(which="both", top=True, right=True, labelsize=fontsize)
    ax2.minorticks_on()
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)

    ax2.legend(frameon=False, loc="upper left", fontsize=fontsize - 2)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_phase_only_with_model(
    times: np.ndarray,
    rvels: np.ndarray,
    errs: np.ndarray,
    period: float,
    model_func: Callable[..., np.ndarray],
    model_params: np.ndarray,
    *,
    time_label: str = "MJD",
    rv_label: str = "Лучевая скорость, км/с",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    marker: str = "o",
    point_color: str = "k",
    model_color: str = "tab:red",
    fontsize: int = 14,
    grid_alpha: float = 0.5,
    model_sampling: int = 1000,
) -> Tuple:
    if not (len(times) == len(rvels) == len(errs)):
        raise ValueError("times, rvels и errs должны иметь одинаковую длину")

    rvels_kms = np.asarray(rvels, dtype=float)
    errs_kms = np.asarray(errs, dtype=float)
    times = np.asarray(times, dtype=float)

    phases = (times % period) / period
    order = np.argsort(phases)
    phases_sorted = phases[order]
    rvels_sorted = rvels_kms[order]
    errs_sorted = errs_kms[order]

    phases_dup = np.concatenate([phases_sorted - 1.0, phases_sorted])
    rvels_dup = np.concatenate([rvels_sorted, rvels_sorted])
    errs_dup = np.concatenate([errs_sorted, errs_sorted])

    plt.rcParams.update(
        {
            "font.size": fontsize,
            "font.family": "sans-serif",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.errorbar(
        phases_dup,
        rvels_dup,
        yerr=errs_dup,
        fmt=marker,
        color=point_color,
        ecolor=point_color,
        elinewidth=1.0,
        capsize=3,
        markersize=4,
        label="Лучевая скорость",
    )

    phi_model = np.linspace(-1.0, 1.0, model_sampling)
    phi_model_mod = phi_model % 1.0
    try:
        model_vals = model_func(phi_model_mod, *tuple(model_params))
    except Exception as e:
        raise RuntimeError(f"model_func вызвал ошибку при вычислении: {e}")

    ax.plot(
        phi_model,
        model_vals,
        color=model_color,
        linewidth=1.6,
        label="Аппроксимирующая функция",
    )

    # подписи и оформление (аналогично вашим функциям)
    ax.set_xlabel(f"Фаза (период = {period:.5g} суток)", fontsize=fontsize)
    ax.set_ylabel(rv_label, fontsize=fontsize)
    ax.set_xlim(-1.02, 1.02)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    # автоматический подбор ylim, но шаги/типы сохраняем похожими на предыдущие функции
    # если явных границ нет — оставить автоподбор, иначе можно задать как в примере (-110,110)
    ymin, ymax = (
        np.nanmin(rvels_kms) - 0.1 * np.ptp(rvels_kms)
        if np.ptp(rvels_kms) > 0
        else -1.0,
        np.nanmax(rvels_kms) + 0.1 * np.ptp(rvels_kms)
        if np.ptp(rvels_kms) > 0
        else 1.0,
    )
    ax.set_ylim(ymin, ymax)

    # сетка, тики, рамки
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=grid_alpha)
    ax.tick_params(which="both", top=True, right=True, labelsize=fontsize)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # легенда
    ax.legend(frameon=False, loc="upper left", fontsize=fontsize - 2)

    if title:
        ax.set_title(title, fontsize=fontsize + 1)

    plt.tight_layout()
    return fig, ax


def plot_periodogram(freqs, powers, *, show: bool = True) -> Tuple:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, powers, color="#4e0e1f", linewidth=2)
    ax.set_title("Периодограмма метода Ломба-Скарла", fontsize=16, fontweight="bold")
    ax.set_xlabel("Частота", fontsize=14, fontweight="bold")
    ax.set_ylabel("Мощность", fontsize=14, fontweight="bold")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
