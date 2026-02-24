import os
import csv
import math
import statistics

import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

from pathlib import Path
from typing import Iterable, Callable, Tuple, Optional

from . import DATA_DIR
from .models.spectra import Spectra
from .models.observation import Observation
from .utils import calculate_radial_velocity
from .utils.reader import read_observations_from_file
from .plot import (
    save_figure,
    plot_rv_and_phase,
    plot_rv_phase_with_model,
    plot_phase_only_with_model,
)

FILTER_WINDOW_LENGTH: int = 11
FILTER_POLY_ORDER: int = 3
DELTA_WL: float = 10.0

PathLikeStr = os.PathLike[str] | str | Path


def safe_radial_velocity(spec: Spectra, lab_wl: float) -> float:
    if not spec.data:
        return math.nan
    try:
        return calculate_radial_velocity(spec, lab_wl)
    except Exception:
        return math.nan


def compute_daily_rv_stats(
    observations: list[Observation], rad_wl_list: list[list[float]]
) -> list[tuple[float, float, float]]:
    if len(observations) != len(rad_wl_list):
        raise ValueError("observations and rad_wl_list must have equal length")

    results: list[tuple[float, float, float]] = []

    for obs, rv_vals in zip(observations, rad_wl_list):
        valid = [
            float(v)
            for v in rv_vals
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        n = len(valid)

        if n == 0:
            mean_rv = math.nan
            sem_rv = math.nan
        elif n == 1:
            mean_rv = valid[0]
            sem_rv = math.nan
        else:
            mean_rv = statistics.mean(valid)
            stdev = statistics.stdev(valid)
            sem_rv = stdev / math.sqrt(n)

        results.append((float(obs.mjd), mean_rv, sem_rv))

    return results


def save_rv_stats_to_file(
    out_path: str | Path,
    stats: Iterable[tuple[float, float, float]],
    *,
    delimiter: str = ",",
    float_format: str = "{:.4f}",
    header: tuple[str, str, str] = ("mjd", "mean_rv", "sem_rv"),
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(header)

        for mjd, mean_rv, sem_rv in stats:

            def fmt(x: float) -> str:
                return (
                    "NaN"
                    if x is None or (isinstance(x, float) and math.isnan(x))
                    else float_format.format(float(x))
                )

            writer.writerow([float(mjd), fmt(mean_rv), fmt(sem_rv)])


def read_rv_csv(path: PathLikeStr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    rvels: list[float] = []
    errs: list[float] = []
    with open(path, "r", encoding="utf-8") as fh:
        _header = fh.readline()
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            t, rv, er = map(float, parts[:3])
            times.append(t)
            rvels.append(rv)
            errs.append(er)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(rvels, dtype=np.float64),
        np.asarray(errs, dtype=np.float64),
    )


def compute_periodogram(
    times: np.ndarray, rvels: np.ndarray, errs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ls = LombScargle(times, rvels, dy=errs)
    freqs, powers = ls.autopower(method="auto", normalization="standard")
    return np.asarray(freqs), np.asarray(powers)


def find_best_frequency(freqs: np.ndarray, powers: np.ndarray) -> tuple[float, int]:
    idx = int(np.argmax(powers))
    best_freq = float(freqs[idx])
    return best_freq, idx


def estimate_period_error(
    freqs: np.ndarray, powers: np.ndarray, peak_idx: int, level: float = 0.5
) -> float:
    peak_power = powers[peak_idx]
    threshold = peak_power * level

    left_idx = 0
    for i in range(peak_idx, -1, -1):
        if powers[i] <= threshold:
            left_idx = i
            break

    right_idx = len(powers) - 1
    for i in range(peak_idx, len(powers)):
        if powers[i] <= threshold:
            right_idx = i
            break

    freq_left = freqs[left_idx]
    freq_right = freqs[right_idx]

    delta_freq = abs(freq_right - freq_left) * 0.5
    avg_freq = 0.5 * (freq_left + freq_right)
    if avg_freq == 0:
        return float("nan")

    period_err = delta_freq / (avg_freq**2)
    return float(period_err)


def fold_and_sort(
    times: np.ndarray, rvels: np.ndarray, errs: np.ndarray, period: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phases = (times % period) / period
    order = np.argsort(phases)
    return phases[order], rvels[order], errs[order]


def fit_custom_model(
    phases: np.ndarray,
    rvels: np.ndarray,
    errs: np.ndarray,
    model_func: Callable[..., np.ndarray],
    init_params: Optional[np.ndarray] = None,
    bounds: Tuple | None = None,
    *,
    maxfev: int = 10000,
    absolute_sigma: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Аппроксимирует rvels(phases) заданной model_func с учётом ошибок errs.

    Параметры:
      phases: array_like, фазы (будут приведены к [0,1) при необходимости).
      rvels: array_like, наблюдаемые значения.
      errs: array_like, погрешности наблюдений (sigma для curve_fit).
      model_func: callable(phi, *params) -> array_like предсказаний той же формы, что и rvels.
                  Первый аргумент должен быть массив фаз.
      init_params: начальное приближение параметров (по умолчанию None).
      bounds: (lower_bounds, upper_bounds) для параметров или None.
      maxfev: максимальное число итераций для curve_fit.
      absolute_sigma: если True, sigma интерпретируется как абсолютные погрешности.

    Возвращает:
      popt: оптимальные параметры (np.ndarray)
      perr: 1σ ошибки параметров (np.ndarray: sqrt(diag(pcov)) или nan-значения)
      rmse: корень из среднеквадратичной ошибки модели (float)
      pcov: ковариационная матрица параметров (np.ndarray) или None
    """
    phases = np.asarray(phases, dtype=np.float64) % 1.0
    rvels = np.asarray(rvels, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)

    if not (phases.size == rvels.size == errs.size):
        raise ValueError("phases, rvels и errs должны иметь одинаковую длину")

    if np.any(errs <= 0):
        positive_errs = errs[errs > 0]
        fallback = float(np.nanmean(positive_errs)) if positive_errs.size > 0 else 1.0
        errs = np.where(errs <= 0, fallback, errs)

    try:
        popt, pcov = curve_fit(
            model_func,
            phases,
            rvels,
            p0=init_params,
            sigma=errs,
            absolute_sigma=absolute_sigma,
            bounds=bounds if bounds is not None else (-np.inf, np.inf),
            maxfev=maxfev,
        )
    except Exception:
        npar = 0 if init_params is None else int(np.size(init_params))
        popt = np.full(npar, np.nan)
        perr = np.full(npar, np.nan)
        return popt, perr, float("nan"), None

    if pcov is not None:
        with np.errstate(invalid="ignore"):
            perr = np.sqrt(np.abs(np.diag(pcov)))
    else:
        perr = np.full_like(popt, np.nan)

    try:
        fitted = model_func(phases, *popt)
        residuals = rvels - fitted
        rmse = float(np.sqrt(np.mean(residuals**2)))
    except Exception:
        rmse = float("nan")

    return (
        np.asarray(popt),
        np.asarray(perr),
        rmse,
        (pcov if pcov is not None else None),
    )


def my_model(phi: np.ndarray, A: float, phi0: float, offset: float) -> np.ndarray:
    return A * np.sin(2.0 * np.pi * (phi - phi0)) + offset


def main() -> None:
    times, rvels, errs = read_rv_csv("radial_velocity.csv")
    rvels = rvels / 1000.0
    errs = errs / 1000.0

    freqs, powers = compute_periodogram(times, rvels, errs)
    best_freq, peak_idx = find_best_frequency(freqs, powers)
    best_period = 1.0 / best_freq

    period_err = estimate_period_error(freqs, powers, peak_idx, level=0.5)

    print(f"{best_period:.2f} +/- {period_err:.2f}")

    sorted_phases, sorted_rvels, sorted_errs = fold_and_sort(
        times, rvels, errs, best_period
    )

    plot_rv_and_phase(times, rvels, errs, best_period)

    init_params = np.array([1.0, 0.0, 0.0])
    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    popt, perr, rmse, pcov = fit_custom_model(
        sorted_phases,
        sorted_rvels,
        sorted_errs,
        my_model,
        init_params=init_params,
        bounds=bounds,
    )
    print(f"{popt[0]:.3f} +/- {perr[0]:.3f}")

    fig, _ = plot_rv_phase_with_model(
        times,
        rvels,
        errs,
        best_period,
        my_model,
        popt,
    )
    save_figure(fig, "rv_time_phase_model.png")

    fig, _ = plot_phase_only_with_model(
        times,
        rvels,
        errs,
        best_period,
        my_model,
        popt,
    )
    save_figure(fig, "rv_phase_model.png")


def _main() -> None:
    path_to_obsdat = Path(os.fspath(DATA_DIR)) / "ObsDat.txt"
    observation_list = read_observations_from_file(path_to_obsdat)
    if not observation_list:
        raise SystemExit("No observations found")

    observation_list.sort(key=lambda obs: obs.mjd)

    spectra_list: list[Spectra] = [
        Spectra.from_file(obs.file_path) for obs in observation_list
    ]

    lab_wl_list = [4199.83, 4541.591, 4685.698]
    rad_wl_list: list[list[float]] = []

    for spec in spectra_list:
        filtered_spec = spec.apply_savgol(FILTER_WINDOW_LENGTH, FILTER_POLY_ORDER)

        temp_list: list[float] = []
        for lab_wl in lab_wl_list:
            line_spec = filtered_spec.filter_by_wavelength(
                lab_wl - DELTA_WL, lab_wl + DELTA_WL
            )
            temp_list.append(safe_radial_velocity(line_spec, lab_wl))
        rad_wl_list.append(temp_list)

    velocity_stat = compute_daily_rv_stats(observation_list, rad_wl_list)
    save_rv_stats_to_file("radial_velocity.csv", velocity_stat)


if __name__ == "__main__":
    main()
