import os

from dataclasses import dataclass, field
from pathlib import Path


PathLikeStr = os.PathLike[str] | str | Path


@dataclass(frozen=True)
class Observation:
    """
    Хранит информацию об наблюдении.

    Поля:
    - file_path: pathlib.Path, нормализованный через os.fspath()
    - mjd: float (Modified Julian Date), неотрицательное
    """

    file_path: Path = field()
    mjd: float = field()

    def __init__(self, file_path: PathLikeStr, mjd: float) -> None:
        p = Path(os.fspath(file_path))
        if not str(p):
            raise ValueError("file_path must be a non-empty path")
        if not isinstance(mjd, (int, float)):
            raise TypeError("mjd must be a number")
        if mjd < 0:
            raise ValueError("mjd must be non-negative")

        object.__setattr__(self, "file_path", p)
        object.__setattr__(self, "mjd", float(mjd))
