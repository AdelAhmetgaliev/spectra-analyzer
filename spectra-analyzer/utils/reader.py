import os

from pathlib import Path
from ..models.observation import Observation

PathLikeStr = os.PathLike[str] | str | Path


def read_observations_from_file(
    file_path: PathLikeStr, *, path_sep: str = "/"
) -> list[Observation]:
    p = Path(os.fspath(file_path))
    observations: list[Observation] = []

    with p.open("r", encoding="utf-8") as fh:
        _header = fh.readline()

        for lineno, line in enumerate(fh, start=2):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Malformed line {lineno} in {p}: expected at least 2 columns"
                )

            try:
                mjd = float(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid MJD on line {lineno} in {p}: {parts[0]}"
                ) from exc

            rel_path = parts[1]
            if path_sep != os.path.sep and path_sep != "/":
                rel_parts = rel_path.split(path_sep)
            else:
                rel_parts = rel_path.split("/")

            filepath = Path(p.parent, *rel_parts)

            observations.append(Observation(file_path=filepath, mjd=mjd))

    return observations
