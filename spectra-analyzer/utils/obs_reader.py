import os

from ..models.observation import Observation


def read_observations_from_file(file_path: str | os.PathLike) -> list[Observation]:
    result_list: list[Observation] = []

    with open(file=file_path, mode="r", encoding="utf-8") as data_file:
        data_file.readline()

        for line in data_file:
            line_list = line.split()

            mjd = float(line_list[0])
            rel_filepath_list = line_list[1].split("/")
            filepath = os.path.join(os.path.dirname(file_path), *rel_filepath_list)

            result_list.append(Observation(file_path=filepath, mjd=mjd))

    return result_list
