from dataclasses import dataclass


@dataclass
class Observation:
    file_path: str
    mjd: float
