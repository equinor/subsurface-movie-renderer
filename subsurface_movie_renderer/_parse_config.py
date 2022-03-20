import datetime
import pathlib
from typing import Union

import yaml


def dt2yearfraction(date_object: datetime.datetime) -> float:
    start_of_year = datetime.date(date_object.year, 1, 1).toordinal()
    end_of_year = datetime.date(date_object.year + 1, 1, 1).toordinal()

    return date_object.year + (date_object.toordinal() - start_of_year) / (
        end_of_year - start_of_year
    )


def parse_config(yaml_file: pathlib.Path) -> dict:
    def absolute_path(path: Union[pathlib.Path, str]) -> str:
        if isinstance(path, str):
            path = pathlib.Path(path)

        if path.is_absolute():
            return str(path)
        return str((yaml_file.parent / path).resolve().absolute())

    configuration = yaml.safe_load(yaml_file.read_text())

    # Transform relative file paths (assumed relative to YAML config) to absolute paths
    for well in configuration["wells"]:
        trajectory_file = pathlib.Path(well["trajectory"])
        well["trajectory"] = absolute_path(trajectory_file)

    for horizon_settings in configuration["static_horizons"].values():
        for key in ["X", "Y", "Z"]:
            horizon_settings[key] = absolute_path(horizon_settings[key])

    for survey in configuration["surveys"].values():
        survey["time"] = dt2yearfraction(survey["time"])

    for horizon_settings in configuration["time_dependent_horizons"].values():
        horizon_settings["start_time"] = dt2yearfraction(horizon_settings["start_time"])

    for horizon_settings in configuration["time_dependent_horizons"].values():
        horizon_settings["data_folder"] = absolute_path(horizon_settings["data_folder"])

    return configuration
