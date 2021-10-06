import pathlib

import yaml


def parse_config(yaml_file: pathlib.Path) -> dict:
    configuration = yaml.safe_load(yaml_file.read_text())

    # Transform relative file paths (assumed relative to YAML config) to absolute paths
    for well in configuration["wells"]:
        trajectory_file = pathlib.Path(well["trajectory"])
        if not trajectory_file.is_absolute():
            trajectory_file = (yaml_file.parent / trajectory_file).resolve().absolute()
        well["trajectory"] = str(trajectory_file)

    return configuration
