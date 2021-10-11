import argparse
import shutil
import pathlib

from ._parse_config import parse_config
from ._render_movie import render_movie


def _check_nonpython_dependencies() -> None:

    if shutil.which("blender") is None:
        raise RuntimeError(
            "blender executable not available in $PATH. Download "
            "(https://www.blender.org/download/), extract and make available on $PATH. "
            "Currently only blender <= 2.79 is supported."
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg executable not available in $PATH. Download (https://www.ffmpeg.org/), "
            "extract and make available on $PATH."
        )


def main() -> None:

    parser = argparse.ArgumentParser(prog=("Render movie with subsurface data"))
    parser.add_argument(
        "yaml_file", type=pathlib.Path, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("3D-movie.avi"),
        metavar="OUTPUT_FILE",
        help="Output filename for rendered movie",
    )

    args = parser.parse_args()

    _check_nonpython_dependencies()

    configuration = parse_config(args.yaml_file)

    output_path = (
        args.output
        if args.output.is_absolute()
        else (pathlib.Path.cwd() / args.output).resolve()
    )

    render_movie(output_path=output_path, configuration=configuration)
