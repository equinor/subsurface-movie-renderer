import argparse
import shutil
import pathlib
import subprocess

from ._parse_config import parse_config
from ._render_movie import render_movie
from ._ow2numpy import process_file


def _check_nonpython_dependencies() -> None:

    if shutil.which("blender") is None:
        raise RuntimeError(
            "blender executable not available in $PATH. Download "
            "(https://www.blender.org/download/), extract and make available on $PATH. "
            "Currently only blender <= 2.79 is supported."
        )

    blender_version = (
        subprocess.run(
            ["blender", "--version"], capture_output=True, text=True, check=True
        )
        .stdout.splitlines()[0]
        .split()[1]
    )

    if int(blender_version.split(".")[0]) < 3:
        raise RuntimeError("Blender major version needs to be >= 3")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg executable not available in $PATH. Download (https://www.ffmpeg.org/), "
            "extract and make available on $PATH."
        )


def main_renderer() -> None:

    parser = argparse.ArgumentParser(prog=("Render movie with subsurface data"))
    parser.add_argument(
        "yaml_file", type=pathlib.Path, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("3D-movie.webm"),
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


def main_ow2np() -> None:

    parser = argparse.ArgumentParser(prog=("Convert OpenWorks data to numpy format"))
    parser.add_argument(
        "openworks_exported_file",
        type=pathlib.Path,
        help="Path to OpenWorks exported file",
    )
    parser.add_argument(
        "output_folder",
        type=pathlib.Path,
        metavar="OUTPUT_FOLDER",
        help="Output folder for regularized numpy data",
    )

    parser.add_argument(
        "--samplingint",
        type=float,
        default=12.5,
        help="Sampling interval/distance in regularized output grid",
    )

    args = parser.parse_args()

    process_file(
        input_file=args.openworks_exported_file,
        output_folder=args.output_folder,
        samplingint=args.samplingint,
    )
