import csv
import json
import pathlib
import tempfile
import subprocess

import tqdm
import requests
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


def _create_camera_coordinates(
    camera_path: dict, movie_duration: int, fps: int, output: pathlib.Path
) -> int:

    camera_coordinates = np.array(list(camera_path.values()))
    times_points = list(camera_path.keys())

    times_equal_spacing = list(np.linspace(times_points[0], times_points[-1], 100))

    x_lin = interp1d(times_points, camera_coordinates[:, 0], kind="linear")
    y_lin = interp1d(times_points, camera_coordinates[:, 1], kind="linear")
    z_lin = interp1d(times_points, camera_coordinates[:, 2], kind="linear")

    x_spline = UnivariateSpline(times_equal_spacing, x_lin(times_equal_spacing))
    y_spline = UnivariateSpline(times_equal_spacing, y_lin(times_equal_spacing))
    z_spline = UnivariateSpline(times_equal_spacing, z_lin(times_equal_spacing))

    n_frames = movie_duration * fps
    times_final = list(np.linspace(times_points[0], times_points[-1], n_frames))

    with open(output, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        for txyz in zip(
            times_final,
            x_spline(times_final),
            y_spline(times_final),
            z_spline(times_final),
        ):
            csvwriter.writerow(txyz)

    return n_frames


def render_movie(output_path: pathlib.Path, configuration: dict) -> None:

    movie_duration = configuration["visual_settings"]["movie_duration"]
    fps = configuration["visual_settings"]["fps"]

    with tempfile.TemporaryDirectory(dir=".", prefix=".tmp") as tmp_dir:
        tmp_dir_path = pathlib.Path(tmp_dir)

        n_frames = _create_camera_coordinates(
            camera_path=configuration["visual_settings"]["camera_path"],
            movie_duration=movie_duration,
            fps=fps,
            output=tmp_dir_path / "camera_coordinates.csv",
        )

        (tmp_dir_path / "font.woff").write_bytes(
            requests.get(configuration["visual_settings"]["font"]).content
        )

        with subprocess.Popen(
            [
                "blender",
                "-b",
                "-P",
                pathlib.Path(__file__).parent.resolve() / "_blender_script.py",
                json.dumps(configuration),
            ],
            cwd=tmp_dir_path,
            stdout=subprocess.PIPE,
        ) as process:
            if process.stdout is not None:
                with tqdm.tqdm(
                    total=n_frames,
                    bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} frames rendered ({remaining})",
                ) as pbar:
                    for line in process.stdout:
                        if line.startswith(b"Saved: 'image"):
                            pbar.update()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_dir_path / "image%06d.png",
                "-r",
                str(fps),
                "-b",
                "10000k",
                "-vcodec",
                "msmpeg4v2",
                output_path,
            ],
            capture_output=True,
            check=True,
        )

        print(f"Movie created and saved at {output_path}")
