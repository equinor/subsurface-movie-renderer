from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot


def _regularize_grid(x_data, y_data, z_data, X, Y, samplingint):
    """Regularize unregular data points. Points in the regularized grid which are more
    than a distance 1.1*samplingint away from a point in the original source is given the value NaN.

    X: matrix corresponding to the wanted x-coordinates
    Y: matrix corresponding to the wanter y-coordinates
    samplingint: approximate samplinginterval in the input data
                 (assumed to use the same unit as X and Y)
    """

    x_data_new = []
    y_data_new = []
    z_data_new = []

    [Ny, Nx] = np.shape(X)

    for i in range(Ny):
        for j in range(Nx):
            if (
                np.min(np.min((X[i, j] - x_data) ** 2 + (Y[i, j] - y_data) ** 2))
                > (1.1 * samplingint) ** 2
            ):
                x_data_new.append(X[i, j])
                y_data_new.append(Y[i, j])
                z_data_new.append(0)  # -9999)

        print(f"Finished with {int((i+1)*100/Ny)} %  ", end="\r")

    x_data = np.concatenate((x_data, np.array(x_data_new)))
    y_data = np.concatenate((y_data, np.array(y_data_new)))
    z_data = np.concatenate((z_data, np.array(z_data_new)))

    Z = griddata(
        np.array([x_data, y_data]).transpose(), z_data, (X, Y), method="nearest"
    )
    Z[Z <= -9999] = np.nan

    return Z


def iterate_file(input_file: Path):
    """Iterates line by line through the OpenWorks exported data file,
    skipping lines which does not contain data. Also, keep track
    of which horizon currently streamed data belongs to.
    """

    for line in input_file.read_text().splitlines():
        if line.startswith("!"):
            continue

        if line.startswith(
            " "
        ):  # TODO: Any better file format definition of OpenWorks export
            # that can be used to indentify rows with actual data?
            _line, _trace, x, y, amp = line.strip().split()

            x = float(x)
            y = float(y)
            amp = float(amp)

            yield x, y, amp, horizon_name
        elif len(line.strip().split()) > 4:
            horizon_name = line.strip().split()[3]


# TODO: Remove disable
# pylint: disable=too-many-arguments
def _store_data(
    x_data, y_data, amp_data, horizon_name, X, Y, samplingint, amp_max, output_folder
):
    hor, survey, _, _ = horizon_name.split("+")

    Z = _regularize_grid(
        np.array(x_data), np.array(y_data), np.array(amp_data), X, Y, samplingint
    )
    Z *= 100 / amp_max

    np.save(output_folder / f"{hor}--{survey}.npy", Z)

    Z[Z == 0.0] = np.nan
    pyplot.imshow(Z)
    pyplot.colorbar()
    pyplot.savefig(output_folder / f"{hor}--{survey}.png")
    pyplot.clf()
    print(f"Finished with: {horizon_name}")


# TODO: Remove disable
# pylint: disable=too-many-locals
def process_file(input_file: Path, output_folder: Path, samplingint: float):
    boundary_margin = (
        5 * samplingint
    )  # margin from plume edge to boundary in formatted data

    x_max = -np.inf
    x_min = np.inf

    y_max = -np.inf
    y_min = np.inf

    amp_max = -np.inf

    # Finding final boundary values and max value in data set:

    for x, y, amp, _ in iterate_file(input_file):

        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        if amp > amp_max:
            amp_max = amp

    x_max += boundary_margin
    x_min -= boundary_margin

    y_max += boundary_margin
    y_min -= boundary_margin

    Nx = int((x_max - x_min) / samplingint)
    Ny = int((y_max - y_min) / samplingint)

    x = np.linspace(x_min, x_min + (Nx - 1) * samplingint, Nx)
    y = np.linspace(y_min, y_min + (Ny - 1) * samplingint, Ny)

    X, Y = np.meshgrid(x, y)

    output_folder.mkdir(parents=True, exist_ok=True)

    np.save(output_folder / "X_coordinates.npy", X)
    np.save(output_folder / "Y_coordinates.npy", Y)

    # Converting to numpy format:

    prev_horizon_name = None
    for x, y, amp, horizon_name in iterate_file(input_file):
        if prev_horizon_name != horizon_name:
            if prev_horizon_name is not None:
                _store_data(
                    x_data,
                    y_data,
                    amp_data,
                    prev_horizon_name,
                    X,
                    Y,
                    samplingint,
                    amp_max,
                    output_folder,
                )

            x_data = []
            y_data = []
            amp_data = []
            prev_horizon_name = horizon_name

        x_data.append(x)
        y_data.append(y)
        amp_data.append(amp)

    _store_data(
        x_data,
        y_data,
        amp_data,
        prev_horizon_name,
        X,
        Y,
        samplingint,
        amp_max,
        output_folder,
    )


# TODO: REstructure functions
# TODO: Use tqdm
