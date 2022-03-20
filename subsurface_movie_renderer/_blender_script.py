"""
This file is automatically executed directly through blender (since blender ships its own python).
It will not work outside of the blender provided Python distribution.
"""

import csv
import sys
import json
import math
from typing import List, Tuple
from pathlib import Path

import numpy as np

# Blender specific packages which are available only in Blender shipped Python:
import bpy  # pylint: disable=import-error
from mathutils import Vector, Color  # pylint: disable=import-error

# TODO: Remove these disables
# pylint: disable=redefined-outer-name, too-many-locals, too-many-statements


def _set_world_parameters() -> None:
    world = bpy.data.worlds["World"]
    world.horizon_color = Color((0.5, 0.5, 0.5))
    world.zenith_color = Color((0.8, 0.8, 0.8))
    world.use_sky_blend = True


def _add_text_annotations(annotations: List[dict]) -> None:

    blender_font = bpy.data.fonts.load("font.woff")

    for ann in annotations:
        for i, _ in enumerate(ann["rotation"]):
            ann["rotation"][i] *= math.pi / 180

        x, y, z = ann["location"]
        x *= SCALE_X
        y *= SCALE_Y
        z *= SCALE_Z

        bpy.ops.object.text_add(location=(x, y, z), rotation=ann["rotation"])
        txt = bpy.context.scene.objects.active
        txt.data.body = ann["label"]
        txt.data.font = blender_font


def _add_boundaries(boundary_boxes: List[dict]) -> None:

    for box in boundary_boxes:
        [x_min, x_max] = box["xrange"]
        [y_min, y_max] = box["yrange"]
        [z_min, z_max] = box["zrange"]

        x_min *= SCALE_X
        x_max *= SCALE_X
        y_min *= SCALE_Y
        y_max *= SCALE_Y
        z_min *= SCALE_Z
        z_max *= SCALE_Z

        verts = [
            (x_min, y_max, z_min),
            (x_max, y_max, z_min),
            (x_max, y_min, z_min),
            (x_min, y_min, z_min),
            (x_min, y_max, z_max),
            (x_max, y_max, z_max),
            (x_max, y_min, z_max),
            (x_min, y_min, z_max),
        ]

        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (0, 3, 7, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
        ]

        mesh = bpy.data.meshes.new(box["name"])
        mesh.from_pydata(verts, [], faces)

        new_object = bpy.data.objects.new(box["name"], mesh)
        bpy.context.scene.objects.link(new_object)

        for i, polygon in enumerate(new_object.data.polygons):
            material = bpy.data.materials.new(box["name"] + "_mat" + str(i))
            material.emit = 0.1
            material.diffuse_color = Color(box["color"])
            material.use_transparency = True

            import random  # pylint: disable=import-outside-toplevel

            # TODO: Make this variation deterministic
            material.alpha = random.uniform(
                max(0, box["alpha"] - 0.1), min(1, box["alpha"] + 0.1)
            )

            new_object.data.materials.append(material)
            polygon.material_index = i


def _add_wells(origin: Tuple[int, int, int], wells: List[dict]) -> None:
    for well in wells:
        with open(well["trajectory"]) as csvfile:
            coordinates = [
                (
                    (float(row[0]) - origin[0]) * SCALE_X,
                    (float(row[1]) - origin[1]) * SCALE_Y,
                    -(float(row[2]) - origin[2]) * SCALE_Z,
                )
                for row in csv.reader(csvfile)
            ]

        object_name = "well_" + well["name"]
        curve_name = "well_path_" + well["name"]
        material_name = "well_mat_" + well["name"]

        curvedata = bpy.data.curves.new(name=curve_name, type="CURVE")
        curvedata.dimensions = "3D"

        objectdata = bpy.data.objects.new(object_name, curvedata)
        objectdata.location = coordinates[0]

        bpy.context.scene.objects.link(objectdata)

        polyline = curvedata.splines.new("POLY")
        polyline.points.add(len(coordinates) - 1)
        for i, coordinate in enumerate(coordinates):
            polyline.points[i].co = coordinate + (1,)

        bpy.context.scene.objects.active = bpy.data.objects[object_name]

        well_mat = bpy.data.materials.new(material_name)
        well_mat.emit = 0.5
        well_mat.diffuse_color = Color(well["color"])

        if "alpha" in well:
            well_mat.use_transparency = True
            well_mat.alpha = well["alpha"]

        bpy.data.objects[object_name].data.bevel_depth = 1e-2
        bpy.data.objects[object_name].data.fill_mode = "FULL"
        bpy.data.objects[object_name].data.materials.append(well_mat)


###################################


def define_surface_colorscale(colorscale: List[List[float]]) -> List:
    materials = []

    for i, color in enumerate(colorscale):
        materials.append(bpy.data.materials.new("mat_val" + str(i)))
        materials[-1].emit = 0.4
        materials[-1].use_transparency = True
        materials[-1].diffuse_color = Color(color[:3])

        if i < 10:
            materials[-1].alpha = i / 10.0
        else:
            materials[-1].alpha = 1.0

    return materials


class Horizon:
    def __init__(
        self,
        X: np.array,
        Y: np.array,
        Z: np.array,
        horizon_name: str,
        alpha: float = 1.0,
    ):
        self._X = X
        self._Y = Y
        self._Z = Z
        self._horizon_name = horizon_name
        self._alpha = alpha

        # TODO: Put the above in same fle (as with TDH).
        # TODO: super()

        self._top_ty_materials = []
        n_materials = len(materials)
        for i in range(n_materials):
            red = 1 - 1.0 * i / n_materials
            green = 1 - 1.0 * i / n_materials
            blue = 1 - 1.0 * i / n_materials

            self._top_ty_materials.append(
                bpy.data.materials.new("top_ty_mat_val" + str(i))
            )
            self._top_ty_materials[-1].emit = 0.4
            self._top_ty_materials[-1].use_transparency = True
            self._top_ty_materials[-1].diffuse_color = Color((red, green, blue))

            self._top_ty_materials[-1].alpha = self._alpha

    def update_blender(self) -> None:

        X = self._X
        Y = self._Y
        Z = self._Z

        min_z = np.nanmin(np.nanmin(Z))
        max_z = np.nanmax(np.nanmax(Z))

        amp = 100.0 * (Z - min_z) / (max_z - min_z)
        amp[np.isnan(amp)] = 0

        Z *= -1.0

        # ALL CODE BELOW IS ~REUSED. =>> CONSOLIDATE

        [M, N] = np.shape(Z)

        vertex2index = (
            np.zeros((M + 1) * (N + 1)) * np.nan
        )  # convert vertex location to an index used by Blender
        current_index = -1
        verts = []
        faces = []
        values = []

        for i in range(M):
            for j in range(N):
                if amp[i, j] > 0:
                    if np.isnan(vertex2index[(N + 1) * i + j]):  # top left vertex
                        current_index += 1
                        vertex2index[(N + 1) * i + j] = current_index
                        x_loc = (X[i, j] - 12.5 / 2 - origin[0]) * SCALE_X
                        y_loc = (Y[i, j] + 12.5 / 2 - origin[1]) * SCALE_Y
                        z_loc = (Z[i, j] - origin[2]) * SCALE_Z
                        verts.append((x_loc, y_loc, z_loc))
                    if np.isnan(vertex2index[(N + 1) * i + 1 + j]):  # top right vertex
                        current_index += 1
                        vertex2index[(N + 1) * i + 1 + j] = current_index
                        x_loc = (X[i, j] + 12.5 / 2 - origin[0]) * SCALE_X
                        y_loc = (Y[i, j] + 12.5 / 2 - origin[1]) * SCALE_Y
                        z_loc = (Z[i, j] - origin[2]) * SCALE_Z
                        verts.append((x_loc, y_loc, z_loc))
                    if np.isnan(
                        vertex2index[(N + 1) * (i + 1) + j]
                    ):  # bottom left vertex
                        current_index += 1
                        vertex2index[(N + 1) * (i + 1) + j] = current_index
                        x_loc = (X[i, j] - 12.5 / 2 - origin[0]) * SCALE_X
                        y_loc = (Y[i, j] - 12.5 / 2 - origin[1]) * SCALE_Y
                        z_loc = (Z[i, j] - origin[2]) * SCALE_Z
                        verts.append((x_loc, y_loc, z_loc))
                    if np.isnan(
                        vertex2index[(N + 1) * (i + 1) + 1 + j]
                    ):  # bottom right vertex
                        current_index += 1
                        vertex2index[(N + 1) * (i + 1) + 1 + j] = current_index
                        x_loc = (X[i, j] + 12.5 / 2 - origin[0]) * SCALE_X
                        y_loc = (Y[i, j] - 12.5 / 2 - origin[1]) * SCALE_Y
                        z_loc = (Z[i, j] - origin[2]) * SCALE_Z

                        verts.append((x_loc, y_loc, z_loc))

                    faces.append(
                        (
                            vertex2index[(N + 1) * i + j],
                            vertex2index[(N + 1) * i + 1 + j],
                            vertex2index[(N + 1) * (i + 1) + 1 + j],
                            vertex2index[(N + 1) * (i + 1) + j],
                        )
                    )

                    values.append(int(amp[i, j]))

        try:
            bpy.data.objects[self._horizon_name].select = True
            bpy.ops.object.delete()
            bpy.data.objects[self._horizon_name].select = False
        except KeyError:
            pass

        mesh = bpy.data.meshes.new(self._horizon_name)
        new_object = bpy.data.objects.new(self._horizon_name, mesh)
        new_object.location = bpy.context.scene.cursor_location
        bpy.context.scene.objects.link(new_object)
        mesh.from_pydata(verts, [], faces)
        mesh.update(calc_edges=True)

        bpy.context.scene.objects.active = bpy.data.objects[self._horizon_name]

        for i in range(len(materials)):
            bpy.ops.object.material_slot_add()
            bpy.data.objects[self._horizon_name].material_slots[
                i
            ].material = self._top_ty_materials[i]

        obj = bpy.context.object

        counter = 0
        for polygon in obj.data.polygons:
            polygon.select = True
            polygon.material_index = values[counter]
            counter += 1

        obj.data.update()
        bpy.ops.object.mode_set(mode="OBJECT")  # Return to object mode

        bpy.data.objects[self._horizon_name].select = True
        bpy.ops.object.shade_smooth()
        bpy.data.objects[self._horizon_name].select = False


class TimeDependentHorizon:
    def __init__(self, horizon_name: str, depth: float):
        self._horizon_name = horizon_name
        self._depth = depth

        self._survey_times = json.loads(
            Path(self._horizon_name + "_metadata.json").read_text()
        )

        self._currently_loaded_file = None

    def _load_file(self, time: float) -> None:
        def _get_file_index(time: float) -> int:
            for i, survey_time in enumerate(self._survey_times[1:]):
                if time < survey_time:
                    return i
            return len(self._survey_times) - 2

        file_index = _get_file_index(time)
        file_to_load = self._horizon_name + "_" + str(file_index) + ".npz"

        if self._currently_loaded_file != file_to_load:
            data = np.load(file_to_load)
            self._currently_loaded_file = file_to_load

            self._amp1 = data["amp1"]
            self._amp2 = data["amp2"]

            self.X = data["X"]
            self.Y = data["Y"]

            self._time_a, self._time_b = tuple(
                self._survey_times[file_index : file_index + 2]
            )

            self._AT = self._time_a + (self._time_b - self._time_a) * data["AT"] / 100.0

    def _get_values(self, time: float) -> Tuple[np.array, np.array, np.array]:
        self._load_file(time)

        if time <= self._time_a:
            amp = self._amp1
        elif time >= self._time_b:
            amp = self._amp2
        else:
            interp_scale = (time - self._AT) / (0.01 + self._time_b - self._AT)
            amp = (1 - interp_scale) * self._amp1 + interp_scale * self._amp2
            amp[interp_scale < 0] = 0

        return self.X, self.Y, amp

    def update_blender(self, time: float) -> None:

        X, Y, amp = self._get_values(time)

        Z = amp * (12.5 / 100) - self._depth  # TODO

        [M, N] = np.shape(Z)

        vertex2index = (
            np.zeros((M + 1) * (N + 1)) * np.nan
        )  # convert vertex location to an index used by Blender
        current_index = -1
        verts = []
        faces = []
        values = []

        samplingint = 50

        for i in range(M):
            for j in range(N):
                if amp[i, j] > 0:
                    if np.isnan(vertex2index[(N + 1) * i + j]):  # top left vertex
                        current_index += 1
                        vertex2index[(N + 1) * i + j] = current_index
                        x_loc = (X[i, j] - samplingint / 2 - origin[0]) * SCALE_X
                        y_loc = (Y[i, j] + samplingint / 2 - origin[1]) * SCALE_Y
                        z_loc = (Z[i, j] - origin[2]) * SCALE_Z
                        verts.append((x_loc, y_loc, z_loc))
                    if np.isnan(vertex2index[(N + 1) * i + 1 + j]):  # top right vertex
                        current_index += 1
                        vertex2index[(N + 1) * i + 1 + j] = current_index
                        x_loc = (X[i, j] + samplingint / 2 - origin[0]) * SCALE_X
                        y_loc = (Y[i, j] + samplingint / 2 - origin[1]) * SCALE_Y
                        z_loc = (Z[i, j] - origin[2]) * SCALE_Z
                        verts.append((x_loc, y_loc, z_loc))
                    if np.isnan(
                        vertex2index[(N + 1) * (i + 1) + j]
                    ):  # bottom left vertex
                        current_index += 1
                        vertex2index[(N + 1) * (i + 1) + j] = current_index
                        x_loc = (X[i, j] - samplingint / 2 - origin[0]) * SCALE_X
                        y_loc = (Y[i, j] - samplingint / 2 - origin[1]) * SCALE_Y
                        z_loc = (Z[i, j] - origin[2]) * SCALE_Z
                        verts.append((x_loc, y_loc, z_loc))
                    if np.isnan(
                        vertex2index[(N + 1) * (i + 1) + 1 + j]
                    ):  # bottom right vertex
                        current_index += 1
                        vertex2index[(N + 1) * (i + 1) + 1 + j] = current_index
                        x_loc = (X[i, j] + samplingint / 2 - origin[0]) * SCALE_X
                        y_loc = (Y[i, j] - samplingint / 2 - origin[1]) * SCALE_Y
                        z_loc = (Z[i, j] - origin[2]) * SCALE_Z

                        verts.append((x_loc, y_loc, z_loc))

                    faces.append(
                        (
                            vertex2index[(N + 1) * i + j],
                            vertex2index[(N + 1) * i + 1 + j],
                            vertex2index[(N + 1) * (i + 1) + 1 + j],
                            vertex2index[(N + 1) * (i + 1) + j],
                        )
                    )

                    values.append(int(amp[i, j]))

        try:
            bpy.data.objects[self._horizon_name].select = True
            bpy.ops.object.delete()
            bpy.data.objects[self._horizon_name].select = False
        except KeyError:
            pass

        mesh = bpy.data.meshes.new(self._horizon_name)
        new_object = bpy.data.objects.new(self._horizon_name, mesh)
        new_object.location = bpy.context.scene.cursor_location
        bpy.context.scene.objects.link(new_object)
        mesh.from_pydata(verts, [], faces)
        mesh.update(calc_edges=True)

        bpy.context.scene.objects.active = bpy.data.objects[self._horizon_name]

        for i, material in enumerate(materials):
            bpy.ops.object.material_slot_add()
            bpy.data.objects[self._horizon_name].material_slots[i].material = material

        obj = bpy.context.object

        counter = 0
        for polygon in obj.data.polygons:
            polygon.select = True
            polygon.material_index = values[counter]
            counter += 1

        obj.data.update()
        bpy.ops.object.mode_set(mode="OBJECT")  # Return to object mode

        bpy.data.objects[self._horizon_name].select = True
        bpy.ops.object.shade_smooth()
        bpy.data.objects[self._horizon_name].select = False


def _configure_static_horizons(static_horizons_config: dict) -> list:
    static_horizons = []
    for horizon, horizon_settings in static_horizons_config.items():
        X = np.load(horizon_settings["X"])
        Y = np.load(horizon_settings["Y"])
        Z = np.load(horizon_settings["Z"])
        static_horizons.append(
            Horizon(X, Y, Z, horizon, alpha=horizon_settings["alpha"])
        )

    return static_horizons


def _add_camera_tracking(pos: Tuple[int, int, int]) -> None:
    empty = bpy.data.objects.new("Empty", None)
    empty.location = Vector((pos[0] * SCALE_X, pos[1] * SCALE_Y, pos[2] * SCALE_Z))
    bpy.context.scene.objects.link(empty)
    bpy.context.scene.update()

    track_to = bpy.data.objects["Camera"].constraints.new("TRACK_TO")
    track_to.target = empty
    track_to.track_axis = "TRACK_NEGATIVE_Z"
    track_to.up_axis = "UP_Y"


def _render_frames(
    width: int,
    height: int,
    static_horizons: List[Horizon],
    td_horizons: List[TimeDependentHorizon] = None,
) -> None:

    scene_key = bpy.data.scenes.keys()[0]
    scene = bpy.data.scenes[scene_key]

    scene.camera = bpy.data.objects["Camera"]

    bpy.context.scene.render.resolution_x = 2 * width
    bpy.context.scene.render.resolution_y = 2 * height

    camera = bpy.data.objects["Camera"]
    lamp = bpy.data.objects["Lamp"]
    lamp.data.energy = 0.0

    if static_horizons is not None:
        for static_horizon in static_horizons:
            static_horizon.update_blender()

    with open("camera_coordinates.csv") as csvfile:
        for i, txyz in enumerate(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)):

            t, x, y, z = txyz

            if td_horizons is not None:
                for td_horizon in td_horizons:
                    td_horizon.update_blender(t)  # type: ignore[arg-type]

            x *= SCALE_X  # type: ignore[operator]
            y *= SCALE_Y  # type: ignore[operator]
            z *= SCALE_Z  # type: ignore[operator]

            camera.location.xyz = lamp.location.xyz = Vector((x, y, z))

            scene_key = bpy.data.scenes.keys()[0]
            bpy.data.scenes[scene_key].render.filepath = (
                "image" + (6 - len(str(i))) * "0" + str(i)
            )

            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":

    user_configuration = json.loads(sys.argv[-1])

    origin = user_configuration["coordinate_system"]["origin"]

    materials = define_surface_colorscale(
        json.loads(Path("colorscale.json").read_text())
    )

    # Delete default blender provided cube and lamp:
    bpy.data.objects["Cube"].select = True
    bpy.ops.object.delete()

    SCALE_X = (
        SCALE_Y
    ) = SCALE_Z = 1e-3  # scale down from meters to reasonable blender units
    SCALE_Z *= user_configuration["coordinate_system"]["vertical_exaggeration"]

    _set_world_parameters()
    _add_wells(origin, user_configuration["wells"])
    _add_text_annotations(user_configuration["text_annotations"])
    _add_boundaries(user_configuration["boundary_boxes"])

    td_horizons = [
        TimeDependentHorizon(horizon, horizon_settings["depth"])
        for horizon, horizon_settings in user_configuration[
            "time_dependent_horizons"
        ].items()
    ]

    _add_camera_tracking(user_configuration["visual_settings"]["camera_track_point"])

    resolution = user_configuration["visual_settings"]["resolution"]

    _render_frames(
        width=resolution["width"],
        height=resolution["height"],
        static_horizons=_configure_static_horizons(
            user_configuration["static_horizons"]
        ),
        td_horizons=td_horizons,
    )
