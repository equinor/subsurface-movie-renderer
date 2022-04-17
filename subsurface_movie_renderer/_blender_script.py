"""
This file is automatically executed directly through blender (since blender ships its own python).
It will not work outside of the blender provided Python distribution.
"""

import csv
import sys
import json
import math
from typing import List, Tuple, Union
from pathlib import Path

import numpy as np

# Blender specific packages which are available only in Blender shipped Python:
import bpy  # pylint: disable=import-error

# TODO: Remove these disables
# pylint: disable=redefined-outer-name, too-many-locals, too-many-statements, too-few-public-methods, too-many-arguments


def _set_world_parameters() -> None:
    bpy.data.worlds["World"].use_nodes = True

    inputs = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs
    inputs[0].default_value[:3] = (1, 1, 1)
    inputs[1].default_value = 1.0


def _add_text_annotations(annotations: List[dict]) -> None:

    blender_font = bpy.data.fonts.load("font.woff")

    txt_material = bpy.data.materials.new("text")
    txt_material.diffuse_color = [0, 0, 0, 1.0]

    for ann in annotations:
        for i, _ in enumerate(ann["rotation"]):
            ann["rotation"][i] *= math.pi / 180

        x, y, z = ann["location"]
        x *= SCALE_X
        y *= SCALE_Y
        z *= SCALE_Z

        bpy.ops.object.text_add(location=(x, y, z), rotation=ann["rotation"])
        txt = bpy.context.view_layer.objects.active
        txt.data.body = ann["label"]
        txt.data.font = blender_font
        txt.data.materials.append(txt_material)


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
        bpy.context.collection.objects.link(new_object)

        for i, polygon in enumerate(new_object.data.polygons):
            material = bpy.data.materials.new(box["name"] + "_mat" + str(i))

            import random  # pylint: disable=import-outside-toplevel

            # TODO: Make this variation deterministic
            alpha = random.uniform(
                max(0, box["alpha"] - 0.1), min(1, box["alpha"] + 0.1)
            )
            material.use_nodes = True
            material.blend_method = "HASHED"

            inputs = material.node_tree.nodes["Principled BSDF"].inputs
            inputs["Alpha"].default_value = alpha
            inputs["Base Color"].default_value = box["color"] + [alpha]

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

        bpy.context.collection.objects.link(objectdata)

        polyline = curvedata.splines.new("POLY")
        polyline.points.add(len(coordinates) - 1)
        for i, coordinate in enumerate(coordinates):
            polyline.points[i].co = coordinate + (1,)

        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]

        well_mat = bpy.data.materials.new(material_name)
        well_mat.diffuse_color = well["color"] + [well.get("alpha", 1.0)]

        bpy.data.objects[object_name].data.bevel_depth = 1e-2
        bpy.data.objects[object_name].data.fill_mode = "FULL"
        bpy.data.objects[object_name].data.materials.append(well_mat)


###################################


def define_surface_colorscale(colorscale: List[List[float]]) -> List:
    materials = []

    for i, color in enumerate(colorscale):
        mat = bpy.data.materials.new("mat_val" + str(i))
        alpha = i / 10.0 if i < 10 else 1.0

        mat.use_nodes = True
        mat.blend_method = "HASHED"

        inputs = mat.node_tree.nodes["Principled BSDF"].inputs
        inputs["Alpha"].default_value = alpha
        inputs["Base Color"].default_value = color[:3] + [alpha]
        inputs["Emission"].default_value = color[:3] + [alpha]
        inputs["Emission Strength"].default_value = 0.1 * alpha

        materials.append(mat)

    return materials


class Horizon:
    def __init__(
        self,
        X: np.array,
        Y: np.array,
        Z: np.array,
        horizon_name: str,
        alpha: Union[float, dict] = 1.0,
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
            red = 0.7 * (1 - i / n_materials)
            green = 0.7 * (1 - i / n_materials)
            blue = 0.7 * (1 - i / n_materials)

            mat = bpy.data.materials.new("top_ty_mat_val" + str(i))
            mat.use_nodes = True
            mat.blend_method = "HASHED"

            inputs = mat.node_tree.nodes["Principled BSDF"].inputs
            inputs["Alpha"].default_value = 1
            inputs["Base Color"].default_value = [red, green, blue, 1]

            self._top_ty_materials.append(mat)

    def update_alpha(self, t):
        if isinstance(self._alpha, (int, float)):
            alpha = self._alpha
        else:
            alpha = np.interp(t, list(self._alpha.keys()), list(self._alpha.values()))

        for mat in self._top_ty_materials:
            inputs = mat.node_tree.nodes["Principled BSDF"].inputs
            inputs["Alpha"].default_value = alpha
            inputs["Base Color"].default_value[3] = alpha

    def update_blender(self) -> None:

        X = self._X
        Y = self._Y
        Z = self._Z

        min_z = np.nanmin(np.nanmin(Z))
        max_z = np.nanmax(np.nanmax(Z))

        amp = 100.0 * (Z - min_z) / (max_z - min_z)
        amp[np.isnan(amp)] = 0

        Z *= -1.0

        # TODO: ALL CODE BELOW IS ~REUSED. =>> CONSOLIDATE

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
                            int(vertex2index[(N + 1) * i + j]),
                            int(vertex2index[(N + 1) * i + 1 + j]),
                            int(vertex2index[(N + 1) * (i + 1) + 1 + j]),
                            int(vertex2index[(N + 1) * (i + 1) + j]),
                        )
                    )

                    values.append(int(amp[i, j]))

        try:
            bpy.data.objects[self._horizon_name].select_set(True)
            bpy.ops.object.delete()
            bpy.data.objects[self._horizon_name].select_set(False)
        except KeyError:
            pass

        mesh = bpy.data.meshes.new(self._horizon_name)
        new_object = bpy.data.objects.new(self._horizon_name, mesh)
        new_object.location = bpy.context.scene.cursor.location
        bpy.context.collection.objects.link(new_object)
        mesh.from_pydata(verts, [], faces)
        mesh.update(calc_edges=True)

        bpy.context.view_layer.objects.active = bpy.data.objects[self._horizon_name]

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

        bpy.data.objects[self._horizon_name].select_set(True)
        bpy.ops.object.shade_smooth()
        bpy.data.objects[self._horizon_name].select_set(False)


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

            self._amp1 = data["AMP1"]
            self._amp2 = data["AMP2"]

            self.X = data["X"]
            self.Y = data["Y"]

            self._time_a, self._time_b = tuple(
                self._survey_times[file_index : file_index + 2]
            )

            self._AT = self._time_a + (self._time_b - self._time_a) * data["AT"] / 100.0

    def _get_values(self, time: float):
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
                            int(vertex2index[(N + 1) * i + j]),
                            int(vertex2index[(N + 1) * i + 1 + j]),
                            int(vertex2index[(N + 1) * (i + 1) + 1 + j]),
                            int(vertex2index[(N + 1) * (i + 1) + j]),
                        )
                    )

                    values.append(int(amp[i, j]))

        try:
            bpy.data.objects[self._horizon_name].select_set(True)
            bpy.ops.object.delete()
            bpy.data.objects[self._horizon_name].select_set(False)
        except KeyError:
            pass

        mesh = bpy.data.meshes.new(self._horizon_name)
        new_object = bpy.data.objects.new(self._horizon_name, mesh)
        new_object.location = bpy.context.scene.cursor.location

        bpy.context.collection.objects.link(new_object)
        mesh.from_pydata(verts, [], faces)
        mesh.update(calc_edges=True)

        bpy.context.view_layer.objects.active = bpy.data.objects[self._horizon_name]

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

        bpy.data.objects[self._horizon_name].select_set(True)
        bpy.ops.object.shade_smooth()
        bpy.data.objects[self._horizon_name].select_set(False)


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


class ParticleSystem:
    def __init__(
        self,
        origin,
        source_pos,
        source_diameter,
        max_distance,
        acceleration,
        fps,
        frame_start,
        frame_end,
        particle_density,
    ) -> None:

        location = (
            (source_pos[0] - origin[0]) * SCALE_X,
            (source_pos[1] - origin[1]) * SCALE_Y,
            (source_pos[2] - origin[2]) * SCALE_Z,
        )
        bpy.ops.mesh.primitive_plane_add(
            location=location, size=source_diameter * SCALE_X
        )

        obj = bpy.data.objects.get("Plane")
        obj.select_set(False)

        obj.modifiers.new("particles", type="PARTICLE_SYSTEM")
        particle_system = obj.particle_systems[0]

        settings = particle_system.settings
        settings.particle_size = 0.002
        settings.render_type = "OBJECT"
        settings.count = particle_density * (frame_end - frame_start)
        settings.lifetime = int(  # s = 0.5at^2
            math.sqrt(2 * max_distance * SCALE_Z / acceleration) * fps
        )
        settings.time_tweak = 25 / fps
        settings.frame_start = frame_start
        settings.frame_end = frame_end
        settings.instance_object = bpy.data.objects["Cube"]


def _add_camera_tracking(pos: Tuple[int, int, int]) -> None:
    empty = bpy.data.objects.new("Empty", None)
    empty.location = (pos[0] * SCALE_X, pos[1] * SCALE_Y, pos[2] * SCALE_Z)
    bpy.context.collection.objects.link(empty)

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

    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.view_settings.view_transform = "Standard"

    camera = bpy.data.objects["Camera"]

    if static_horizons is not None:
        for static_horizon in static_horizons:
            static_horizon.update_blender()

    with open("camera_coordinates.csv") as csvfile:
        for i, txyz in enumerate(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)):

            t, x, y, z = txyz

            if td_horizons is not None:
                for td_horizon in td_horizons:
                    td_horizon.update_blender(t)  # type: ignore[arg-type]

            if static_horizons is not None:
                for static_horizon in static_horizons:
                    static_horizon.update_alpha(t)

            x *= SCALE_X  # type: ignore[operator]
            y *= SCALE_Y  # type: ignore[operator]
            z *= SCALE_Z  # type: ignore[operator]

            camera.location.xyz = (x, y, z)

            scene_key = bpy.data.scenes.keys()[0]
            bpy.data.scenes[scene_key].render.filepath = (
                "image" + (6 - len(str(i))) * "0" + str(i)
            )

            bpy.context.scene.frame_set(i)

            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":

    user_configuration = json.loads(sys.argv[-1])

    origin = user_configuration["coordinate_system"]["origin"]

    materials = define_surface_colorscale(
        json.loads(Path("colorscale.json").read_text())
    )

    fps = user_configuration["visual_settings"]["fps"]
    movie_duration = user_configuration["visual_settings"]["movie_duration"]
    time_axis = [
        float(t) for t in user_configuration["visual_settings"]["camera_path"].keys()
    ]

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

    ACCELERATION = 0.1
    bpy.context.scene.gravity = (0, 0, ACCELERATION)
    bpy.context.scene.frame_end = int(fps * movie_duration)

    bpy.data.objects["Cube"].material_slots[0].material = materials[1]
    bpy.data.objects["Cube"].hide_render = True

    for i, particle_system in enumerate(user_configuration.get("particle_systems", [])):
        if i > 0:
            raise NotImplementedError("Multiple particle systems not implemented yet.")
        frame_start = int(
            fps
            * movie_duration
            * (particle_system["start_time"] - time_axis[0])
            / (time_axis[-1] - time_axis[0])
        )
        frame_end = int(fps * movie_duration)
        ParticleSystem(
            origin=origin,
            source_pos=particle_system["source_pos"],
            source_diameter=particle_system["source_diameter"],
            fps=fps,
            frame_start=frame_start,
            frame_end=frame_end,
            particle_density=particle_system["particle_density"],
            acceleration=ACCELERATION,
            max_distance=170,
        )

    _render_frames(
        width=resolution["width"],
        height=resolution["height"],
        static_horizons=_configure_static_horizons(
            user_configuration["static_horizons"]
        ),
        td_horizons=td_horizons,
    )
