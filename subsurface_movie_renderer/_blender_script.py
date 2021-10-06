"""
This file is automatically executed directly through blender (since blender ships its own python).
It will not work outside of the blender provided Python distribution.
"""

import csv
import sys
import json
import math
from typing import List, Tuple

# Blender specific packages which are available only in Blender shipped Python:
import bpy  # pylint: disable=import-error
from mathutils import Vector, Color  # pylint: disable=import-error


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

            # TODO: Make this variation deterministic  # pylint: disable=fixme
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


def _add_camera_tracking(pos: Tuple[int, int, int]) -> None:
    empty = bpy.data.objects.new("Empty", None)
    empty.location = Vector((pos[0] * SCALE_X, pos[1] * SCALE_Y, pos[2] * SCALE_Z))
    bpy.context.scene.objects.link(empty)
    bpy.context.scene.update()

    track_to = bpy.data.objects["Camera"].constraints.new("TRACK_TO")
    track_to.target = empty
    track_to.track_axis = "TRACK_NEGATIVE_Z"
    track_to.up_axis = "UP_Y"


def _render_frames(width: int, height: int) -> None:
    scene_key = bpy.data.scenes.keys()[0]
    scene = bpy.data.scenes[scene_key]

    scene.camera = bpy.data.objects["Camera"]

    bpy.context.scene.render.resolution_x = 2 * width
    bpy.context.scene.render.resolution_y = 2 * height

    camera = bpy.data.objects["Camera"]
    lamp = bpy.data.objects["Lamp"]
    lamp.data.energy = 0.0

    with open("camera_coordinates.csv") as csvfile:
        for i, txyz in enumerate(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)):
            x, y, z = txyz[1:]
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

    # Delete default blender provided cube and lamp:
    bpy.data.objects["Cube"].select = True
    bpy.ops.object.delete()

    SCALE_X = (
        SCALE_Y
    ) = SCALE_Z = 1e-3  # scale down from meters to reasonable blender units
    SCALE_Z *= user_configuration["coordinate_system"]["vertical_exaggeration"]

    _set_world_parameters()
    _add_wells(
        user_configuration["coordinate_system"]["origin"], user_configuration["wells"]
    )
    _add_text_annotations(user_configuration["text_annotations"])
    _add_boundaries(user_configuration["boundary_boxes"])

    _add_camera_tracking(user_configuration["visual_settings"]["camera_track_point"])

    resolution = user_configuration["visual_settings"]["resolution"]
    _render_frames(width=resolution["width"], height=resolution["height"])
