"""Render a slowly turning head of a GLB with Blender (albedo only, no lighting).

Usage:
    blender -b --factory-startup -P tools/before_after/turn_head.py -- MODEL.glb OUT_DIR [FRAMES] [SIZE]

Writes OUT_DIR/f000.png … one PNG per frame with a transparent background. The
camera is orthographic and frames the top 13 % of the figure (the head) from
the front (+Z in glTF, -Y in Blender); the model yaws ±28° on a sine. Workbench
FLAT shading shows the texture as stored, so before/after renders differ only
in the texture.
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
glb, outdir = argv[0], argv[1]
frames = int(argv[2]) if len(argv) > 2 else 24
size = int(argv[3]) if len(argv) > 3 else 320

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
bpy.context.view_layer.update()
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
points = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
top = max(p.z for p in points)
height = top - min(p.z for p in points)
band = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices
        if (o.matrix_world @ v.co).z > top - .13 * height]
cx = sum(p.x for p in band) / len(band)
cy = sum(p.y for p in band) / len(band)
cz = top - .075 * height

pivot = bpy.data.objects.new('pivot', None)
bpy.context.scene.collection.objects.link(pivot)
pivot.location = (cx, cy, cz)
bpy.context.view_layer.update()
for o in [o for o in bpy.context.scene.objects if o.parent is None and o is not pivot]:
    o.parent = pivot
    o.matrix_parent_inverse = pivot.matrix_world.inverted()

camera_data = bpy.data.cameras.new('camera')
camera_data.type = 'ORTHO'
camera_data.ortho_scale = .17 * height
camera = bpy.data.objects.new('camera', camera_data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (cx, cy - 3 * height, cz)
camera.rotation_euler = (math.pi / 2, 0, 0)

scene = bpy.context.scene
scene.camera = camera
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'FLAT'
scene.display.shading.color_type = 'TEXTURE'
scene.render.resolution_x = scene.render.resolution_y = size
scene.render.film_transparent = True
scene.view_settings.view_transform = 'Standard'
for i in range(frames):
    pivot.rotation_euler = (0, 0, math.radians(28 * math.sin(2 * math.pi * i / frames)))
    scene.render.filepath = f'{outdir}/f{i:03d}.png'
    bpy.ops.render.render(write_still=True)
