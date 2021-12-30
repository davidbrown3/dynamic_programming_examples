import os
import bpy
import sys
import json
import numpy as np
import bmesh
import shutil
import tempfile
# blender -b  -P ddpy/examples/solutions/render_cart_pole.py -- cartpole.json

argv = sys.argv
argv = argv[argv.index('--') + 1:]

directory = os.path.dirname(os.path.realpath(__file__))

file = argv[0]
with open(os.path.join(directory, file), 'r') as f:
    data = json.load(f)

positions = data['position']
angles = data['angle']


pole_length = 2.5
position = 0.0
angle = 0.0
x_camera = 25
y_camera = np.median(positions)
y_range = 30

# Add a sun lamp above the grid.
bpy.ops.object.light_add(type='POINT', radius=1.0, location=(0.0, 0.0, 20.0))

# Add cart
bpy.ops.mesh.primitive_cube_add(
    location=(0, 0, 0),
    size=0.5
)
bpy.context.object.name = 'Cart'
cart_material = bpy.data.materials.new(name='Material')
cart_material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
bpy.data.objects['Cart'].data.materials.append(cart_material)
bpy.data.objects['Cart'].active_material = cart_material

# Add sphere
bpy.ops.mesh.primitive_uv_sphere_add(
    location=(0, 0, 0),
    radius=0.25
)
bpy.context.object.name = 'Sphere'
sphere_material = bpy.data.materials.new(name='Material')
sphere_material.diffuse_color = (0.05, 0.05, 0.05, 1.0)
bpy.data.objects['Sphere'].data.materials.append(sphere_material)
bpy.data.objects['Sphere'].active_material = sphere_material

# Add pole
bpy.ops.mesh.primitive_cylinder_add(
    location=(0, 0, 0),
    depth=pole_length,
    radius=0.05
)
bpy.context.object.scale = (0.1, 1, 1)
bpy.context.object.name = 'Pole'
pole_material = bpy.data.materials.new(name='Material')
pole_material.diffuse_color = (0.1, 0.1, 0.1, 1.0)
bpy.data.objects['Pole'].data.materials.append(pole_material)
bpy.data.objects['Pole'].active_material = pole_material

mesh = bpy.data.meshes.new("Mesh")  # add a new mesh
obj = bpy.data.objects.new("GridLines", mesh)  # add a new object using the mesh
scene = bpy.context.scene
bpy.context.collection.objects.link(obj)  # put the object into the scene (link)
bpy.context.view_layer.objects.active = obj   # set as the active object in the scene

# Adding floor
bpy.ops.mesh.primitive_plane_add(
    size=y_range,
    location=(0, 0, 0)
)
bpy.context.object.name = 'Floor'
bpy.context.object.scale = (0.01, 1, 1)

# White background
bpy.context.scene.world.color = (0, 0, 0)

# Removing default object
bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)

frame_num = 0

bpy.ops.object.select_all(action='DESELECT')

bpy.data.objects['Camera'].location = (
    x_camera,
    y_camera,
    5.0
)
bpy.data.objects['Camera'].rotation_euler[0] = np.deg2rad(-100)
bpy.data.objects['Camera'].rotation_euler[1] = np.pi
bpy.data.objects['Camera'].rotation_euler[2] = -np.pi/2

bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.color_mode = 'RGBA'

for position, angle in zip(positions[::4], angles[::4]):  # 25fps

    bpy.context.scene.frame_set(frame_num)

    bpy.data.objects['Sphere'].location = (
        0.26,
        position + pole_length * np.sin(-angle),
        pole_length * np.cos(-angle)
    )

    bpy.data.objects['Pole'].location = (
        0.26,
        position + pole_length/2 * np.sin(-angle),
        pole_length / 2 * np.cos(-angle)
    )

    bpy.data.objects['Pole'].rotation_euler[0] = angle

    bpy.data.objects['Cart'].location = (
        0,
        position,
        0
    )

    bpy.data.objects['Floor'].keyframe_insert(data_path='location', index=-1)
    bpy.data.objects['Camera'].keyframe_insert(data_path='location', index=-1)
    bpy.data.objects['Sphere'].keyframe_insert(data_path='location', index=-1)
    bpy.data.objects['Pole'].keyframe_insert(data_path='location', index=-1)
    bpy.data.objects['Pole'].keyframe_insert(data_path='rotation_euler', index=-1)
    bpy.data.objects['Cart'].keyframe_insert(data_path='location', index=-1)

    frame_num += 1

bpy.context.scene.frame_end = frame_num

bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cart'].select_set(True)

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        space = area.spaces.active
        if space.type == 'VIEW_3D':
            space.shading.type = 'SOLID'

bpy.data.objects['GridLines'].show_all_edges = True

rnd = bpy.data.scenes['Scene'].render
rnd.fps = 50
rnd.image_settings.file_format = 'PNG'
rnd.resolution_x = 1280
rnd.resolution_y = 550
rnd.resolution_percentage = 100

with tempfile.TemporaryDirectory() as tmpdirname:

    rnd.filepath = os.path.join(tmpdirname, 'frame')

    for scene in bpy.data.scenes:
        scene.render.engine = 'BLENDER_WORKBENCH'

    bpy.ops.render.render(animation=True)

    os.system(f'ffmpeg -f lavfi -i color=ffffff -i {tmpdirname}/frame%0004d.png render_cartpole.gif -filter_complex "[0][1]scale2ref[bg][gif];[bg]setsar=1[bg];[bg][gif]overlay=shortest=1"')
