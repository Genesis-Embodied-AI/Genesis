# flake8: noqa

import bpy
import os


def delete_nonresult(bpy):
    objects = bpy.data.objects  # use data.objects instead of context.scene.objects
    if len(objects) <= 1:
        return

    try:
        # earlier than blender <2.8
        objects[0].select = False  # keep the first object
        for other in objects[1:]:  # remove all other objects
            other.select = True
        bpy.ops.object.delete()
        objects[0].select = True
    except AttributeError:
        # blender 2.8 changed this
        objects[0].select_set(False)
        for other in objects[1:]:
            other.select_set(True)
        bpy.ops.object.delete()
        objects[0].select_set(True)


if __name__ == '__main__':
    # clear scene of default box
    bpy.ops.wm.read_homefile()
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except BaseException:
        pass
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=True)

    # get temporary files from templated locations
    mesh_pre = $MESH_PRE
    mesh_post = os.path.abspath(r'$MESH_POST')

    for filename in mesh_pre:  # use data.objects instead of context.scene.objects
        bpy.ops.import_mesh.stl(filepath=os.path.abspath(filename))

    mesh = bpy.data.objects[0]
    # Make sure mesh is the active object
    try:
        # earlier than blender <2.8
        bpy.context.scene.objects.active = mesh
    except AttributeError:
        # blender 2.8 changed this
        bpy.context.view_layer.objects.active = mesh

    for other in bpy.data.objects[1:]:
        # add boolean modifier
        mod = mesh.modifiers.new('boolean', 'BOOLEAN')
        mod.object = other
        mod.operation = '$OPERATION'
        # used mod.name instead of hard-coded "boolean"
        bpy.ops.object.modifier_apply(modifier=mod.name)

    delete_nonresult(bpy)
    bpy.ops.export_mesh.stl(
        filepath=mesh_post,
        use_mesh_modifiers=True)
