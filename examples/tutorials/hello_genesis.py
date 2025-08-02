import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene()

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    # gs.morphs.URDF(
    #     file='urdf/panda_bullet/panda.urdf',
    #     fixed=True,
    # ),
    # gs.morphs.MJCF(file="/home/ez/Documents/Genesis/genesis_loco/skeleton/skeleton_torque.xml"),
    gs.morphs.MJCF(file='/home/ez/Documents/Genesis/genesis_loco/skeleton/skeleton_restructured_panda_format.xml'),
)

scene.build()
for i in range(1000):
    scene.step()
