#!/usr/bin/env python3
"""
Test complete skeleton in panda format
"""

import genesis as gs

def test_complete_skeleton():
    """Test complete skeleton with all body parts"""
    print("Initializing Genesis...")
    gs.init()
    
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, -3.0, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=45,
        ),
        show_viewer=True,
    )
    
    # Add ground plane
    plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    
    # Test complete skeleton
    print("Loading complete skeleton...")
    try:
        skeleton = scene.add_entity(gs.morphs.MJCF(
            file="/home/ez/Documents/Genesis/genesis_loco/skeleton/skeleton_complete_panda_format.xml"
        ))
        print("✓ Complete skeleton loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load complete skeleton: {e}")
        return False
    
    scene.build(n_envs=1)
    
    print("Model info:")
    print(f"  - Root position: {skeleton.get_pos()}")
    print(f"  - Number of joints: {len(skeleton.joints)}")
    print(f"  - Number of DOFs: {skeleton.n_dofs}")
    print(f"  - Joint names: {[joint.name for joint in skeleton.joints]}")
    
    print("Running simulation... Look for the complete skeleton!")
    print("You should see: pelvis, both legs, torso, head, both arms, and hands with fingers!")
    
    for i in range(1500):
        scene.step()
        
        if i % 300 == 0:
            pos = skeleton.get_pos()
            print(f"Step {i}: Root position: {pos}")
    
    print("Success!")
    return True

if __name__ == "__main__":
    test_complete_skeleton()