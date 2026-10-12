# OneRobotics A1 Asset Attribution

## Robot Models

OneRobotics A1 robot models for integration with Genesis.

Included configurations:

- A1 independent right arm: 7 DoF
- A1 independent left arm: 7 DoF
- A1 bimanual robot with stand: 14 DoF

The models do not include grippers.

## Source

Original repository:
https://github.com/katazen/onerobot_h1

Asset source revision:
3db6f614c8443bd9396fcd9304cf64a519f98f2a

## License

OneRobotics A1 robot assets © 2026 OneRobotics.

Licensed under Creative Commons Attribution 4.0
International (CC BY 4.0).

https://creativecommons.org/licenses/by/4.0/

## Modifications

All three URDF files, the STL meshes, and model_parameters.yaml are copied unchanged from the pinned OneRobotics source revision:

3db6f614c8443bd9396fcd9304cf64a519f98f2a

Before that source revision was published, the bimanual URDF had undergone non-physical text changes to make mesh paths portable, normalize line endings, and remove exporter contact information.

Those upstream changes did not modify the robot's physical parameters or geometry.

No additional changes to the robot model files were made for this Genesis contribution.

Only the asset directory placement and accompanying documentation were added.

## Known Model Limitations

The independent-arm models retain the source Link7 temporary flange mass and inertia values.

The bimanual model retains the original zero effort and velocity placeholders.

The confirmed actuator parameters are documented in model_parameters.yaml.

The bimanual stand model comes from a separate CAD revision and is not synthesized from the independent-arm models.
