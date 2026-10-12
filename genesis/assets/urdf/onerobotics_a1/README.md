# OneRobotics A1 2026 source model

This directory contains two distinct OneRobotics A1 CAD revisions: the current
independent right- and left-arm source models, and a hardware-team-supplied
whole-robot bimanual stand model. The revisions are published side by side and
are not mixed or synthesized from one another.

The independent-arm source files were imported without geometric or inertial
edits from the OneRobotics SDK asset bundle whose SHA-256 is:

```text
9dfa9f0e1b7fa8d05b12b47dd67ab0484bc376b4db60ce541ee832455a0be8c9
```

## Published contents

- `a1_r.urdf`: independent right-arm 7-DoF source model;
- `a1_l.urdf`: independent left-arm 7-DoF source model;
- `meshes/a1_r/` and `meshes/a1_l/`: eight referenced STL files per arm;
- `bimanual_stand/a1_bimanual_stand.urdf`: complete stand with two 7-DoF arms;
- `bimanual_stand/meshes/`: the 17 meshes referenced by that whole-robot model;
- `model_parameters.yaml`: shared model and simulation-control metadata.

The 36 robot-model files and `model_parameters.yaml` are Copyright 2026
OneRobotics and are distributed under CC BY 4.0 as declared in the
repository's `ASSET_LICENSE_STATUS.md`. Neither CAD revision includes a
gripper.

## Isaac Lab review set

`SHA256SUMS` freezes the exact right-arm, bimanual-stand, and shared parameter
files consumed by the current Isaac Lab contribution. Verify the review set
from this directory with:

```bash
sha256sum --check SHA256SUMS
```

The checksum file identifies review-stage source bytes. It does not imply
NVIDIA legal approval or prescribe the final asset-hosting mechanism.

## Bimanual stand provenance

The delivered bimanual source was repaired by the hardware team for coordinate
frame consistency without changing world-space geometry or inertial behavior.
The hashes below make the reviewed lineage explicit:

- raw baseline before coordinate-frame repair:
  `29ee4a14696c3827a0ed2611b1d32fb27d5204a89cae07eee3e30a02085b4ecb`;
- hardware-team source after that repair:
  `3ae0bc193bdaa3aed205655e03fd20584a594d5e76e7f9cf6511ab82b761486f`;
- publication candidate in this directory:
  `4719154dd2395498a646f60a468ccdd3c72e9f9754478c4df74eb495e605ef9d`.

The publication candidate changes only non-physical text: mesh URIs use the
portable `./meshes/` directory, the MuJoCo compiler uses `meshdir="meshes"`,
and the generated header omits the exporter's contact address. No mass, center
of mass, inertia, joint transform, axis, or joint limit was changed. The
published text also uses repository-normalized line endings, so its byte hash
intentionally differs from the delivered source.

## Hardware overlay

The URDFs contain the model geometry, link inertias, joint transforms, and
position limits. The bimanual source retains the delivered raw
`effort="0"`/`velocity="0"` placeholders on all 14 revolute joints. These zero
values are not actuator ratings: the actuator and control values confirmed by
OneRobotics are recorded in `model_parameters.yaml` and must overlay each 7-DoF
arm at integration time rather than rewriting the source URDF.

The 4340 actuator is used for joints 1--3 with a 48.19 reduction ratio. The
4310 actuator is used for joints 4--7 with a 10:1 reduction ratio. Both use a
motor-side rotor inertia of `2.193e-5 kg m^2`; the joint-side armature is the
rotor inertia multiplied by the square of the reduction ratio.

The confirmed joint-output rated/peak torques are `15.0/26.859 N m` for
joints 1--3 and `3.0/5.975 N m` for joints 4--7. The simulation uses the peak
values as its static solver saturation limits and retains the rated values as
actuator metadata. A peak-duration transition and thermal derating are not
modeled; they are not required to apply the supplied peak value as a simple
hard torque bound.

## Known source limitation

The two independent-arm URDFs explicitly identify the Link7 mass (`0.20 kg`)
and inertia as a temporary v1 flange placeholder. The values are retained
exactly as supplied; they are not presented as measured bare-flange parameters.
This limitation belongs to that CAD revision and is not applied to the distinct
bimanual stand revision.
