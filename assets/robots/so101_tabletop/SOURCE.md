# SO101 tabletop asset provenance

This directory contains an adapted copy of the SO101 tabletop MuJoCo scene
from Vector OS Nano.

- Upstream: `https://github.com/VectorRobotics/vector-os-nano`
- Source checkout: `/home/liber/embodied_agent/vector-os-nano`
- Commit: `cd7029aad15d6510c956f504e57181cbb6651996`
- License: Apache License 2.0
- Scene source: `vector_os_nano/hardware/sim/so101_mujoco.xml`
- Mesh source: `vector_os_nano/hardware/urdf/meshes/`

Local changes:

- Changed the MJCF `meshdir` to the repository-local `meshes/` directory.
- Added SPDX, copyright, and modification notices.
- No dynamics, geometry, actuator, object, camera, or equality parameters were
  changed.

The robot meshes are attributed upstream to LeRobot SO-ARM100 assets under
Apache-2.0. Object mesh provenance should be re-audited before external asset
redistribution; this repository preserves the upstream Apache-2.0 notices and
records every copied file in `SHA256SUMS`.
