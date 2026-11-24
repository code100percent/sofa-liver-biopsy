# Franka Panda liver-targeting scene

This SOFA scene places a seven-joint Panda above a deformable liver phantom. A teal hub holds the needle in the gripper. A green marker identifies the target on the liver surface and follows that point as the tissue moves.

## Run

From the repository root on Ubuntu 24.04 x86-64:

```bash
./setup.sh
./run.sh
```

The setup script downloads the official SOFA 25.12.00 Linux/Python 3.12 distribution, verifies its SHA-256 checksum, and installs it in the ignored project-local `.sofa/` directory. Rerunning setup reuses the installation. The launcher uses the system Python 3.12, NumPy, and SciPy. The Panda model and meshes are bundled in `model/`. Run `./run.sh --replay` to watch the included trained policy.

The simulation animates at launch. To change the pose, edit the seven `joints` values on `panda/control` in the scene inspector while the simulation is running. The controller also accepts scene key events: 1–7 select a joint, + or - moves it one degree, and H returns home. Joint commands are in radians and are clipped to the model limits.

Run the integration check without a desktop display:

```bash
./run.sh --check
```

## Simulation state

The arm uses forward kinematics from the Panda link frames. Each visual part moves with its own link. The tool endpoints share a rigid mapping with the hand. The liver uses the SOFA tutorial tetrahedral mesh, a corotational elastic model, and moving spring supports with a four-second sinusoidal cycle. Mesh scale, stiffness, mass, and breathing parameters are illustrative phantom settings.

Training and replay read these SofaPython3 fields. Coordinates use the world frame in metres, with Z up.

| Field | Meaning |
| --- | --- |
| `root.panda.control.joints.value` | Seven commanded joint angles |
| `root.panda.hand.tool.points.position.value` | Shaft base and tip positions, in that order |
| `root.liver.target.point.position.value[0]` | Current surface target position |

## Train the needle tip to touch a surface target

Run these commands from this project directory:

```bash
./run.sh --train
./run.sh --replay
./run.sh --train --eval-only --seed 42 --eval-episodes 10
```

Training uses the green target on the breathing phantom by default. To pick another point, supply its initial world coordinates in metres. The script prints the nearest surface vertex and its coordinates, attaches the marker there, and saves the selection with the policy. You can also specify an exact zero-based surface vertex with `--target-vertex`.

```bash
./run.sh --train --target-xyz 0.53 0.01 0.29 --checkpoint runs/custom.npz
./run.sh --replay runs/custom.npz
```

Each four-second rollout starts with a randomized wrist pose. At 25 Hz the policy observes the three-dimensional tip-to-target displacement in the hand frame. Two wrist commands correct lateral error. Once the forward approach is aligned within 3 mm laterally, an additional command moves the arm along the shaft. A seven-joint Jacobian converts this translation to joint motion. Forward travel per tick is capped at 4 mm and at the remaining measured depth.

The default training run clones eight demonstrations, then refines the policy over three generations of reward-based search. The reward uses actual tip-to-marker distance plus a bonus for reaching and holding the target. Evaluation samples at 100 Hz. During the final second the tip must reach within 0.1 mm of the surface vertex and stay within 1 mm throughout. These thresholds measure geometric contact on the simulated moving surface. The model does not compute tissue puncture or contact force.

The default checkpoint is `runs/touch.npz`. To replay previously saved targets, pass the original checkpoint path, for example `./run.sh --replay runs/custom.npz`. For older two-output checkpoints the loader retains the learned wrist commands and supplies the demonstration controller's depth gain.

The liver mesh comes from the [SOFA distribution](https://github.com/sofa-framework/sofa/releases/tag/v25.12.00). The Panda frames and visual meshes come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/eea4c7431b63da8278b0ff31638e0b0e941cec0e/franka_emika_panda) and retain the license in [model/LICENSE](model/LICENSE).
