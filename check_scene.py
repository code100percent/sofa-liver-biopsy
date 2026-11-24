"""Exercise the initialized SOFA scene without creating an OpenGL window."""

import numpy as np
import Sofa.Core
import Sofa.Simulation
from scene import createScene


def main():
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.initRoot(root)
    tool = root.panda.hand.tool.points
    target = root.liver.target.point
    start = np.array(tool.position.value)
    initial_target = np.array(target.position.value[0])
    q = np.array(root.panda.control.joints.value)
    q[0] += 0.12
    root.panda.control.apply(q)
    for _ in range(100):
        Sofa.Simulation.animate(root, root.dt.value)
    current = np.array(tool.position.value)
    hand = np.array(root.panda.hand.frame.position.value[0])
    from scipy.spatial.transform import Rotation
    from panda import TIP_Z, SHAFT_START
    axis = Rotation.from_quat(hand[3:]).apply([0, 0, 1])
    np.testing.assert_allclose(current, hand[:3] + np.outer([SHAFT_START, TIP_Z], axis), atol=1e-7)
    assert np.linalg.norm(current[1] - start[1]) > 0.02
    tissue = np.array(root.liver.tissue.position.value)
    assert np.isfinite(tissue).all()
    target_pos = np.array(target.position.value[0])
    assert np.min(np.linalg.norm(tissue - target_pos, axis=1)) < 1e-8
    movement = np.linalg.norm(target_pos - initial_target)
    assert 0.0001 < movement < 0.02, movement
    print(f"SOFA check passed: {len(tissue)} liver nodes, tool follows hand, "
          f"target moved {1000 * movement:.2f} mm in 1 simulated second.")


if __name__ == "__main__":
    main()
