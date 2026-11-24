"""Cartesian tip tracking shared by SOFA training and replay."""

import numpy as np
from scipy.spatial.transform import Rotation


WRIST = slice(4, 6)  # Panda joints 5 and 6 steer the needle around the home pose.
MAX_ADVANCE = 0.004  # metres along the simulated shaft per policy tick
TOUCH_TOLERANCE = 0.0001  # geometric contact tolerance in metres
HOLD_TOLERANCE = 0.001  # allowed tracking error as the surface moves
MAX_DELTA = 0.025  # radians per policy tick
SUBSTEPS = 4
HORIZON = 100  # one four-second breathing cycle


class Pointing:
    def __init__(self, root):
        self.root = root
        self.robot = root.panda.control
        self.joint_frames = [frame for _, _, _, joint, frame in self.robot.links if joint is not None]

    def observe(self):
        hand = self.root.panda.hand.frame.position.value[0]
        rot = Rotation.from_quat(hand[3:])
        tip = self.root.panda.hand.tool.points.position.value[1]
        target = self.root.liver.target.point.position.value[0]
        # x/y are signed lateral errors in the tool frame; +z is the forward ray.
        return rot.inv().apply(target - tip)

    def command(self, action, error):
        """Steer with the wrist and translate the arm along the aligned shaft."""
        delta = np.clip(action[:2], -MAX_DELTA, MAX_DELTA)
        # Limit approach to the measured depth. Align laterally before closing the gap.
        depth = error[2]
        advance = np.clip(action[2], min(depth, 0.0), max(depth, 0.0))
        advance = np.clip(advance, -MAX_ADVANCE, MAX_ADVANCE)
        if np.linalg.norm(error[:2]) > 0.003 and advance > 0:
            advance = 0.0

        # Spatial Jacobian at the needle tip. Solve for translation along the
        # hand axis with zero extra wrist rotation, then add lateral steering.
        tip = self.root.panda.hand.tool.points.position.value[1]
        poses = np.array([frame.position.value[0] for frame in self.joint_frames])
        axes = Rotation.from_quat(poses[:, 3:]).apply([0, 0, 1])
        jac = np.vstack((np.cross(axes, tip - poses[:, :3]).T, axes.T))
        hand = self.root.panda.hand.frame.position.value[0]
        direction = Rotation.from_quat(hand[3:]).apply([0, 0, 1])
        motion = np.r_[direction * advance, np.zeros(3)]
        joint_step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), motion)
        joint_step[WRIST] += delta
        q = np.array(self.robot.joints.value)
        self.robot.apply(q + np.clip(joint_step, -MAX_DELTA, MAX_DELTA))

    def teacher(self, error):
        target = self.root.liver.target.point.position.value[0]
        hand = self.root.panda.hand.frame.position.value[0]
        inverse = Rotation.from_quat(hand[3:]).inv()
        columns = []
        for link in (self.root.panda.link5, self.root.panda.link6):
            pose = link.frame.position.value[0]
            axis = Rotation.from_quat(pose[3:]).apply([0, 0, 1])
            columns.append(-inverse.apply(np.cross(axis, target - pose[:3]))[:2])
        jac = np.column_stack(columns)
        steer = -0.25 * np.linalg.solve(jac.T @ jac + 0.0001 * np.eye(2), jac.T @ error[:2])
        return np.r_[steer, 0.85 * error[2]]


def load_policy(path):
    with np.load(path, allow_pickle=False) as saved:
        weights = saved["weights"]
        vertex = int(saved["vertex"])
    if weights.shape == (2, 2):
        # Earlier files store only wrist steering; retain it and use the
        # demonstration controller's axial gain for the missing action.
        extended = np.diag([0.0, 0.0, 0.85])
        extended[:2, :2] = weights
        weights = extended
    if weights.shape != (3, 3) or not np.isfinite(weights).all():
        raise ValueError("Invalid pointing policy weights")
    return weights, None if vertex == -1 else vertex
