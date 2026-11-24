"""Kinematic Franka Panda in SOFA, using the workspace's Menagerie link frames."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import Sofa.Core


MODEL = Path(__file__).resolve().parent / "model"
# Tool coordinates in metres along hand-local +Z (towards the fingers).
HUB_START, SHAFT_START, TIP_Z = 0.080, 0.118, 0.225


def spindle(rings, segments=20):
    """Make a closed surface of revolution from (height, radius) pairs."""
    angles = np.arange(segments) * 2 * np.pi / segments
    vertices = [[r * np.cos(a), r * np.sin(a), z] for z, r in rings for a in angles]
    triangles = []
    for k in range(len(rings) - 1):
        for i in range(segments):
            a, b = k * segments + i, k * segments + (i + 1) % segments
            triangles.extend([[a, b, a + segments], [b, b + segments, a + segments]])
    for ring, reverse in [(0, True), (len(rings) - 1, False)]:
        offset = ring * segments
        for i in range(1, segments - 1):
            face = [offset, offset + i, offset + i + 1]
            triangles.append(face[::-1] if reverse else face)
    return vertices, triangles


class Panda(Sofa.Core.Controller):
    def __init__(self, parent, visuals=True):
        super().__init__()
        self.name = "control"
        xml = ET.parse(MODEL / "panda.xml").getroot()
        self.home = np.fromstring(xml.find("keyframe/key").get("qpos"), sep=" ")[:7]
        self.addData(name="joints", type="vector<double>", value=self.home.tolist(),
                     help="Commanded joint positions in radians")
        self.selected = 0
        self.links = []
        self.limits = []
        palette = {m.get("name"): m.get("rgba") for m in xml.findall("asset/material")}
        base = xml.find("worldbody/body")
        parents = {child: body for body in base.iter("body") for child in body.findall("body")}
        indices = {}
        for body in base.iter("body"):
            name = body.get("name")
            node = parent.addChild(name)
            frame = node.addObject("MechanicalObject", template="Rigid3d", name="frame",
                                   position=[[0, 0, 0, 0, 0, 0, 1]])
            pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            wxyz = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
            joint = body.find("joint")
            rotation = Rotation.from_quat(np.roll(wxyz, -1))
            if name in ("left_finger", "right_finger"):
                pos += rotation.apply([0, 0.009, 0])
            joint_index = None
            if joint is not None and joint.get("name").startswith("joint"):
                joint_index = int(joint.get("name")[5:]) - 1
                self.limits.append(np.fromstring(joint.get("range", "-2.8973 2.8973"), sep=" "))
            indices[body] = len(self.links)
            self.links.append((indices.get(parents.get(body)), pos, rotation,
                               joint_index, frame))
            visual_geoms = body.findall("geom[@class='visual']") if visuals else ()
            for geom in visual_geoms:
                mesh = geom.get("mesh")
                part = node.addChild(mesh + "_visual")
                part.addObject("MeshOBJLoader", name="mesh", filename=str(MODEL / "assets" / (mesh + ".obj")),
                               handleSeams=False, loadMaterial=False)
                part.addObject("OglModel", name="surface", src="@mesh", color=palette[geom.get("material")])
                part.addObject("RigidMapping", input=frame.getLinkPath(), output="@surface")
            if name == "hand":
                self._mount_tool(node, frame)
        self.limits = np.array(self.limits)
        self.apply(self.home)

    def _mount_tool(self, hand, frame):
        for name, rings, color in [
            ("hub", [(HUB_START, 0.009), (0.103, 0.009), (SHAFT_START, 0.004)], "0.05 0.58 0.68 1"),
            ("needle", [(SHAFT_START, 0.0013), (TIP_Z - 0.005, 0.0013), (TIP_Z, 0.00005)], "0.82 0.86 0.90 1"),
        ]:
            part = hand.addChild(name)
            points, faces = spindle(rings)
            part.addObject("OglModel", name="surface", position=points, triangles=faces, color=color)
            part.addObject("RigidMapping", input=frame.getLinkPath(), output="@surface")
        tool = hand.addChild("tool")
        tool.addObject("MechanicalObject", name="points", template="Vec3d",
                       position=[[0, 0, SHAFT_START], [0, 0, TIP_Z]])
        tool.addObject("RigidMapping", input=frame.getLinkPath(), output="@points")

    def apply(self, joints):
        q = np.asarray(joints, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite Panda joint angles in radians")
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        self.joints.value = q.tolist()
        poses = []
        for parent, offset, rotation, joint, frame in self.links:
            if joint is not None:
                rotation = rotation * Rotation.from_rotvec([0, 0, q[joint]])
            if parent is not None:
                parent_pos, parent_rot = poses[parent]
                offset = parent_pos + parent_rot.apply(offset)
                rotation = parent_rot * rotation
            poses.append((offset, rotation))
            frame.position.value = [np.r_[offset, rotation.as_quat()].tolist()]

    def onAnimateBeginEvent(self, event):
        self.apply(self.joints.value)

    def onKeypressedEvent(self, event):
        key = event["key"]
        if key in tuple("1234567"):
            self.selected = int(key) - 1
        elif key in ("+", "=", "-"):
            q = np.array(self.joints.value)
            q[self.selected] += np.deg2rad(1 if key != "-" else -1)
            self.apply(q)
        elif key.lower() == "h":
            self.apply(self.home)
