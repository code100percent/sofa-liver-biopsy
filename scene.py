"""SOFA scene for pointing a hand-mounted needle at a moving liver phantom."""

import os
from pathlib import Path
import sys

import numpy as np
import Sofa.Core

# runSofa loads a scene by filename, including when launched from a different directory.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from panda import Panda


class Breathing(Sofa.Core.Controller):
    def __init__(self, anchors, rest):
        super().__init__()
        self.name = "breathing"
        self.anchors = anchors
        self.rest = rest
        # Different prescribed displacements at the support points deform the volume.
        self.weights = 0.5 + 0.5 * (rest[:, 0] - rest[:, 0].min()) / np.ptp(rest[:, 0])

    def onAnimateBeginEvent(self, event):
        t = self.getContext().getTime()
        displacement = 0.006 * np.sin(2 * np.pi * t / 4)
        positions = self.rest.copy()
        positions[:, 2] += displacement * self.weights
        self.anchors.position.value = positions.tolist()


def createScene(root):
    return build_scene(root)


def build_scene(root, *, target_vertex=None, target_xyz=None, visuals=True):
    root.addObject("RequiredPlugin", pluginName=[
        "Sofa.Component.AnimationLoop", "Sofa.Component.Collision.Geometry",
        "Sofa.Component.IO.Mesh", "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mapping.Linear", "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.Mass", "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.FEM.Elastic", "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer", "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual", "Sofa.Component.Setting", "Sofa.GL.Component.Rendering3D",
    ])
    root.dt = 0.01
    root.gravity = [0, 0, 0]
    root.addObject("DefaultAnimationLoop")
    root.addObject("DefaultVisualManagerLoop")
    root.addObject("VisualStyle", displayFlags="showVisualModels showCollisionModels hideBehaviorModels")
    root.addObject("BackgroundSetting", color=[0.14, 0.17, 0.20, 1])
    # Keep world Z upright and frame both the base and liver in the initial view.
    root.addObject("InteractiveCamera", name="camera", position=[0.78, -1.45, 0.82],
                   orientation=[0.59558814, 0.10543755, 0.13881838, 0.7841474],
                   distance=1.60259)
    robot = root.addChild("panda")
    robot.addObject(Panda(robot, visuals=visuals))
    add_liver(root, target_vertex, target_xyz)
    return root


def add_liver(root, target_vertex, target_xyz):
    node = root.addChild("liver")
    mesh_file = Path(os.environ["SOFA_ROOT"]) / "share/sofa/mesh/liver.msh"
    loader = node.addObject("MeshGmshLoader", name="mesh", filename=str(mesh_file))
    loader.init()
    raw = np.array(loader.position.value)
    tets = np.array(loader.tetrahedra.value, dtype=int)
    # Normalize the tutorial mesh to a 20 cm phantom near the home tool axis.
    points = (raw - (raw.min(0) + raw.max(0)) / 2) * (0.20 / np.ptp(raw, axis=0).max())
    top = int(np.argmax(points[:, 2]))
    points += np.array([0.55, 0.02, 0.30]) - points[top]
    target_index = top if target_vertex is None else target_vertex
    node.addObject("EulerImplicitSolver", rayleighStiffness=0.02, rayleighMass=0.2)
    node.addObject("CGLinearSolver", iterations=100, tolerance=1e-10, threshold=1e-14)
    node.addObject("TetrahedronSetTopologyContainer", name="topology",
                   position=points.tolist(), tetrahedra=tets.tolist())
    node.addObject("MechanicalObject", name="tissue", template="Vec3d", position=points.tolist())
    node.addObject("UniformMass", totalMass=1.0)
    node.addObject("TetrahedronFEMForceField", method="large", youngModulus=4000, poissonRatio=0.35)

    indices = np.flatnonzero(points[:, 2] <= np.quantile(points[:, 2], 0.20))
    supports = root.addChild("supports")
    anchors = supports.addObject("MechanicalObject", name="points", position=points[indices].tolist())
    node.addObject("RestShapeSpringsForceField", points=indices.tolist(),
                   external_rest_shape=anchors.getLinkPath(), external_points=list(range(len(indices))),
                   stiffness=1500)
    root.addObject(Breathing(anchors, points[indices]))

    # Orient exposed tetrahedron faces outward and discard shared interior faces.
    faces = {}
    for tet in tets:
        for i, opposite in enumerate(tet):
            face = np.delete(tet, i)
            key = tuple(sorted(face))
            if key in faces:
                del faces[key]
                continue
            a, b, c = points[face]
            if np.dot(np.cross(b - a, c - a), points[opposite] - a) > 0:
                face = face[::-1]
            faces[key] = face.tolist()
    surface_ids = np.unique(list(faces.values()))
    if target_xyz is not None:
        distances = np.linalg.norm(points[surface_ids] - target_xyz, axis=1)
        target_index = int(surface_ids[np.argmin(distances)])
    if target_index not in surface_ids:
        raise ValueError(f"Target vertex {target_index} is not on the liver surface")
    surface = node.addChild("skin")
    surface.addObject("OglModel", name="visual", position=points.tolist(),
                      triangles=list(faces.values()), color="0.55 0.12 0.10 1")
    surface.addObject("IdentityMapping", input="@../tissue", output="@visual")

    target = node.addChild("target")
    target.addObject("MechanicalObject", name="point", position=[points[target_index].tolist()])
    target.addObject("SubsetMapping", indices=[target_index], input="@../tissue", output="@point")
    # Sphere visual follows the mapped tissue point.
    target.addObject("SphereCollisionModel", radius=0.003, moving=True, simulated=False,
                     color="0.15 0.95 0.55 1", group=1)
