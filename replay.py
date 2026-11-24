"""SOFA viewer scene for a trained pointing policy."""

import os
from pathlib import Path
import sys

import Sofa.Core

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pointing import Pointing, SUBSTEPS, WRIST, load_policy
from scene import build_scene


class Replay(Sofa.Core.Controller):
    def __init__(self, root, weights):
        super().__init__()
        self.name = "pointing_policy"
        self.task = Pointing(root)
        self.weights = weights
        self.tick = 0

    def onAnimateBeginEvent(self, event):
        if self.tick == 0:
            q = self.task.robot.home.copy()
            q[WRIST] += [0.25, -0.20]
            self.task.robot.apply(q)
        elif (self.tick - 1) % SUBSTEPS == 0:
            error = self.task.observe()
            self.task.command(self.weights @ error, error)
        self.tick += 1

    def reset(self):
        self.tick = 0


def createScene(root):
    weights, vertex = load_policy(os.environ["POINTING_CHECKPOINT"])
    build_scene(root, target_vertex=vertex)
    root.addObject(Replay(root, weights))
    return root
