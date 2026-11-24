"""Train a simulated Panda to bring the needle tip to a breathing liver surface marker."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import Sofa.Core
import Sofa.Simulation

from pointing import HOLD_TOLERANCE, HORIZON, SUBSTEPS, TOUCH_TOLERANCE, WRIST, Pointing, load_policy
from scene import build_scene


def rollout(task, offset, weights, samples=None):
    root = task.root
    Sofa.Simulation.reset(root)
    q = task.robot.home.copy()
    q[WRIST] += offset
    task.robot.apply(q)
    Sofa.Simulation.animate(root, root.dt.value)
    distances = []
    total_reward = 0.0
    for _ in range(HORIZON):
        obs = task.observe()
        if samples is None:
            delta = weights @ obs
        else:
            delta = task.teacher(obs)
            samples.append((obs.copy(), delta))
        task.command(delta, obs)
        for _ in range(SUBSTEPS):
            Sofa.Simulation.animate(root, root.dt.value)
            distance = np.linalg.norm(task.observe())
            distances.append(distance)
            if not np.isfinite(distance):
                return -1000., False, np.inf
        total_reward -= 100 * distance

    # Measure tip-to-vertex distance at every physics frame of the final second.
    hold = distances[-(HORIZON * SUBSTEPS // 4):]
    worst = max(hold)
    held = min(hold) <= TOUCH_TOLERANCE and worst <= HOLD_TOLERANCE
    return total_reward + (100.0 if held else 0.0), held, worst


def evaluate(label, task, starts, weights):
    results = [rollout(task, offset, weights) for offset in starts]
    reward, successes, distances = np.array(results).T
    print(f"{label}: {int(sum(successes))}/{len(starts)} reached 0.1 mm and held within 1 mm; "
          f"worst tip-to-marker distance across all holds {distances.max() * 1000:.3f} mm; "
          f"mean return {reward.mean():.3f}", flush=True)


def fit_policy(task, rng, demos, iterations):
    samples = []
    for offset in rng.uniform(-0.30, 0.30, (demos, 2)):
        rollout(task, offset, None, samples)
    features, labels = map(np.array, zip(*samples))
    weights = np.linalg.lstsq(features, labels, rcond=None)[0].T
    cloned = weights.copy()
    print(f"Behavior cloning: {demos} demonstrations, {len(samples)} control samples.", flush=True)
    # Episodic reinforcement learning with a Gaussian cross-entropy policy search.
    # Score the incumbent and candidates on identical starts within each generation.
    mean = weights.copy()
    spread = np.full(weights.shape, max(np.abs(weights).max() * 0.35, 0.1))
    for generation in range(iterations):
        starts = rng.uniform(-0.30, 0.30, (2, 2))
        population = [weights, *rng.normal(mean, spread, size=(7, *weights.shape))]
        scores = [np.mean([rollout(task, start, candidate)[0] for start in starts])
                  for candidate in population]
        elite = np.asarray(population)[np.argsort(scores)[-3:]]
        weights = elite[-1].copy()
        mean = elite.mean(axis=0)
        spread = np.maximum(elite.std(axis=0), 0.05)
        print(f"RL generation {generation + 1}: mean return {max(scores):.3f}", flush=True)
    return cloned, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/touch.npz"))
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="select the surface vertex nearest these rest-world coordinates in metres")
    target.add_argument("--target-vertex", type=int, help="zero-based liver surface vertex (default: green top marker)")
    parser.add_argument("--demos", type=int, default=8)
    parser.add_argument("--rl-iters", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if min(args.demos, args.eval_episodes) < 1 or min(args.seed, args.rl_iters) < 0:
        parser.error("Use positive episode counts and a nonnegative seed and RL iteration count.")
    vertex = args.target_vertex
    if args.eval_only:
        weights, saved_vertex = load_policy(args.checkpoint)
        if vertex is None and args.target_xyz is None:
            vertex = saved_vertex
    if args.target_xyz is not None and not np.isfinite(args.target_xyz).all():
        parser.error("Target coordinates must be finite.")
    root = build_scene(Sofa.Core.Node("root"), target_vertex=vertex, target_xyz=args.target_xyz, visuals=False)
    Sofa.Simulation.initRoot(root)
    marker = np.array(root.liver.target.point.position.value[0])
    vertex = int(np.argmin(np.linalg.norm(root.liver.tissue.position.value - marker, axis=1)))
    print(f"Target: surface vertex {vertex}, rest-world XYZ {marker.round(5)} m", flush=True)
    task = Pointing(root)
    training, evaluation = [np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(2)]
    starts = evaluation.uniform(-0.30, 0.30, (args.eval_episodes, 2))
    if not args.eval_only:
        cloned, weights = fit_policy(task, training, args.demos, args.rl_iters)
        args.checkpoint.parent.mkdir(exist_ok=True, parents=True)
        # Publish a complete checkpoint even when evaluation runs concurrently.
        with TemporaryDirectory(dir=args.checkpoint.parent) as folder:
            temporary = Path(folder) / "policy.npz"
            np.savez(temporary, weights=weights, vertex=vertex)
            temporary.replace(args.checkpoint)
        print(f"Saved {args.checkpoint}", flush=True)
        evaluate("Imitation only", task, starts, cloned)
    evaluate("Saved policy", task, starts, weights)


if __name__ == "__main__":
    main()
