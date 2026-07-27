"""Motion planning through clutter generated to be hard, with the reference motion replayed as a ghost.

The obstacles are not authored. A reference motion is drawn first in an EMPTY world - a spline through randomly
drawn joint configurations - then boxes are placed against it, each pushed until its closest approach over the
WHOLE motion is the target clearance. That fixes both halves of a meaningful problem: the reference stays
collision-free, so a solution provably exists, and no box is decorative, since each comes within centimetres of
the robot at some instant. Half of them are seeded on the straight line from start to goal, the route a planner
tries first, which has to be shut off for clutter to pose anything at all.

The planner is given only the two endpoints. Both motions are then replayed together - the arm follows the plan,
a translucent ghost follows the reference - and they are rarely alike, which is the point of showing both.

Run with -v for the viewer, -n to pose several independent problems at once and play back the first.
"""

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities import RigidEntity
from genesis.utils.misc import tensor_to_array

# Clearance every box is placed at, and the range box edge lengths are drawn from.
CLEARANCE = 0.02
BOX_MIN, BOX_MAX = 0.04, 0.12
# Configurations sampled along the motion, and boxes dropped at each. Their product crowds the scene and is the
# strongest difficulty control there is - a quarter of this clutter and most problems fall to the first route
# tried. It keeps paying up to about eighty boxes, past which the scene is merely expensive.
N_CONFIGS, M_PER_CONFIG = 10, 8
N_BOXES = N_CONFIGS * M_PER_CONFIG
# Control points of the reference spline, and samples per segment. Three bends the reference without letting it
# wander: two degenerates into a single swing clutter cannot constrain, more sweeps so far from any route a
# planner would take that the boxes line a path nobody needs. The sampling keeps consecutive states a fraction
# of a box apart, which is what lets the curve be walked by teleport without missing what it sweeps between.
N_CONTROL, STEPS_PER_SEGMENT = 3, 60
# Redraws allowed per env, and the least end-effector path a reference must sweep. The floor is deliberately
# low: a shorter reference runs closer to the route a planner would pick, so boxes placed against it are in
# that route's way rather than lining a detour.
N_SPLINE_DRAWS, MIN_EE_PATH = 40, 1.75
# Settling walks one state in this many, at a phase advancing with the pass, so the passes together see every
# state at a fraction of the cost. The clearing pass that follows walks all of them.
SETTLE_STRIDE = 6
# Passes over the trajectory, and how many a box may stay in contact before being respawned rather than pushed on.
N_PASSES, RESPAWN_AFTER = 400, 10
# How far past the contact a push goes, so a box comes out just clear of what it hit rather than resting on it.
NUDGE = 0.002
# Past this distance from the base a box cannot be near the arm at all, so a push that would cross it respawns.
REACH_LIMIT = 1.0
# Rounds of pairwise separation applied whenever boxes move, each splitting every overlapping pair apart.
N_SEPARATE = 40
# How far a push may be turned off the contact normal, under 1 so it stays in the normal's half-space and still
# separates. The exact normal lets a box cycle between two sweeps of the arm, and moves boxes that share a seed
# in lockstep so they stay piled.
PUSH_JITTER = 0.7
# Overlap below this counts as none: boxes end up exactly touching, and rounding puts the depth either side of 0.
OVERLAP_TOL = 1e-9
# Attempts at clearing the swept motion, each pushing aside whatever the last one found in the way.
N_CLEAR = 24
Q_START = np.array([[0.0, -0.3, 0.0, -1.0, 0.0, 1.5, 0.785, 0.04, 0.04]])
# Where a box waits while it is not part of the scene: out of the arm's reach.
PARKED = np.array([4.0, 0.0, 0.5])


@dataclass
class Workspace:
    """Everything the generator needs to pose a problem in a built scene, most of it read off the scene itself."""

    scene: gs.Scene
    franka: RigidEntity
    boxes: list[RigidEntity]
    probes: list[RigidEntity]
    box_size: np.ndarray
    n_envs: int

    def __post_init__(self):
        self.n_envs = max(self.n_envs, 1)
        self.hand = self.franka.get_link("hand")
        self.box_radius = 0.5 * np.linalg.norm(self.box_size, axis=1)
        self.half_extent = 0.5 * self.box_size
        self.box_links = [box.base_link.idx for box in self.boxes]
        self.probe_links = [probe.base_link.idx for probe in self.probes]
        self.franka_geoms = {geom.idx for link in self.franka.links for geom in link.geoms}
        self.arm_geoms = sorted(self.franka_geoms)
        self.geom_to_box = {g.idx: i for i, box in enumerate(self.probes) for link in box.links for g in link.geoms}
        self.q_lower, self.q_upper = (tensor_to_array(v) for v in self.franka.get_dofs_limit())


def spline_waypoints(work, rng):
    """A smooth joint-space trajectory: a Catmull-Rom spline through randomly drawn control points.

    Coefficients rather than steps: a random walk accumulates small increments and drifts, producing a reference
    the direct route can shortcut. The spline passes through every control point, needs no solve, and takes its
    tangents from the neighbours, so the arm keeps moving through a point rather than stopping on it. The first
    is the start configuration, so the motion begins where the plan will.
    """
    control = rng.uniform(work.q_lower, work.q_upper, size=(work.n_envs, N_CONTROL, len(work.q_lower)))
    control[:, 0] = Q_START
    padded = np.concatenate([control[:, :1], control, control[:, -1:]], axis=1)
    t = np.linspace(0.0, 1.0, STEPS_PER_SEGMENT, endpoint=False)[:, None]
    basis = 0.5 * np.hstack(
        [-(t**3) + 2.0 * t**2 - t, 3.0 * t**3 - 5.0 * t**2 + 2.0, -3.0 * t**3 + 4.0 * t**2 + t, t**3 - t**2]
    )
    out = [np.einsum("sk,bkd->sbd", basis, padded[:, i_seg : i_seg + 4]) for i_seg in range(N_CONTROL - 1)]
    return np.clip(np.concatenate([*out, control[None, :, -1]]), work.q_lower, work.q_upper)


def arm_in_contact(work):
    """Per env: is the arm touching ANYTHING in the configuration currently set?

    Used while the reference is drawn, every box parked away, so whatever it finds is the arm against itself or
    the floor. Both make a motion the robot cannot perform, and singling out one is how the other slips through.
    """
    work.scene.rigid_solver.collider.detection()
    contacts = work.scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    return np.array(
        [
            any(
                int(a) in work.franka_geoms or int(b) in work.franka_geoms
                for a, b in zip(contacts["geom_a"][i_b], contacts["geom_b"][i_b])
            )
            for i_b in range(work.n_envs)
        ]
    )


def box_contacts(work):
    """Box-vs-arm contacts at the configuration currently set, as (box, env, penetration, normal, position).

    Ragged form, one array per env: the tensor form pads every env out to the batch's largest contact count, and
    those padding rows carry stale values that would steer a box by a contact that is not there.
    """
    work.scene.rigid_solver.collider.detection()
    contacts = work.scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    hits = []
    for i_b in range(work.n_envs):
        geom_a, geom_b = contacts["geom_a"][i_b], contacts["geom_b"][i_b]
        depth, normal, pos = contacts["penetration"][i_b], contacts["normal"][i_b], contacts["position"][i_b]
        for i_c in range(len(geom_a)):
            g_a, g_b = int(geom_a[i_c]), int(geom_b[i_c])
            i_box = work.geom_to_box.get(g_a, work.geom_to_box.get(g_b, -1))
            if i_box >= 0 and (g_a in work.franka_geoms or g_b in work.franka_geoms):
                hits.append((i_box, i_b, float(depth[i_c]), np.array(normal[i_c]), np.array(pos[i_c])))
    return hits


def move_boxes(work, links_idx, placed):
    """Move a whole set of boxes at once.

    One solver call rather than one per box: settling repositions every probe on every pass, so per-box calls
    spend as long dispatching kernels as the collision queries themselves take.
    """
    work.scene.rigid_solver.set_base_links_pos(placed.transpose(1, 0, 2), links_idx)


def park(work, links_idx):
    """Move a set of boxes out of the arm's reach, which is how they stop being part of the problem."""
    move_boxes(work, links_idx, np.tile(PARKED, (len(links_idx), work.n_envs, 1)))


class BoxPairs(NamedTuple):
    """Every pair of boxes, with what the overlap test needs and their positions cannot change.

    The separating-axis test needs fifteen candidate axes per pair and both boxes' reach projected onto them,
    which follow from the ORIENTATIONS and sizes alone - drawn once, then fixed, while only positions move - plus
    the distance beyond which a pair cannot touch, for the broad phase.
    """

    i_box: np.ndarray
    j_box: np.ndarray
    axes: np.ndarray
    reach: np.ndarray
    span: np.ndarray


def box_pairs(axis_box, half_extent, box_radius):
    """Precompute, once per scene, everything about box pairs that does not depend on where the boxes are."""
    i_box, j_box = np.triu_indices(len(axis_box), k=1)
    crossed = np.cross(axis_box[i_box][:, :, :, None, :], axis_box[j_box][:, :, None, :, :])
    axes = np.concatenate([axis_box[i_box], axis_box[j_box], crossed.reshape(len(i_box), -1, 9, 3)], axis=2)
    length = np.linalg.norm(axes, axis=-1)
    # Parallel edges give a zero-length cross product, which is no axis at all. Rather than dropping it and
    # leaving pairs with unequal axis counts, it gets infinite reach: never the axis a pair overlaps least along,
    # so it can never hide a real overlap either.
    is_axis = length > 1e-9
    axes = axes / np.where(is_axis, length, 1.0)[..., None]
    reach = sum(
        np.einsum("pbak,pk->pba", np.abs(np.einsum("pbad,pbkd->pbak", axes, axis_box[box])), half_extent[box])
        for box in (i_box, j_box)
    )
    return BoxPairs(i_box, j_box, axes, np.where(is_axis, reach, np.inf), box_radius[i_box] + box_radius[j_box])


def box_overlap(placed, pairs):
    """Depth and direction of each box pair's overlap, by the separating-axis theorem (SAT).

    Two oriented boxes are disjoint exactly when their projections are apart on one of fifteen axes: the three
    edge directions of each, plus the nine cross products between them. Returned as a depth, negative when the
    pair is already apart, and a unit direction pointing from the first box toward the second.
    """
    delta = placed[pairs.j_box] - placed[pairs.i_box]
    # Bounding spheres first. Boxes further apart than their two half-diagonals cannot touch whatever their
    # orientations, and nearly every pair is in that position at any moment, so the exact test runs on the few
    # that are close rather than on the quadratically many that exist.
    is_near = np.linalg.norm(delta, axis=-1) < pairs.span[:, None]
    depth = np.full(delta.shape[:2], -np.inf)
    direction = np.zeros_like(delta)
    axes = pairs.axes[is_near]
    along = np.einsum("kad,kd->ka", axes, delta[is_near])
    # The largest gap decides: a pair is disjoint as soon as one axis separates it, and otherwise that axis is
    # the one it overlaps least along - the least motion that would pull the two apart.
    i_axis = np.argmax(np.abs(along) - pairs.reach[is_near], axis=-1)
    i_near = np.arange(len(i_axis))
    depth[is_near] = (pairs.reach[is_near] - np.abs(along))[i_near, i_axis]
    direction[is_near] = axes[i_near, i_axis] * np.sign(along[i_near, i_axis])[:, None]
    return depth, direction


def separate_boxes(placed, pairs, box_radius):
    """Drive overlapping boxes apart, as far as the rounds allow.

    Boxes may intersect - two of them crossing is a perfectly good obstacle. What this prevents is the pile:
    boxes seeded at the same instant, pushed the same way, fusing into one blob the box count no longer
    describes. Each overlapping pair is split along the axis it overlaps least, half the depth each, and the set
    re-tested, since moving one pair apart can push a box into a third.
    """
    for _ in range(N_SEPARATE):
        depth, direction = box_overlap(placed, pairs)
        # Over the overlapping pairs alone: a mostly-zero push for every pair in every env costs more than
        # finding the overlaps did.
        i_pair, i_env = np.nonzero(depth > OVERLAP_TOL)
        if not len(i_pair):
            break
        push = 0.5 * depth[i_pair, i_env][:, None] * direction[i_pair, i_env]
        np.subtract.at(placed, (pairs.i_box[i_pair], i_env), push)
        np.add.at(placed, (pairs.j_box[i_pair], i_env), push)
        # The floor is not one of the pairs, so a split that buries a box has to be undone here.
        placed[:, :, 2] = np.maximum(placed[:, :, 2], box_radius[:, None])
    return placed


def reference_motion(work, rng):
    """Draw spline trajectories until every env has one the arm can perform in full, and return their poses.

    A drawn spline can fold the arm into itself or drive it through the floor, and a reference doing either is a
    motion the robot cannot perform - the planner would be right to refuse it. The whole trajectory is REDRAWN
    per env when that happens, never truncated at the first bad sample: truncating keeps whatever prefix was
    clean, throwing away the span that makes the problem worth posing. Nothing downstream sees a bad reference.
    """
    park(work, work.box_links)
    park(work, work.probe_links)
    waypoints = spline_waypoints(work, rng)
    is_ok = np.zeros(work.n_envs, dtype=bool)
    for _ in range(N_SPLINE_DRAWS):
        is_clean = np.ones(work.n_envs, dtype=bool)
        ee = []
        for q_step in waypoints:
            work.franka.set_qpos(q_step, zero_velocity=False)
            is_clean &= ~arm_in_contact(work)
            ee.append(tensor_to_array(work.hand.get_pos()))
        # Long enough as well as performable: difficulty riding on the luck of the draw mostly measures easy draws.
        is_clean &= np.linalg.norm(np.diff(np.stack(ee), axis=0), axis=-1).sum(axis=0) >= MIN_EE_PATH
        is_ok |= is_clean
        if is_ok.all():
            break
        waypoints = np.where(is_ok[None, :, None], waypoints, spline_waypoints(work, rng))
    if not is_ok.all():
        gs.raise_exception(f"{int((~is_ok).sum())} envs never drew a performable reference trajectory")
    return waypoints


def place_boxes(work, rng, traj):
    """Settle every box just clear of the robot, walking the trajectory and nudging whatever it hits.

    Seeding drops each box onto a randomly chosen robot geom, in a random orientation, at one of N
    configurations along the motion, so every box starts inside the robot at a known instant - each has a contact
    to be pushed out of, and none begins where it would never have constrained the motion.

    Settling walks the trajectory and moves any box in contact just clear of its DEEPEST contact over the whole
    walk, until a walk finds nothing. Only boxes in contact move: nudging the clear ones too walks every box away
    from the motion, destroying the clearance the problem is built on while looking convergent. The deepest
    contact rather than each as it is met, because freeing a box at one instant can push it into the arm at an
    instant already passed, which oscillates instead of settling.

    A contact is reported only once geometry overlaps, so the clearance cannot be observed directly. Probes
    inflated by it stand in during the walk: a probe just out of contact is a real box exactly that far away.
    """
    # Two pools, because the two things that make a scene hard are separate: boxes along the REFERENCE narrow
    # the corridor the arm threads, boxes on the STRAIGHT LINE from start to goal shut off the route the planner
    # tries first. Direct seeds come from the samples furthest from the reference - one seeded where the routes
    # run close is cleared off both and blocks neither, while one that never touches the reference stays put.
    pools = []
    for q_pool in (
        [traj[0] + (traj[-1] - traj[0]) * t for t in np.linspace(0.0, 1.0, N_CONFIGS)],
        [traj[i_t] for i_t in np.linspace(0, len(traj) - 1, N_CONFIGS).astype(int)],
    ):
        geoms = []
        for q_step in q_pool:
            work.franka.set_qpos(q_step, zero_velocity=False)
            geoms.append(tensor_to_array(work.scene.rigid_solver.get_geoms_pos(work.arm_geoms)))
        pools.append(np.stack(geoms).transpose(1, 0, 2, 3).reshape(work.n_envs, -1, 3))
    pool_direct, pool_reference = pools
    n_sample = pool_direct.shape[1]
    gap = np.linalg.norm(pool_direct[:, :, None] - pool_reference[:, None], axis=-1).min(axis=-1)
    direct_far = np.argsort(-gap, axis=-1)[:, : max(1, n_sample // 4)]

    def spawn(i_box, i_b, is_biased=False):
        if i_box % 2 == 0:
            # Respawning draws from the WHOLE direct route, unlike first seeding: a box is respawned because it
            # had no room, so returning it to the same preferred region is how it cycles instead of settling.
            i_far = direct_far[i_b, rng.integers(0, direct_far.shape[1])] if is_biased else rng.integers(0, n_sample)
            seed_pos = pool_direct[i_b, i_far]
        else:
            seed_pos = pool_reference[i_b, rng.integers(0, n_sample)]
        # A geom close to the ground would seed a box through the floor, so it rests on the floor instead - still
        # inside the robot, which is all seeding has to guarantee.
        return np.array([seed_pos[0], seed_pos[1], max(seed_pos[2], work.box_radius[i_box] + 1e-3)])

    placed = np.empty((N_BOXES, work.n_envs, 3))
    for i_box in range(N_BOXES):
        for i_b in range(work.n_envs):
            placed[i_box, i_b] = spawn(i_box, i_b, is_biased=True)
    quats = np.stack(
        [
            [gu.rotvec_to_quat(rng.normal(size=3) * rng.uniform(0.0, np.pi)) for _ in range(work.n_envs)]
            for _ in range(N_BOXES)
        ]
    )
    pairs = box_pairs(gu.quat_to_R(quats).transpose(0, 1, 3, 2), work.half_extent, work.box_radius)
    placed = separate_boxes(placed, pairs, work.box_radius)
    park(work, work.box_links)
    for i_box in range(N_BOXES):
        work.probes[i_box].set_quat(quats[i_box])
    move_boxes(work, work.probe_links, placed)

    n_stuck = np.zeros((N_BOXES, work.n_envs))
    # Convergence needs a whole cycle of phases clean, which is the full trajectory checked in pieces - one
    # clean pass saw a single state in SETTLE_STRIDE and says nothing about the rest.
    n_clean = 0
    for i_pass in range(1, N_PASSES + 1):
        pen = np.zeros((N_BOXES, work.n_envs))
        step = np.zeros((N_BOXES, work.n_envs, 3))
        for q_step in traj[i_pass % SETTLE_STRIDE :: SETTLE_STRIDE]:
            work.franka.set_qpos(q_step, zero_velocity=False)
            for i_box, i_b, depth, normal, pos in box_contacts(work):
                if depth > pen[i_box, i_b]:
                    # The normal gives the separating line, but its stored orientation belongs to the contact
                    # pair, so the sign comes from the box's own side of it rather than being assumed.
                    away = np.sign(float(np.dot(normal, placed[i_box, i_b] - pos))) or 1.0
                    pen[i_box, i_b] = depth
                    out = normal * away + PUSH_JITTER * gu.normalize(rng.normal(size=3))
                    out = out / np.linalg.norm(out)
                    reach = depth + NUDGE
                    # The floor is invisible here - plane against fixed box is not a checked pair - so it is
                    # imposed: a step that would bury the box is laid flat instead, which still separates it from
                    # the arm unless the contact is exactly overhead. Clamping afterwards would leave it stuck.
                    if placed[i_box, i_b, 2] + out[2] * reach < work.box_radius[i_box]:
                        flat = np.array([out[0], out[1], 0.0])
                        span = np.linalg.norm(flat)
                        if span < 1e-6:
                            flat = placed[i_box, i_b] - pos
                            flat[2] = 0.0
                            span = np.linalg.norm(flat)
                        out = flat / span if span > 1e-6 else np.array([1.0, 0.0, 0.0])
                    step[i_box, i_b] = out * reach
        n_clean = n_clean + 1 if not (pen > 0.0).any() else 0
        if n_clean >= SETTLE_STRIDE:
            break
        n_stuck += pen > 0.0
        trial = placed + step
        # Out of reach, or stuck for many passes: either way it has no room where it is, so it goes back onto
        # the robot and starts over rather than being nudged forever in a pocket. That is the only way a box
        # legitimately leaves reach - letting it fly and filtering after would silently drop it.
        flung = (np.linalg.norm(trial, axis=-1) >= REACH_LIMIT) | (n_stuck >= RESPAWN_AFTER)
        for i_box, i_b in zip(*np.nonzero(flung)):
            trial[i_box, i_b] = spawn(i_box, i_b)
            n_stuck[i_box, i_b] = 0.0
        # Separating here rather than at the end: overlapping boxes are ones the walk treats as a single
        # obstacle, so every later pass sees the arrangement the problem will actually have.
        placed = separate_boxes(trial, pairs, work.box_radius)
        move_boxes(work, work.probe_links, placed)

    # Sanity checks, not filters: a box seeded on the robot and pushed just clear has no business under the floor
    # or out of the workspace, and one that never settles means the schedule failed to free it.
    if (pen > 0.0).any():
        gs.raise_exception(f"{int((pen > 0.0).sum())} boxes never settled clear of the motion")
    if (placed[:, :, 2] < work.box_radius[:, None]).any():
        gs.raise_exception("boxes were pushed through the floor")
    if (np.linalg.norm(placed, axis=-1) >= REACH_LIMIT).any():
        gs.raise_exception("boxes were pushed outside the workspace")

    # Settling walked a subsample, blind between its samples: the arm sweeps through space they never visited,
    # and a box there makes the reference unperformable. These passes close that gap at full resolution.
    for i_clear in range(N_CLEAR):
        is_clean = True
        for q_step in traj:
            work.franka.set_qpos(q_step, zero_velocity=False)
            for i_box, i_b, depth, normal, pos in box_contacts(work):
                away = np.sign(float(np.dot(normal, placed[i_box, i_b] - pos))) or 1.0
                placed[i_box, i_b] = placed[i_box, i_b] + normal * away * (depth + CLEARANCE)
                work.probes[i_box].set_pos(placed[i_box])
                is_clean = False
        if is_clean:
            break
    else:
        gs.raise_exception("the reference motion still hits boxes after the correction budget")
    park(work, work.probe_links)
    move_boxes(work, work.box_links, placed)
    for i_box in range(N_BOXES):
        work.boxes[i_box].set_quat(quats[i_box])
    gs.logger.info(f"scene posed: {N_BOXES} boxes cleared in {i_clear + 1} pass(es)")
    return placed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-n", "--n_envs", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu, seed=args.seed)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, 0.6, 1.2),
            camera_lookat=(0.15, 0.0, 0.3),
            camera_fov=45,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Plane(),
    )
    # A morph's size is fixed at build time, so sizes vary across boxes while poses vary per env. Two sets at the
    # same poses: the real boxes the planner avoids, and probes inflated by the clearance, which is what makes
    # that clearance observable - only an overlap ever is.
    box_size = np.random.default_rng(args.seed).uniform(BOX_MIN, BOX_MAX, size=(N_BOXES, 3))
    boxes, probes = (
        [
            scene.add_entity(
                gs.morphs.Box(
                    size=tuple(box_size[i_box] + (2.0 * CLEARANCE if is_probe else 0.0)),
                    pos=(4.0 + 0.2 * i_box, 2.0 * is_probe, 0.5),
                    fixed=True,
                    visualization=not is_probe,
                ),
                # Semi-transparent: opaque clutter hides the very corridor the plan threads through it.
                surface=gs.surfaces.Default(color=(0.65, 0.65, 0.72, 0.35)),
            )
            for i_box in range(N_BOXES)
        ]
        for is_probe in (False, True)
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    # The reference motion, as a translucent ghost. Kinematic, so it is drawn and posed but takes no part in the
    # physics - the planner must not see the motion it is asked to rediscover, nor collide with a second arm.
    ghost = scene.add_entity(
        material=gs.materials.Kinematic(),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        surface=gs.surfaces.Default(color=(0.05, 0.05, 0.05, 0.55)),
    )
    # Marker at the goal's end-effector position, so what the planner was asked for is visible, not implied.
    goal_marker = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(5.0, 0.0, 0.5),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(0.15, 0.85, 0.25, 0.55)),
    )

    ########################## build ##########################
    scene.build(n_envs=args.n_envs)

    work = Workspace(scene, franka, boxes, probes, box_size, args.n_envs)

    ########################## pose the problem ##########################
    rng = np.random.default_rng(args.seed)
    traj = reference_motion(work, rng)
    place_boxes(work, rng, traj)

    ########################## plan ##########################
    goal = traj[-1]
    franka.set_qpos(goal, zero_velocity=False)
    goal_marker.set_pos(work.hand.get_pos())
    franka.set_qpos(np.repeat(Q_START, work.n_envs, axis=0))
    path = franka.plan_path(goal)
    gs.logger.info(f"planned {int(path.is_valid.sum())}/{work.n_envs} of the problems posed")

    ########################## replay the plan against the reference ##########################
    # Each motion runs to its own end and holds: the planner's route is usually far shorter than the reference.
    for i_frame in range(max(len(path.qpos), len(traj))):
        franka.set_qpos(path.qpos[min(i_frame, len(path.qpos) - 1)], zero_velocity=False)
        ghost.set_qpos(traj[min(i_frame, len(traj) - 1)])
        scene.step()


if __name__ == "__main__":
    main()
