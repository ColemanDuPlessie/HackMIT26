"""Turn pose-estimation's kinematic humanoid into an actuated one.

`humanoid.xml` exists to be posed by IK, so it has no actuators at all (nu = 0). Physics
tracking needs torques, so this injects one motor per rotational degree of freedom: three for
each ball joint (about the joint frame's x, y and z) and one per hinge, 34 in total, which is
exactly the model's actuated DOF count (nv 40 minus the free root's 6).

Torque limits are per body part rather than uniform: a wrist that can produce hip-level torque
makes the search space much larger for no benefit.

The source model also disables collisions on every body geom (`contype="0"`), which is right
for IK — the solver poses the skeleton and contacts would only fight it — and fatal for
physics: without them the humanoid falls through the floor. They are switched back on here.
"""

import re
from pathlib import Path

import mujoco
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "pose-estimation" / "humanoid.xml"

# Peak torque (N·m) by joint, roughly proportional to the muscle mass involved.
TORQUE = {
    "spine": 200.0, "neck": 40.0,
    "l_shoulder": 100.0, "r_shoulder": 100.0,
    "l_elbow": 60.0, "r_elbow": 60.0,
    "l_wrist": 20.0, "r_wrist": 20.0,
    "l_hip": 300.0, "r_hip": 300.0,
    "l_knee": 200.0, "r_knee": 200.0,
    "l_ankle": 90.0, "r_ankle": 90.0,
}
DEFAULT_TORQUE = 50.0
AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


# The IK model's geom default; physics needs the opposite.
NO_CONTACT = 'contype="0" conaffinity="0"'
CONTACT = 'contype="1" conaffinity="1" friction="0.9 0.1 0.1"'
# 10 ms is fine for replaying poses and diverges immediately under torque control (MuJoCo
# reports "huge value in QACC" and silently resets, which looks exactly like actions having no
# effect). Armature also goes up: the IK model's 0.01 makes the joints numerically stiff.
IK_TIMESTEP = '<option timestep="0.01"/>'
PHYSICS_TIMESTEP = '<option timestep="0.002" iterations="50" solver="Newton"/>'
IK_JOINT = '<joint damping="1" armature="0.01"/>'
PHYSICS_JOINT = '<joint damping="2" armature="0.05"/>'


def actuated_xml(source: Path = SOURCE) -> str:
    """The source model with body collisions enabled and an <actuator> block appended."""
    model = mujoco.MjModel.from_xml_path(str(source))
    rows = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        kind = int(model.jnt_type[j])
        limit = TORQUE.get(name, DEFAULT_TORQUE)
        if kind == int(mujoco.mjtJoint.mjJNT_BALL):
            for axis, suffix in zip(AXES, "xyz"):
                rows.append(f'    <motor name="{name}_{suffix}" joint="{name}" '
                            f'gear="{" ".join(str(a) for a in axis)}" ctrlrange="-{limit} {limit}"/>')
        elif kind == int(mujoco.mjtJoint.mjJNT_HINGE):
            rows.append(f'    <motor name="{name}" joint="{name}" gear="1" '
                        f'ctrlrange="-{limit} {limit}"/>')
    block = "  <actuator>\n" + "\n".join(rows) + "\n  </actuator>\n"
    xml = source.read_text()
    for wanted, replacement in ((NO_CONTACT, CONTACT), (IK_TIMESTEP, PHYSICS_TIMESTEP),
                                (IK_JOINT, PHYSICS_JOINT)):
        if wanted not in xml:
            raise ValueError(f"expected {wanted!r} in {source}; the model has changed shape")
        xml = xml.replace(wanted, replacement, 1)
    return re.sub(r"</mujoco>\s*$", block + "</mujoco>\n", xml)


def load_actuated(source: Path = SOURCE) -> mujoco.MjModel:
    """An MjModel of the humanoid with motors on every rotational DOF."""
    model = mujoco.MjModel.from_xml_string(actuated_xml(source),
                                           {p.name: p.read_bytes() for p in source.parent.glob("*.xml")})
    assert model.nu == model.nv - 6, (model.nu, model.nv)
    return model


def joint_layout(model: mujoco.MjModel) -> list[dict]:
    """Where each actuated joint lives in qpos/qvel/ctrl, so the PD controller can index them."""
    layout, ctrl = [], 0
    for j in range(model.njnt):
        # int(): jnt_type is a numpy int, and `numpy_int in (enum, enum)` is always False,
        # which silently produced an empty layout and therefore zero torque everywhere.
        kind = int(model.jnt_type[j])
        if kind not in (int(mujoco.mjtJoint.mjJNT_BALL), int(mujoco.mjtJoint.mjJNT_HINGE)):
            continue
        width = 3 if kind == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        layout.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
            "ball": kind == int(mujoco.mjtJoint.mjJNT_BALL),
            "qpos": int(model.jnt_qposadr[j]),
            "qvel": int(model.jnt_dofadr[j]),
            "ctrl": ctrl,
            "width": width,
        })
        ctrl += width
    return layout


def rotvec_between(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation (axis * angle) taking quaternion `current` to `target`, in the parent frame."""
    inverse = np.empty(4)
    mujoco.mju_negQuat(inverse, current)
    difference = np.empty(4)
    mujoco.mju_mulQuat(difference, target, inverse)
    rotvec = np.empty(3)
    mujoco.mju_quat2Vel(rotvec, difference, 1.0)
    return rotvec
