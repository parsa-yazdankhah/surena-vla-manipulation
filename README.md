# SURENA VLA Manipulation

SURENA V manipulation support for:

- standalone MuJoCo simulation;
- robosuite robot registration;
- LIBERO manipulation environments;
- MiniVLA / OpenVLA action execution;
- joint-space, inverse-kinematics, and VLA delta-action control.

The SURENA MJCF, meshes, controllers, and integration code are owned by this
repository. Do not copy SURENA files into LIBERO or robosuite.

---

## 1. Supported environment

The project currently preserves the following research environment:

| Component | Version |
|---|---:|
| Python | 3.10 |
| NumPy | 1.26.4 |
| MuJoCo | 2.3.7 |
| Mink | 1.1.1 |
| DAQP | 0.8.7 |
| robosuite | 1.4.1 |
| PyTorch | 2.2.0+cu121 |
| torchvision | 0.17.0+cu121 |
| torchaudio | 2.2.0+cu121 |

### Known MuJoCo/Mink mismatch

The frozen environment contains:

```text
mujoco==2.3.7
mink==1.1.1
```

Mink 1.1.1 declares `mujoco>=3.8.1`. The current project intentionally keeps
MuJoCo 2.3.7 because that is the version used with the existing
robosuite/LIBERO integration.

Consequences:

- do not install the complete `requirements.txt` in one resolver run;
- install Mink separately with `--no-deps`;
- install the editable project repositories with `--no-deps`;
- `pip check` is expected to report the Mink/MuJoCo mismatch.
- the project validators and representative rollouts are the acceptance tests.

Upgrading MuJoCo should be treated as a separate migration and validated against
IK, contacts, rendering, robosuite, LIBERO, and VLA rollouts.

---

## 2. System prerequisites

The commands below target Ubuntu/Debian Linux:

```bash
sudo apt update
sudo apt install -y \
    git \
    git-lfs \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    patchelf \
    libgl1 \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    libosmesa6-dev \
    ffmpeg

git lfs install
```

For GPU inference, verify the NVIDIA driver:

```bash
nvidia-smi
```

---

## 3. Clone the repositories

A convenient layout is:

```text
~/surena-vla-ws/
├── LIBERO/
├── openvla-mini/
└── surena-vla-manipulation/
```

Create the workspace:

```bash
export SURENA_WORKSPACE="$HOME/surena-vla-ws"
mkdir -p "$SURENA_WORKSPACE"
cd "$SURENA_WORKSPACE"
```

Clone this repository:

```bash
git clone https://github.com/parsa-yazdankhah/surena-vla-manipulation.git surena-vla-manipulation
```

Clone the two external development repositories at the revisions used by this
project:

```bash
git clone https://github.com/parsa-yazdankhah/LIBERO.git LIBERO

git clone https://github.com/parsa-yazdankhah/openvla-mini.git openvla-mini
```


---

## 4. Verify the SURENA assets

The repository must contain the SURENA STL files under:

```text
src/surena_vla/assets/meshes/
```

Run:

```bash
cd "$SURENA_WORKSPACE/surena-vla-manipulation"
find src/surena_vla/assets/meshes -type f -iname '*.stl' | wc -l
```

Expected count:

```text
32
```

The canonical MJCF is:

```text
src/surena_vla/assets/surena_arm.xml
```

Do not keep duplicate SURENA XML or mesh copies inside LIBERO or robosuite.

---

## 5. Create the Python environment

Create one Python 3.10 virtual environment:

```bash
cd "$SURENA_WORKSPACE"
python3.10 -m venv .vla
source .vla/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Verify the interpreter:

```bash
python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), sys.version
print(sys.version)
PY
```

Activate it in every new terminal:

```bash
source "$SURENA_WORKSPACE/.vla/bin/activate"
```

---

## 6. Reconstruct the Python environment

Run all commands from the SURENA repository:

```bash
cd "$SURENA_WORKSPACE/surena-vla-manipulation"
```

### 6.1 Create the resolver-safe base requirements

Exclude PyTorch packages and Mink because they require separate installation:

```bash
awk '
    /^torch==/       { next }
    /^torchvision==/ { next }
    /^torchaudio==/  { next }
    /^mink==/        { next }
    { print }
' requirements.txt > /tmp/surena-vla-requirements-base.txt
```

### 6.2 Install the CUDA 12.1 PyTorch wheels

```bash
python -m pip install \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 6.3 Install the remaining pinned packages

```bash
python -m pip install --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r /tmp/surena-vla-requirements-base.txt
```

### 6.4 Install Mink without dependency resolution

```bash
python -m pip install --no-deps mink==1.1.1
```

This preserves `mujoco==2.3.7` instead of allowing Mink metadata to request a
newer MuJoCo release.

### 6.5 Install the project repositories in editable mode

```bash
python -m pip install -e "$SURENA_WORKSPACE/LIBERO" --no-deps
python -m pip install -e "$SURENA_WORKSPACE/openvla-mini" --no-deps
python -m pip install -e "$SURENA_WORKSPACE/surena-vla-manipulation" --no-deps
```

Editable installation makes source changes immediately visible. Reinstall only
after changing package metadata, build configuration, or console entry points.

---

## 7. Verify imports and critical versions

```bash
python - <<'PY'
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

import libero.libero as libero_package
import surena_vla


def package_source_path(module_name: str) -> Path:
    spec = find_spec(module_name)

    if spec is None:
        raise RuntimeError(f"Python module not found: {module_name}")

    if spec.origin:
        return Path(spec.origin).resolve()

    locations = spec.submodule_search_locations
    if locations:
        return Path(next(iter(locations))).resolve()

    raise RuntimeError(f"Could not determine source path for: {module_name}")


print("surena_vla :", Path(surena_vla.__file__).resolve())
print("LIBERO     :", Path(libero_package.__file__).resolve())
print("prismatic  :", package_source_path("prismatic"))

expected = {
    "numpy": "1.26.4",
    "mujoco": "2.3.7",
    "mink": "1.1.1",
    "daqp": "0.8.7",
    "robosuite": "1.4.1",
    "torch": "2.2.0+cu121",
    "torchvision": "0.17.0+cu121",
    "torchaudio": "2.2.0+cu121",
}

errors = []

for package, required in expected.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        errors.append(f"{package}: not installed")
        print(
            f"{package:12s} installed={'NOT INSTALLED':18s} "
            f"required={required:18s} MISSING"
        )
        continue

    status = "OK" if installed == required else "MISMATCH"

    print(
        f"{package:12s} installed={installed:18s} "
        f"required={required:18s} {status}"
    )

    if installed != required:
        errors.append(f"{package}: {installed} != {required}")

if errors:
    raise SystemExit(
        "Version verification failed:\n  " + "\n  ".join(errors)
    )

print("\nVersion and source-path verification passed.")
PY
```

The displayed module paths must point into the local workspace clones.

The following command is expected to report the known Mink/MuJoCo mismatch:

```bash
python -m pip check
```

Use the project validators below as the real acceptance tests.

---

## 8. Validate the installation

From the SURENA repository root:

```bash
cd "$SURENA_WORKSPACE/surena-vla-manipulation"
```

Validate standalone MuJoCo:

```bash
python scripts/validate_surena_mujoco.py
```

Validate robosuite registration:

```bash
python scripts/validate_surena_robosuite.py
```

Validate LIBERO reset and a short rollout:

```bash
python scripts/validate_surena_libero.py --steps 10
```

For headless NVIDIA rendering:

```bash
export MUJOCO_GL=egl
python scripts/validate_surena_libero.py --steps 10
```

Do not proceed to VLA inference until all three validators pass.

---

## 9. Runtime registration

Call registration once in every Python process before creating a SURENA
robosuite or LIBERO environment:

```python
from surena_vla.integrations import register_all

register_all()
```

Example:

```python
from surena_vla.integrations import register_all
from surena_vla.integrations.libero import SurenaLift

register_all()

env = SurenaLift(
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview"],
    camera_heights=224,
    camera_widths=224,
    control_freq=20,
)

try:
    obs = env.reset()
    print(obs["agentview_image"].shape)
    print(obs["robot0_eef_pos"])
finally:
    env.close()
```

Registration is process-local. Notebook kernels, worker processes, and separate
evaluation scripts must each call `register_all()`.

---

## 10. Use the SURENA controller

```python
from surena_vla.control import SurenaArmController

controller = SurenaArmController(
    model=env.sim.model._model,
    data=env.sim.data._data,
    prefix="robot0_",
    apply_home=False,
)
```

After a reset that reconstructs the MuJoCo model:

```python
obs = env.reset()
controller.rebind(env)
controller.reset_vla()
```

Do not retain cached MuJoCo IDs across a hard reset without calling `rebind()`.

---

## 11. Standalone MuJoCo commands

Hold the initial pose:

```bash
surena-mujoco --mode hold
```

Run the built-in IK target:

```bash
surena-mujoco --mode ik
```

The script entry point is also available:

```bash
python scripts/run_surena_mujoco.py --mode hold
```

---

## 12. Sticky SURENA hand

The sticky gripper is an intentional command-level approximation of SURENA's
fixed hand; it does not add fingers or substitute another robot's gripper. The
local MiniVLA prediction path unnormalizes a continuous seven-dimensional
action and leaves the seventh component in the Bridge/RLDS `[0, 1]` convention:
`0=close`, `1=open`. The SURENA adapter preserves that raw value and continuously
maps it to close strength with `normalized = 1 - raw` before qualification.

Continuous command handling and discrete physical state are separate:

```text
raw -> affine normalization -> optional EMA -> hysteresis/dwell -> intent
    -> OPEN / SEEKING / ATTACHED / RELEASE_PENDING
```

Defaults use a `0.65` close threshold and `0.35` release threshold. Values in
between retain the previous intent. Close, release, and candidate dwell default
to two controller updates. Filtering is disabled by default; when enabled,
`filter_alpha` applies `y[t] = alpha*x[t] + (1-alpha)*y[t-1]`. Commands are not
clipped or written back into the caller's action array.

```python
sticky = controller.enable_sticky_gripper(
    env,
    object_name_filter=None,
    attach_distance=0.09,
    close_threshold=0.65,
    release_threshold=0.35,
    close_dwell_ticks=2,
    release_dwell_ticks=2,
    candidate_dwell_ticks=2,
    filter_alpha=None,
)
```

Call `sticky_update(action[6])` once per controller/VLA command update. Call
`sticky_enforce()` after raw MuJoCo steps; enforcement never advances dwell
counters. `rebind(env)` retains configuration but clears attachment, candidate,
filter, counters, and cached model IDs. Disabling also releases and resets.

Candidate distance currently uses the freejoint body's origin rather than a
geometry/contact distance. While attached, velocity is zeroed; release velocity
inheritance is not modeled. These are limitations of the approximation, not a
physical grasp model.

---

## 13. Hierarchical adaptive IK

`SurenaIK` starts every request from the measured MuJoCo configuration, not a
cached Mink iterate. DAQP is the primary solver. The normal path makes one
strict full-pose attempt and returns immediately when it is feasible and within
tolerance. Only a rejected candidate triggers this deterministic hierarchy:

1. strict full pose from the measured configuration;
2. full pose using previous accepted/commanded, home, or joint-center seeds and
   compatible alternate solvers discovered through `qpsolvers`;
3. progressively relaxed orientation (`0.15`, then `0.40` rad acceptance);
4. position-dominant IK with a weak orientation preference;
5. fully reevaluated line-search projection from current configuration toward
   the best prior iterate;
6. explicit `hold_current_no_safe_candidate` when nothing safely improves the
   request.

Each candidate records convergence separately from feasibility and acceptance.
Non-finite values, model-evaluation failures, hard joint-limit violations,
critical collision penetration, critical singularity, and a per-request joint
step over `0.40` rad are hard failures. Feasible candidates use a dimensionless
weighted score:

```text
4 (position_error / 0.02 m)^2
+ 1 (orientation_error / 0.20 rad)^2
+ 0.35 (joint_displacement / 0.35 rad)^2
+ 0.15 (second_difference / 0.20 rad)^2
+ 0.25 joint_limit_cost
+ 0.20 singularity_cost
+ 2.0 collision_cost
+ stage_penalty
```

The joint-limit term is a smooth hinge inside 15% of each model-provided joint
range. Singularity uses the smallest singular value of the 6x7 controlled-arm
EEF Jacobian. Collision scoring uses MuJoCo contacts involving robot geoms;
palm contact with a movable freejoint object is treated as intentional, while
fixture contact is not. MuJoCo's contact list does not provide comprehensive
look-ahead collision avoidance, so this is candidate contact validation rather
than a global collision planner.

Online continuity retains only the last two accepted solutions and evaluates
`||q[t] - 2 q[t-1] + q[t-2]||²`. The corresponding trajectory metric is the
mean squared second difference. Projected candidates are completely rescored;
joint clipping is not used as projection. Runtime application uses actuator
targets unless the caller explicitly requests `teleport=True`.

```python
from surena_vla.control import RobustIKConfig

ctrl.configure_robust_ik(RobustIKConfig(
    maximum_joint_step=0.35,
    max_total_attempts=10,
))
result = ctrl.move_eef_to(target_position, target_orientation)
print(result["ik_stage"], result["solver"], result["score"])
```

`reset_ik()` clears continuity history. Hard controller `rebind(env)` rebuilds
all model-dependent IDs, limits, Jacobian/collision metadata, and Mink objects
while preserving the validated IK configuration. Attempts are bounded by stage,
seed, iteration, and total-attempt limits; no threads or random seeds are used.

---

## 14. Dependency-maintenance rules

- Keep `requirements.txt` as the pinned third-party environment.
- Do not add local editable LIBERO or openvla-mini Git lines back into it.
- Keep `pyproject.toml` free of resolver-managed runtime dependencies while the
  MuJoCo/Mink mismatch exists.
- Install Mink and the editable repositories with `--no-deps`.
- Do not regenerate the curated requirements file blindly with `pip freeze`.
- Update dependency pins only after all three validators and representative VLA
  rollouts pass.
