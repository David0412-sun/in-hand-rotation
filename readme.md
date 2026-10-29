# Robot Synesthesia Codebase

This repository is a code implementation of the following paper:

**[Robot Synesthesia: In-Hand Manipulation with Visuotactile Sensing](https://yingyuan0414.github.io/visuotactile/)**

Ying Yuan, Haichuan Che, Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Kang-Won Lee, Yi Wu, Soo-Chul Lim, Xiaolong Wang

International Conference on Robotics and Automation (ICRA), 2024




https://github.com/YingYuan0414/robosyn/assets/74405101/77c87802-59e7-43ac-afc1-330df13fb930




## Preparation

We suggest using conda environment with python 3.8. Install **Isaac Gym Preview 4 release** on your laptop (should have a GPU) and server (follow the instructions on the NVIDIA's website, you need to register an account). Other required packages include **pytorch3d**, **hydra-core**, **ray**, **tensorboard**, **wandb**, etc. You may install them via **pip**.

For detailed instruction, see [install.md](install.md).

## Launch Training
### Teacher Policy Training

Training outputs use fixed object/task directories. Select one object with
`task.env.objectName=<name>` while keeping its original `task.env.objSet`.
For example:

```bash
bash scripts/teacher_cross.sh 0 task.env.objectName=cross4_0 headless=True wandb_activate=False max_iterations=2000
bash scripts/teacher_axis.sh 0 task.env.objectName=set_obj1_regular_block task.env.axis=x headless=True wandb_activate=False max_iterations=2000
```

These save checkpoints under `runs/cross4_0/z-axis/nn/` and
`runs/set_obj1_regular_block/x-axis/nn/`. Each task directory also contains
`config.yaml` and `summaries/`. Changing training parameters, `experiment`, or
`user_prefix` does not create another run directory. The same task reuses its
directory: the latest training config and best-model filename may be overwritten,
while periodic checkpoints and TensorBoard event files remain there. Teacher
playback (`test=True`) creates no run directory and does not overwrite its config.
Use the same object selection and axis when replaying or resuming a checkpoint.

The available objects are:

| Object set | Object names |
| --- | --- |
| `ball` | `ball` (the Baoding task still contains two balls per environment) |
| `cross` | `cross4_0`, `cross4_1`, `cross4_2`, `cross4_3`, `cross4_4` |
| `C` | `set_obj1_regular_block`, `set_obj2_block`, `set_obj3_block`, `set_obj4_block`, `set_obj5_block`, `set_obj6_block_corner`, `set_obj7_block`, `set_obj8_short_block`, `set_obj9_thin_block`, `set_obj10_thin_block_corner`, `set_obj11_cylinder`, `set_obj12_cylinder_corner`, `set_obj13_irregular_block`, `set_obj14_irregular_block_cross`, `set_obj15_irregular_block_time`, `set_obj16_cylinder_axis` |

The C objects support separate X/Y/Z training tasks. With an empty `objectName`,
the original whole-set training remains available and saves to the fixed
`runs/_object_sets/<objSet>/<axis>-axis/` directory. The single-entry ball set
always saves to `runs/ball/<axis>-axis/`.

Student training and teacher-data collection use fixed `student/` and
`collection/` subdirectories under the same object/task directory; explicit
teacher checkpoint and demonstration input paths are unchanged.

The preserved 2000-iteration Baoding run is archived under
`runs/ball/z-axis/archive_2000/`: `best.pth`, `epoch_2000.pth`, and the original
`config.yaml`. New training writes to `nn/` and does not overwrite this archive.

(1) For training a state-based policy for double-ball rotation, run
```
scripts/teacher_baoding.sh 0
```
(2) For training a state-based policy for wheel-wrench rotation, run
```
scripts/teacher_cross.sh 0
```
(3) For training a state-based policy for three-axis rotation, run
```
scripts/teacher_axis.sh 0 task.env.axis=x experiment=x-axis train.params.config.user_prefix=x-axis
```
```
scripts/teacher_axis.sh 0 task.env.axis=y experiment=y-axis train.params.config.user_prefix=y-axis
```
```
scripts/teacher_axis.sh 0 task.env.axis=z experiment=z-axis train.params.config.user_prefix=z-axis
```

We also provide code for training [PS (Rotating Without Seeing)](https://touchdexterity.github.io/) and Visual RL baselines as follows:

(1) For PS baselines, run
```
scripts/teacher_baoding.sh 0 task.env.observationType=partial_stack_baoding experiment=baoding-ps train.params.config.user_prefix=baoding-ps
```
```
scripts/teacher_cross.sh 0 task.env.observationType=partial_stack experiment=wheel-wrench-ps train.params.config.user_prefix=wheel-wrench-ps
```
```
scripts/teacher_axis.sh 0 task.env.observationType=partial_stack task.env.axis=[axis] experiment=[axis]-ps train.params.config.user_prefix=[axis]-ps
```

(2) For Visual RL baselines, run
```
scripts/teacher_baoding_visrl.sh 0
```
```
scripts/teacher_cross_visrl.sh 0
```
```
scripts/teacher_axis_visrl.sh 0 task.env.axis=[axis] experiment=[axis]-visualrl train.params.config.user_prefix=[axis]-visualrl
```


### RapidHand / Vision Hand teacher

`RapidHandRotation` uses the fixed-base, five-finger `vision_hand` from the
adjacent `rapidhand_isaacgym` project. Its URDF and 41 meshes are self-contained
under [assets/rapidhand](assets/rapidhand/README.md). It uses the same Isaac Gym
Preview 4 / Python 3.8 environment as the Allegro tasks; Isaac Lab is not required.

The task has 20 actions and 15 tactile channels. Every URDF link whose name
contains `tactile` (case-insensitive) remains a separate rigid body. Its actual
net contact force is read from the GPU tensor; contact flags enter the policy
observation. These are link-level contact forces, not pressure images or an
object-specific contact measurement. Camera housings remain geometry only.

An optional `urdf/vision_hand_simple_collision.urdf` replaces 11 structural
collision meshes with boxes and retains 30 collision meshes. All visual meshes, links,
joints, inertials and 15 tactile collision meshes are preserved. Add
`task.env.handAssetFile=urdf/vision_hand_simple_collision.urdf` to an existing
training or replay command to select it; the original URDF remains the default.
Use the same asset for training and replay. Output folders and network dimensions
are unchanged. See [asset details and the tactile check](assets/rapidhand/README.md)
for retained geometry and limitations.

In the viewer, adjacent hand links alternate blue, orange, purple and cyan.
Tactile links are red when inactive and green when triggered. Their colors use
the same contact flags as the actor: peak force over the completed control step
at least `contactThreshold` (default 0.05 N). Reset clears them back to red.
Color updates run only with a viewer and do not change physics or observations.

Hand self-collision is enabled by default (`task.env.selfCollision=True`).
Different fingers, nonadjacent finger segments and distal segments against the
palm can collide. Fixed attachments and directly joint-connected assemblies are
filtered out. The five palm/finger-mount pairs in `selfCollisionIgnorePairs` are
also excluded because their housings overlap by design. Tactile bodies stay
separate, so their signals and green indicators can now include contact with
other parts of the hand as well as the object.

Each environment groups its hand and manipulated object(s) into one PhysX
aggregate by default (`task.env.aggregateMode=1`). Capacity uses the actual
asset body/shape counts, including both balls for the two-object task.
Intra-aggregate collisions remain enabled with the existing shape filters;
tactile bodies and camera mounts remain separate. Set `task.env.aggregateMode=0`
for a comparison with the previous behavior. GPU memory savings require a
long-running comparison and are not guaranteed by this grouping alone.

Checkpoints trained before self-collision was enabled still load, but the changed
physics can change their behavior. Use `task.env.selfCollision=False` to replay
with the earlier collision settings; train with the default `True` for policies
that learn under self-collision.

For a single object (`objSet=cross`, `C` or `custom`), the actor receives four frames of
`[joint positions (20), joint targets (20), touch flags (15), rotation axis (3)]`
plus the current object state (13): 245 values. The central critic receives
`[positions (20), velocities (20), targets (20), actions (20), object state (13),
axis (3), tactile forces (45)]`: 141 values. With `objSet=ball`, each environment
contains two balls; their two states (26 values) give 258 actor observations and
154 critic inputs. These input layouts are unchanged by the unified entry point.
Single-object checkpoints and two-ball checkpoints have different
input dimensions and cannot be exchanged. Existing two-ball checkpoints load
through this entry point with `task.env.objSet=ball`.
Joint positions and targets are normalized by their URDF limits; object position
is relative to the hand base. Each reset clears that environment's observation
history and contact accumulation. Completed environments return fresh initial
observations together with the previous episode's terminal reward/done flags,
so the next policy action uses the correct state. The actor and central-critic
MLPs and PPO algorithm inherit the existing teacher configuration. Allegro
checkpoints have different input/action dimensions and cannot be loaded into
this task.

Activate `robosyn`, then train from scratch with 4096 environments:

```bash
bash scripts/teacher_rapidhand.sh 0 task.env.numEnvs=4096 max_iterations=2000
```

Both policy and central-value PPO minibatches default to 16384. With the default
16-step horizon, `numEnvs * 16` must be divisible by 16384; 1024, 2048, 4096 and
8192 environments satisfy this requirement. The task's environment-count default
remains 256, so small training runs must explicitly reduce both minibatches,
as in the examples below.

The single RapidHand task entry point is `task=RapidHandRotation`, selected by
`scripts/teacher_rapidhand.sh`. Its default is `cross4_0`, Z rotation.
Use `task.env.objSet` to choose `cross`, `C`, `ball` or `custom`, `task.env.objectName`
to choose a specific object (empty means the whole set), and `task.env.axis`
to select the rotation axis. Without an explicit object name, `cross` selects
`cross4_0`; `C` selects its full set and `ball` selects its sole sphere type.
For example:

Append these object-selection overrides to the training command above:

| Selection | Overrides |
| --- | --- |
| Single cross | `task.env.objectName=cross4_1` |
| Single C object | `task.env.objSet=C task.env.objectName=set_obj1_regular_block` |
| All cross shapes | `task.env.objectName=` |
| All C shapes | `task.env.objSet=C task.env.objectName=` |
| Two balls | `task.env.objSet=ball` |

Whole-set training distributes different shapes across environments, with one
object per environment. The default grasp is intended for `cross4_0`; other
shapes can need different initial poses or scale.

Models, TensorBoard logs and the latest configuration stay in
`runs/rapidhand/<object>/<axis>-axis/{nn,summaries,config.yaml}` (by default,
`runs/rapidhand/cross4_0/z-axis/`). Whole-set policies use
`runs/rapidhand/_object_sets/<set>/<axis>-axis/`.
Changing hyperparameters reuses this directory; back up models you want to keep
before starting another run. Original Allegro output directories are unaffected.

Replay the best available checkpoint in a viewer:

```bash
bash scripts/teacher_rapidhand.sh 0 test=True headless=False task.env.numEnvs=16 \
  checkpoint=runs/rapidhand/cross4_0/z-axis/nn/cross4_0_z.pth
```

Press Escape in the viewer or Ctrl+C in the terminal to exit. As in the original
teacher, a best checkpoint is normally saved after iteration 100; a final
checkpoint is also saved when the iteration limit is reached.

With `objSet=ball`, the reward measures the two balls' rotation around their
common center about world Z (the currently supported axis), rather than either
ball's own spin. To train for 10 iterations with a viewer:

```bash
bash scripts/teacher_rapidhand.sh 0 task.env.objSet=ball \
  max_iterations=10 headless=False force_render=True \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024
```

Its models, TensorBoard logs and configuration are written under
`runs/rapidhand/ball/z-axis/`. This short run checks the simulation and training
pipeline; it does not establish that the policy has learned the task. Each ball
has a 0.022 m radius from `assets/urdf/objects/ball.urdf`, scaled by `objectScale`,
and a configured mass of 0.05 kg. Starting positions, separation limits and all
other RapidHand settings share
`isaacgymenvs/cfg/task/RapidHandRotation.yaml`. Camera observations and student
distillation are not part of this task.

For two 55 mm balls, select `isaacgymenvs/cfg/object/ball_55mm.yaml`:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=ball_55mm \
  task.env.numEnvs=256 max_iterations=10 test=False \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  headless=False force_render=True wandb_activate=False \
  task.sim.physx.default_buffer_size_multiplier=1.0 \
  task.sim.physx.max_gpu_contact_pairs=4194304
```

This preset scales the existing ball URDF by 1.25, keeps each ball at 50 g
and friction at 1.0, and sets initial centres 63 mm apart. It reuses the same
task and PPO networks. Outputs go to
`runs/rapidhand/ball_55mm/ball/z-axis/{nn,summaries,config.yaml}`, separate
from the original 44 mm ball models. Use `+object=ball_55mm` during replay
as well. Ten iterations validate the pipeline; they do not establish a
trained rotation policy.

For the 60 mm chamfered block, select
`isaacgymenvs/cfg/object/block_corner_60mm.yaml` (`objectScale=1`). For 85 mm,
use `+object=block_corner_85mm`; that preset inherits the 60 mm configuration
and uses `objectScale=85/60`, scaling visual and collision geometry together.
The 85 mm preset also opts into the pose-progress reward described below.
For 75 mm, use `+object=block_corner_75mm`; it inherits the 85 mm settings
with `objectScale=75/60=1.25` and the centred small-step reward described below.
All three use the same task output directory. Mass remains 50 g:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=block_corner_60mm \
  max_iterations=10 headless=False force_render=True \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024
```

This preset selects `set_obj6_block_corner`, the simplified-collision hand URDF
and self-collision. It uses the previous 60 mm cube test's open-hand joint pose,
with the object at hand-local `[0.005, -0.0715, 0.11]`, moved back 30 mm from
the previous forward position. The 60 mm preset keeps the legacy RapidHand reward:
angular velocity projected onto the target world axis, with spin coefficient
2.0 and the existing speed clamp. Contact reward inherits coefficient 0.2;
the fall penalty remains -15. There are no separate thumb rewards.
After a 2 s settling period and a full 3 s window, each environment resets if
its signed orientation increments about the target axis sum to less than
5 degrees in magnitude over the last 3 s. Back-and-forth motion cancels in
this check; a short pause for regrasping is allowed. Stalls terminate the
episode without an additional penalty. The same reset rule applies in
training and replay. `rotationStallTimeout=0` disables it, which remains
the default for other objects and the two-ball task.
It sets PhysX buffers to `1.0 / 4194304` and inherits 256 environments. Both PPO
minibatches default to 16384; the short-run command above overrides both to 1024.
Outputs stay in
`runs/rapidhand/set_obj6_block_corner/z-axis/{nn,summaries,config.yaml}`.
Use the same object preset during replay. The short run is intended to check
the initial grasp, tactile response and training pipeline for this combination.

The 85 mm preset uses `rotationRewardMode=pose_delta`: the rotation term is
`2 * clamp(signed_pose_increment / control_dt, -10, 10)`. The quaternion
increment is projected onto the fixed world rotation axis, shared with the
stall check, and cleared on reset. Reverse rotation remains negative; invalid
or fallen terminal motion earns no rotation reward. Contact, position, action,
joint-velocity and fall terms retain their existing coefficients. There is no
additional stall penalty. After the same 2 s grace and full 3 s window, a net
**forward** angle below 15 degrees ends the episode (`rotationStallForwardOnly=True`).
Other presets retain their existing reward and termination behavior.

The 75 mm preset retains pose-based progress and the same stall/reset rules,
and enables `centeredRotationReward`. With signed speed `w` in rad/s and
distance `e` in metres from the fixed initial palm-frame object centre, it uses:

```text
d = max(e - 0.005, 0)
g = exp(-0.5 * (d / 0.015)^2)
rotation_reward = 2 * (g * min(max(w, 0), 0.6) + min(w, 0))
position_penalty = -0.2 * (d / 0.01)^2
```

Forward reward saturates at about 3.4 degrees per 0.1 s control step; reverse
progress is penalised linearly without clipping or position attenuation.
The 5 mm tolerance allows small translations during regrasping. The reference
does not follow the object. The quadratic term replaces the linear position
penalty, and `contactRewardScale=0` removes the ordinary contact bonus. Other
reward terms and the -15 fall penalty are unchanged; fallen or invalid terminal
steps still earn no rotation reward. `rotation/position_reward_gate` and
`rotation/position_penalty` report the current per-environment means (the latter
as a positive penalty magnitude). No extra overspeed or action-change penalty is
enabled. Train with the existing network and output directory:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=block_corner_75mm \
  task.env.numEnvs=8192 max_iterations=4000
```

This starts a new training run with the viewer and cameras disabled. The usual
best-checkpoint filename in the shared object directory will be overwritten as
new best models are saved. To replay a 75 mm checkpoint with its earlier reward
settings, append `task.env.centeredRotationReward=False
task.env.positionPenaltyScale=2.0 task.env.contactRewardScale=0.2`.

For this mode TensorBoard `Episode/rotation/` logs completed-episode net angle
(`net_angle_deg`), `mean_actual_spin`, `mean_sampled_spin`, reverse-step fraction,
rotation/total returns, duration, and mutually exclusive termination fractions
(stall, drop, nonfinite, timeout). Episodes are weighted equally, and these
metrics cover every control step rather than only the last step of a rollout.
Net angle is accumulated projected orientation change; with substantial off-axis
motion it is not a certified count of complete fixed-axis revolutions. Reward
scores across the old and new modes are not directly comparable.

To replay an older 85 mm checkpoint under its original reward/reset settings,
append `task.env.rotationRewardMode=angular_velocity
task.env.rotationStallForwardOnly=False task.env.rotationStallMinAngleDeg=5`.
The observation dimensions and PPO network are unchanged, so older weights
remain loadable; learning the revised objective requires training.

The ideal `box_50x50x50` is a 50 x 50 x 50 mm cube, with one rigid body and
one primitive box collision shape. It belongs to the RapidHand `custom` set.
Select its object preset to use a matching initial position and open-hand pose:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=box_50x50x50 \
  task.env.numEnvs=256 max_iterations=10 headless=True force_render=False \
  train.params.config.minibatch_size=1024 \
  train.params.config.central_value_config.minibatch_size=1024 \
  task.sim.physx.default_buffer_size_multiplier=1.0 \
  task.sim.physx.max_gpu_contact_pairs=4194304
```

The preset is `isaacgymenvs/cfg/object/box_50x50x50.yaml`; its provisional
mass is 0.05 kg and friction is 1.0, not measured real-object values. Scale is
1.0 and the cube is initially aligned with the world axes. `objectPositionLocal` uses hand
coordinates; `objectRotationEuler` uses world-frame Euler angles in radians.
The task overrides URDF mass with `task.env.objectMass` and recomputes inertia.
Use the same `+object=box_50x50x50` preset for training and replay to retain
the cube's initial position and finger pose. Existing object defaults
are unchanged. A short run validates the pipeline, not learned rotation skill.

Outputs go to `runs/rapidhand/box_50x50x50/<axis>-axis/`, with weights in `nn/`,
TensorBoard events in `summaries/` and the training settings in
`config.yaml`. The default axis is Z. To replay a saved box checkpoint, use:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=box_50x50x50 \
  task.env.numEnvs=16 test=True headless=False force_render=True \
  task.sim.physx.default_buffer_size_multiplier=1.0 \
  task.sim.physx.max_gpu_contact_pairs=4194304 \
  'checkpoint="<path-to-the-box-checkpoint.pth>"'
```

For a **single 75 mm-diameter sphere**, use `+object=sphere_75mm`. This preset
inherits the 75 mm chamfered block's hand pose, fixed centre, centred rotation
reward, stall settings and PhysX parameters. It selects `objSet=custom`, so
the sphere rotates about its own axis using the single-object policy, with
the same observation/action dimensions. Visual and collision geometry are
both a primitive sphere of radius 37.5 mm, scale 1.0, with provisional mass
50 g and friction 1.0. It adds no joints or mesh collision geometry.

To try the existing block policy on the sphere with camera preview only:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=sphere_75mm \
  test=True headless=False force_render=False task.env.numEnvs=16 \
  train.params.config.player.deterministic=True \
  task.camera.enabled=True task.camera.display=True task.camera.record=False \
  checkpoint=runs/rapidhand/set_obj6_block_corner/z-axis/nn/set_obj6_block_corner_z.pth
```

These weights were trained on the block; performance on the sphere is not
guaranteed by compatible network dimensions. The uniform sphere has no visual
orientation marker; use the rotation-angle logs to quantify its spin.
Future sphere training uses `runs/rapidhand/sphere_75mm/z-axis/`, with the best
checkpoint named `nn/sphere_75mm_z.pth`. Existing object assets and policies
remain available.

For a **75 x 75 x 75 mm cube**, use `+object=cube_75mm`. This reuses the
source `set_obj1_regular_block` asset (60 mm) at scale 1.25 for both visual
and collision geometry, and inherits the 75 mm chamfered block's centred
rotation reward, hand pose, mass, friction and simulation settings. To try
the existing block policy with camera preview and no recording:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=cube_75mm \
  test=True headless=False force_render=False task.env.numEnvs=16 \
  train.params.config.player.deterministic=True \
  task.camera.enabled=True task.camera.display=True task.camera.record=False \
  checkpoint=runs/rapidhand/set_obj6_block_corner/z-axis/nn/set_obj6_block_corner_z.pth
```

This is a shape-transfer test using the chamfered-block weights. Future cube
training outputs go to `runs/rapidhand/set_obj1_regular_block/z-axis/`, with
the best checkpoint named `nn/set_obj1_regular_block_z.pth`.

Object files and settings are stored here:

| Content | Location |
| --- | --- |
| Box geometry, reference mass and inertia | `assets/urdf/objects/box_50x50x50.urdf` |
| Single 75 mm sphere geometry, reference mass and inertia | `assets/urdf/objects/sphere_75mm.urdf` |
| All object URDFs, including `ball.urdf`, `cross4_*.urdf` and `set_obj*.urdf` | `assets/urdf/objects/` |
| Cross visual/collision meshes | `assets/urdf/objects/meshes/non-convex-param/` |
| C-set visual/collision meshes | `assets/urdf/objects/meshes/set2/` |
| Object lists and output-directory routing | `isaacgymenvs/utils/object_tasks.py` |
| Common RapidHand physics, reward and control settings | `isaacgymenvs/cfg/task/RapidHandRotation.yaml` |
| Common RapidHand PPO training settings | `isaacgymenvs/cfg/train/RapidHandRotationPPO.yaml` |

RapidHand training logs GPU memory by default (`task.gpuMemoryLog=True`); replay
does not log it. The existing task directory contains
`summaries/gpu_memory.csv`, appended across launches with a timestamp, session ID,
PID and epoch number. Each epoch has `epoch_start`, `rollout_end` (before PPO
updates) and `epoch_end` records. Caught exceptions add an `error` record before
environment cleanup. Every record is flushed and synced to disk, so completed
records survive a CUDA crash; a hard kill can leave only the last recorded phase.

Memory values are in MiB. `device_used_mib` includes PhysX, PyTorch, graphics and
other processes on that GPU; `process_used_mib` is the driver-reported memory of
this training PID, when available. Physical GPU indices/UUIDs come from
`nvidia-smi`; `torch_device` separately identifies the logical CUDA device used
by PPO. PyTorch allocated/reserved memory and their peaks since epoch start
exclude PhysX allocations. Device/process values are snapshots at the recorded
phases, not instantaneous peaks over an entire epoch. Unavailable readings stay
blank; failed queries add a diagnostic in `error`. No extra package is required.
Use `task.gpuMemoryLog=False` to disable this monitoring on a later launch.

For example, follow the truncated-block memory log with:

```bash
tail -n 12 -f runs/rapidhand/set_obj6_block_corner/z-axis/summaries/gpu_memory.csv
```

The box and ball use primitive geometry and need no mesh files or pretrained
point-cloud embeddings. Teacher training samples experience from simulation;
no external dataset is required. Two-ball checkpoints have different network
input dimensions and cannot be resumed as a single-box policy.

Position control starts from the source model's stiffness 3.0, damping 0.5 and
URDF limits (1 Nm, 1 rad/s). Objects have gravity and the hand base is fixed.
Initial hand/object poses, contact threshold, action scale and reset noise are
configured in `isaacgymenvs/cfg/task/RapidHandRotation.yaml`.

To physically validate every tactile link with a force-driven sphere, run:

```bash
python scripts/check_rapidhand_tactile.py --headless --output /tmp/rapidhand_tactile.json
```

This uses two isolated GPU environments and checks all 15 pads: force above
0.05 N during contact, decay after withdrawal, and no response in the untouched
environment. The diagnostic uses an open hand and a finer physics timestep than
training to access each pad independently. Its JSON report includes force
vectors, probe motion and parent-body contacts. Use `--pads IDIP_TACTILE` for a
single pad, or `--viewer` to display the probe. A failed check exits nonzero.

#### 指尖相机预览与录像

RapidHand 的训练和回放共用 `task.camera` 配置，默认全部关闭。开启后，
五个指尖相机随对应 `TDIP_CAMERA`、`IDIP_CAMERA`、`MDIP_CAMERA`、
`RDIP_CAMERA`、`LDIP_CAMERA` link 运动，输出 **112×112 RGB、水平视场角90°**。
图像仅用于监视和录像，不加入策略观测，已有 checkpoint 可以直接使用。

以下命令回放 75 mm 截角方块模型，打开原生 Isaac Gym Viewer 和旁边的五路相机窗口，
默认只预览，不保存视频、帧记录或录像元数据：

```bash
bash scripts/teacher_rapidhand.sh 0 +object=block_corner_75mm \
  test=True task.env.numEnvs=4 headless=False force_render=False \
  train.params.config.player.deterministic=True \
  task.camera.enabled=True task.camera.display=True task.camera.record=False \
  checkpoint=runs/rapidhand/set_obj6_block_corner/z-axis/nn/set_obj6_block_corner_z.pth
```

在 Isaac Gym 窗口中按 `P`，再点击目标手的包围框，切换相机显示对象。
黄色框表示当前选中的手，青色框表示其他可选手。也可以按 `[`、`]`，
或使用相机窗口的环境下拉框选择。相机预览是独立的并排窗口，
不是嵌入原生 Viewer 的面板；原生 Actors 面板的选择不会同步到相机窗口。
选中某只手后看到的是它的实际固定相机朝向，镜头不会自动转向物体。

`task.camera.display=True` 要求 `headless=False`。相机窗口的录像开关可以
开始或停止保存；切换手时结束当前片段，再记录新选中的手。
只有手动勾选相机窗口的 `Record`，或将命令中的 `task.camera.record=False`
改为 `True`，才会保存录像。旧命令中的 `record=True` 会覆盖默认的不录像设置。

无窗口训练录像时，在现有训练命令后添加以下参数即可；保留原来的物体、
环境数、训练轮数及 PPO 参数：

```bash
task.camera.enabled=True task.camera.record=True \
  task.camera.display=False headless=True force_render=False
```

默认只给 `env 0` 创建五个相机，不会给全部训练环境创建相机。
修改 `task.camera.env_id=3` 可以固定记录 `env 3`。显示模式默认给前四个环境
创建相机（不足四个时使用现有环境）；需要指定候选手时，可在启动命令中加入：

```bash
'task.camera.env_ids=[0,3]' task.camera.env_id=0
```

相机在启动时创建，运行中只能切换已有候选手，不能增加候选环境。
Isaac Gym 每次会渲染所有已创建的相机：四只候选手就是20路相机，
候选环境越多，渲染时间和显存开销越大。大规模训练建议只记录一个环境。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `task.camera.enabled` | `False` | 总开关；关闭时不创建相机、预览窗口或录像资源 |
| `task.camera.display` | `False` | 打开五路相机窗口及 Viewer 选手功能 |
| `task.camera.record` | `False` | 启动时开始保存五路视频 |
| `task.camera.capture_every` | `1` | 每几个控制步采集一次图像 |
| `task.camera.segment_steps` | `300` | 每段录像最多覆盖的控制步数 |
| `task.camera.record_interval_steps` | `0` | `0` 为连续录像；正数为间隔录制周期 |
| `task.camera.record_steps` | `200` | 每个间隔周期开始时录制的控制步数 |

五路图像在每个采样控制步结束后、环境重置前同步采集。当前控制周期约0.1秒，
默认录像约10 FPS，300步对应约30秒仿真时间；视频时间按仿真时间计算，
不等于训练实际耗时。周期录像可使用
`task.camera.record_interval_steps=1000 task.camera.record_steps=200`，
即每1000个控制步录制前200步。`record_steps` 不应大于正的录像周期。

视频随当前任务保存，路径为：

```text
<当前任务保存目录>/videos/train或replay/<时间>/env_0000/segment_0001/
├── thumb.mp4
├── index.mp4
├── middle.mp4
├── ring.mp4
├── little.mp4
├── metadata.json
└── frames.jsonl
```

`metadata.json` 保存分辨率、视场角、挂载变换、帧率及 checkpoint 信息；
`frames.jsonl` 记录逐帧仿真时间和回合边界，便于区分环境重置。
录像通过 FFmpeg 流式编码，不会把整段视频缓存到内存。正常退出或 Ctrl+C
会关闭编码器并保存当前片段。预览使用 Tkinter 和 Pillow，录像需要 PATH 中
可用的 `ffmpeg`；只录像时不需要桌面窗口。

当前镜头挂载是基于 CAD 的近似值，尚未做实机光学标定：镜头前端朝 camera link
的 `+Z`，仿真光心放在前端外1 mm。Isaac Gym 相机的原生视线是 `+X`
（画面右方 `-Y`、上方 `+Z`），因此挂载旋转采用弧度
`[π/2, -π/2, 0]`，使视线朝 link `+Z`、画面右方朝 link `+X`。
各路位置和旋转可以在 `task.camera.mounts` 中调整。张开的手可能暂时拍到空处，
这是固定安装方向与当前手姿共同决定的；本功能不模拟自动对焦、镜头畸变或
滚动快门。关闭相机功能使用 `task.camera.enabled=False`。

### Student Policy Training
We provide an implementation of behavior cloning here. You may also implement your own method of distillation with the task environment we provide. 

#### Data Collection
To collect simulation rollouts of a trained teacher policy checkpoint, run
```
scripts/collect_baoding.sh 0 teacher_logdir=[directory/of/teacher/checkpoint] teacher_resume=[name/of/teacher/checkpoint]
```
```
scripts/collect_cross.sh 0 teacher_logdir=[directory/of/teacher/checkpoint] teacher_resume=[name/of/teacher/checkpoint]
```
```
scripts/collect_axis.sh 0 teacher_logdir=[directory/of/teacher/checkpoint] teacher_resume=[name/of/teacher/checkpoint] task.env.axis=[axis] distill.teacher_data_dir=demonstration-[axis] experiment=bc-[axis]-collect train.params.config.user_prefix=bc-[axis]-collect
```
Collected trajectory data will be stored in `distill.teacher_data_dir`, by default the directory `demonstration-baoding`. You may also collect data in parallel by launching more than one job with different `distill.worker_id`.

#### Behavior Cloning

We then train the student policy with collected data:
```
scripts/bc_baoding.sh 0 distill.teacher_data_dir=[directory/of/data]
```
```
scripts/bc_cross.sh 0 distill.teacher_data_dir=[directory/of/data]
```
```
scripts/bc_axis.sh 0 distill.teacher_data_dir=[directory/of/data] task.env.axis=[axis] distill.student_logdir=runs/student/bc-[axis] experiment=bc-[axis] train.params.config.user_prefix=bc-[axis]
```

To use different sensing capabilities, set `distill.ablation_mode` to:

* `multi-modality-plus` for Touch+Cam+Aug+Syn
* `aug` for Touch+Cam+Aug
* `no-tactile` for Cam+Aug
* `no-pc` for Touch

## Citing
Please cite this work as:
```
@inproceedings{
	yuan2024robosyn,
	author = {Ying Yuan, Haichuan Che, Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Kang-Won Lee, Yi Wu, Soo-Chul Lim, Xiaolong Wang},
	title = {Robot Synesthesia: In-Hand Manipulation with Visuotactile Sensing},
	booktitle = {ICRA},
	year = {2024}
} 
```

**Note** if you use [Rotating without Seeing](https://touchdexterity.github.io/) in your work, please ensure you cite the following work:
```
@article{
      touch-dexterity,
      title          = {Rotating without Seeing: Towards In-hand Dexterity through Touch },
      author         = {Yin, Zhao-Heng and Huang, Binghao and Qin, Yuzhe and Chen, Qifeng and Wang, Xiaolong},
      journal        = {Robotics: Science and Systems},
      year           = {2023},
}
```
