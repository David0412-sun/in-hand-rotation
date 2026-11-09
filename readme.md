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
d = max(e - positionTolerance, 0)
g = exp(-0.5 * (d / 0.015)^2)
rotation_reward = 2 * (g * min(max(w, 0), 0.6) + min(w, 0))
position_penalty = -0.2 * (d / 0.01)^2
```

Forward reward saturates at about 3.4 degrees per 0.1 s control step; reverse
progress is penalised linearly without clipping or position attenuation.
The tolerance is 10 mm only for `axis=y` with `objectName=set_obj6_block_corner`
in this preset, allowing more translation during regrasping. Other axes and the
inherited cube/sphere presets retain 5 mm. The 3 s stall window is unchanged.
To evaluate an existing Y-axis checkpoint with its original reward, override
`task.env.positionTolerance=0.005`. The reference does not follow the object.
The quadratic term replaces the linear position
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

For a separate 60 mm Z-axis task with the same centred reward as the 75 mm
run whose best checkpoint was epoch 3544 (reward 538.45), select
`+object=block_corner_60mm_centered`. It retains the 5 mm tolerance and
all reward, stall, PPO and simulation settings, changing only the object scale
and the output base directory:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=block_corner_60mm_centered \
  task.env.numEnvs=8192 max_iterations=4000
```

Its outputs are saved under
`runs/rapidhand/block_corner_60mm_centered/set_obj6_block_corner/z-axis/`,
with the best model at `nn/set_obj6_block_corner_z.pth`. It uses a separate
directory from the existing 60/75/85 mm runs. Reuse this preset during replay.

The independent `block_corner_65mm_centered`, `block_corner_75mm_centered` and
`block_corner_85mm_centered` presets retain the same historical Z-axis reward
and 5 mm tolerance. Their output roots are separate from each other and from
the 60 mm task:

| Preset | Output root |
| --- | --- |
| `block_corner_65mm_centered` | `runs/rapidhand/block_corner_65mm_centered/` |
| `block_corner_75mm_centered` | `runs/rapidhand/block_corner_75mm_centered/` |
| `block_corner_85mm_centered` | `runs/rapidhand/block_corner_85mm_centered/` |

Each root contains `set_obj6_block_corner/z-axis/`, with the best model at
`nn/set_obj6_block_corner_z.pth`. Start a new 75 mm run with:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=block_corner_75mm_centered \
  task.env.numEnvs=8192 max_iterations=4000
```

For 65 mm, use `+object=block_corner_65mm_centered` in the same command.
Its model, training configuration and logs use the independent 65 mm root.

For `set_obj8_short_block` with 75 mm long edges, use
`+object=short_block_75mm_centered`. The visual size is 60 x 75 x 75 mm;
the centred Z-axis reward, 5 mm tolerance, hand pose, PPO and simulation
settings match `block_corner_75mm_centered`:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=short_block_75mm_centered \
  task.env.numEnvs=8192 max_iterations=4000
```

Its independent output directory is
`runs/rapidhand/short_block_75mm_centered/set_obj8_short_block/z-axis/`, with
the best model at `nn/set_obj8_short_block_z.pth`. Reuse the preset for replay.

For this mode TensorBoard `Episode/rotation/` logs completed-episode net angle
(`net_angle_deg`), `mean_actual_spin`, `mean_sampled_spin`, reverse-step fraction,
rotation/total returns, duration, and mutually exclusive termination fractions
(stall, drop, nonfinite, timeout). Episodes are weighted equally, and these
metrics cover every control step rather than only the last step of a rollout.
Net angle is accumulated projected orientation change; with substantial off-axis
motion it is not a certified count of complete fixed-axis revolutions. Reward
scores across the old and new modes are not directly comparable.

#### Fixed per-face chamfered-block texture

Chamfered-block presets enable one fixed visual texture profile for
`set_obj6_block_corner`. Its 26 faces (6 main faces, 12 edge bevels and 8 corner
triangles) each have a single solid colour in the existing UV atlas, with no
labels, arrows, dots or common dark border. Colours are assigned using actual
shared mesh edges to give neighbouring faces strong colour contrast. Shared
UV-edge texels have a deterministic owner; texture filtering can
blend colours at edges. The seed is fixed at `20261102`; environment number,
episode, policy seed and pose do not select a new texture. The atlas is shared
by all instances and is not re-created during reset. Other objects keep their
existing appearance.

The profile consists of
`assets/textures/chamfered_block_solid_v3/atlas.png` and `manifest.json`.
The manifest records PNG/OBJ SHA256 values, texture ID, face IDs, source vertices,
UV polygons, object-frame normals and coordinate conventions. Runtime loading
rejects mismatched assets. Reproduce the profile with:

```bash
python scripts/build_chamfered_texture.py
```

The generator accepts `--seed`, `--resolution` and `--output-dir`, but refuses to
overwrite an existing profile with different pixels or metadata. Use another
directory for a new profile and point `task.object_texture.atlas_file` and
`task.object_texture.manifest_file` to it (paths are relative to `assets`).
The earlier detailed texture remains in `chamfered_block_fixed_v1`; reproduce it
with `python scripts/build_chamfered_texture.py --profile fixed_v1`. The two-colour
profile remains in `chamfered_block_simple_v2`, reproducible with `--profile simple_v2`.
To replay data collected with an older profile, set both texture paths to its
directory. Each profile retains a distinct texture ID in pose/image metadata.

Texture is attached to the object's local surface coordinates and moves rigidly
with its 6D pose. The same pose has the same surface pattern; images can still
change with camera pose, lighting and hand occlusion. No UV, collision geometry,
mass, friction, reward or policy input is changed. Plain headless training with
no cameras skips GPU texture allocation.

Replay existing chamfered-block checkpoints using their usual preset. Add
`task.object_texture.enabled=False` to recover the untextured appearance.
The runtime exposes `task.object_texture_metadata` and
`task.get_object_visual_state(env_id)` for pose/image provenance.
When finger-camera recording is explicitly enabled, clip `metadata.json`
includes the fixed texture manifest, and every `frames.jsonl` row includes
`object_visual_state`: texture ID, object scale, seven-value `pose_sim` and
`pose_hand` labels (position in metres, quaternion `xyzw`), hand transform and
environment origin. These labels are captured before auto-reset, matching the
recorded image. `pose_sim` uses the actor root-state tensor's coordinate frame;
`pose_hand` uses the hand base. The manifest's source mesh scale and the per-frame
object scale specify how the original face vertices transform with these poses.
Preview still defaults to `task.camera.record=False`; synchronized top-view
datasets use the separate `task.dataset` switch described below.

Replay also enables `task.viewer_axes` in the Viewer. Red, green and blue mean
+X, +Y and +Z. Solid arrows follow the object's root pose; dashed arrows retain
the world directions and are displayed at a fixed point beside the hand
(`world_position: [0.16, 0.0, 0.30]`, in environment coordinates). This default
translates the world-frame reference for visibility; it does not claim that the
displayed reference point is the global origin. To place it at the actual global
origin, use `task.viewer_axes.world_origin_mode=global`.
The arrows are 0.10 m long and follow the selected camera environment, or
environment 0 when there is no camera selection. Set `task.viewer_axes.env_id`
to select another environment, `task.viewer_axes.axis_length` to change length,
or `task.viewer_axes.enabled=False` to hide them. Axes are Viewer overlays,
excluded from camera images and recordings, and skipped during headless training.
RapidHand defaults to `task.viewerRenderCollision=False` so the Viewer displays
visual meshes and their texture. Setting it to `True` displays collision geometry
for inspection; this flag only changes drawing, not collision simulation.

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

For a **53 x 53 x 53 mm cube**, select `+object=cube_53mm`. The source cube's
visual and collision meshes both use scale `53/60`, and all six faces have one
fixed, distinct colour with strong adjacent-face contrast. The texture profile
is `assets/textures/cube_solid_v1/{atlas.png,manifest.json}`; reproduce it with
`python scripts/build_chamfered_texture.py --profile cube_solid_v1`. Its manifest
records the source mesh, UVs, face colours and texture ID for the same pose/image
metadata interface as the chamfered block. Viewer axes remain enabled on replay.
The preset retains the existing cube reward, initial hand pose and physics.

```bash
bash scripts/teacher_rapidhand.sh 0 +object=cube_53mm \
  test=True headless=False force_render=False task.env.numEnvs=16 \
  train.params.config.player.deterministic=True \
  task.camera.enabled=True task.camera.display=True task.camera.record=False \
  checkpoint=runs/rapidhand/set_obj6_block_corner/z-axis/nn/set_obj6_block_corner_z.pth
```

This replays the existing chamfered-block policy with the new cube; it does not
retrain the policy. Future 53 mm cube training saves separately under
`runs/rapidhand/cube_53mm/set_obj1_regular_block/z-axis/`.

For a **65 x 65 x 65 mm cube**, replace `+object=cube_53mm` with
`+object=cube_65mm` in the same command. This changes the source-mesh scale to
`65/60`, retaining the fixed six-face colours, pose metadata, Viewer axes and
rotation settings. Future training saves separately under
`runs/rapidhand/cube_65mm/set_obj1_regular_block/z-axis/`.

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
的 `+Z`，仿真光心放在前端外 0.1 mm，近裁剪距离为 0.1 mm。Isaac Gym 相机的原生视线是 `+X`
（画面右方 `-Y`、上方 `+Z`），因此挂载旋转采用弧度
`[π/2, -π/2, 0]`，使视线朝 link `+Z`、画面右方朝 link `+X`。
各路位置和旋转可以在 `task.camera.mounts` 中调整。张开的手可能暂时拍到空处，
这是固定安装方向与当前手姿共同决定的；本功能不模拟自动对焦、镜头畸变或
滚动快门。关闭相机功能使用 `task.camera.enabled=False`。

近距离接触时，原来的 1 mm 光心偏移可能让光心进入物体的视觉网格，
原来的 1 mm 近裁剪距离也会切掉贴近镜头的表面。现在五路
`task.camera.mounts.<finger>.position` 均为 `[0,0,0.0001]`，
`task.camera.near_plane=0.0001`，避免上述已验证轨迹中的异常消失。
这不保证物体始终在视野内：真实遮挡、转出画面仍然可能发生；极近的表面可能铺满整个画面。
已有 HDF5/RGB 是固定记录，修改成像参数后需要重新采集；当前状态输入教师无需为此重新训练。

#### 同步传感器数据集：五指 RGB、top-view、触觉和状态

`task.dataset.enabled` 默认关闭。开启后，每个被选中的环境创建五个指尖相机和
一个固定在手部 `BASE` 的 top-view 相机；相机数据不进入现有教师策略的输入。
默认仅采集 `env 0`，不会给所有并行训练环境创建相机。

用现有教师模型采集当前 **65 mm 立方体**，无需启动训练：

```bash
conda activate robosyn
bash scripts/collect_rapidhand.sh 0 +object=cube_65mm \
  checkpoint=runs/rapidhand/set_obj6_block_corner/z-axis/nn/set_obj6_block_corner_z.pth \
  task.dataset.output_dir=datasets/rapidhand/cube_65mm \
  task.dataset.max_samples=10000
```

默认一个仿真环境，确定性动作，不显示窗口、不录制 MP4。采满 10000 条同步样本后
自动关闭文件并退出；Ctrl+C 会保存已经完成的样本。该入口直接复用教师网络，不受
旧 player 中 16 环境假设限制。它是在新物体上运行已有策略，采集本身不会训练策略。
安装环境需要 `h5py==3.10.0`；本机 `robosyn` 已安装，未更新原 NumPy/PyTorch。

需要边看边采时，在上述命令后添加：

```bash
headless=False task.camera.display=True
```

沿用 `teacher_rapidhand.sh` 训练时也可添加 `task.dataset.enabled=True`。
训练过程中采到 `max_samples` 后停止写数据，训练继续；独立采集入口则自动退出。
增加采集环境可使用 `task.env.numEnvs=4 'task.dataset.env_ids=[0,1,2,3]'`。
`max_samples` 是所有采集环境合计的行数。

每次运行在 output_dir 下创建独立时间目录，包含 `metadata.json`、`manifest.json`
及 `shard_00000.h5` 等分片。默认每片 1000 条，LZF 无损压缩，异步 CPU 队列上限为
8 条；写入跟不上时等待，不丢帧，也不累积 GPU 图像。每片 `_committed_rows`
标记完整写入的安全行数，读取时忽略其后的未提交部分。正常关闭会写出剩余样本。

| HDF5 字段 | 单条样本形状 / 类型 | 内容 |
| --- | --- | --- |
| `rgb/finger` | `[5,112,112,3] uint8` | thumb、index、middle、ring、little，RGB/HWC |
| `rgb/top` | `[256,256,3] uint8` | 固定顶视 RGB |
| `state/joint_position`, `state/joint_velocity` | `[20] float32` | 关节角 rad、速度 rad/s；顺序在 metadata |
| `tactile/contact` | `[15] uint8` | 各触觉 link 在本控制步是否超过接触阈值 |
| `tactile/peak_force_env`, `tactile/peak_force_hand` | `[15,3] float32` | 环境坐标系 / 手 base 坐标系下的峰值净接触力 N |
| `object/pose_env`, `object/pose_world`, `object/pose_hand` | `[K,7] float32` | 物体位姿：位置 xyz 米 + 四元数 xyzw；单物体 K=1，双球 K=2 |
| `object/pose_top_camera` | `[K,7] float32` | 顶视相机光学坐标系下的物体位姿 |
| `camera/pose_env`, `camera/pose_world`, `camera/pose_hand` | `[6,7] float32` | 六路相机外参，顺序为五指 + top |
| `index/*`, `time/*`, `transition/done` | 标量 | 样本、环境、episode、控制步、仿真时刻和终止标记 |
| `transition/action`, `transition/*_before` | 动作 / 状态数组 | 刚执行的动作，以及执行前的关节、位姿和教师输入 |

另外保存关节目标、物体速度、触觉 link 的位姿、原始相机 view 矩阵、reward、
timeout/stall 和有限数值标记。metadata 含相机内参 K、光学坐标轴约定、关节/触觉名称、
物体尺寸缩放和固定纹理 ID/manifest、URDF/模型哈希及完整配置。
相机 pose 表示 `T_parent_camera`；光学轴为 **X 右、Y 下、Z 前**。
`pose_world` 在 `pose_env` 位置上加对应环境原点。
原始 Gym view 矩阵使用 world 点和行向量乘法，具体约定亦存入 metadata。

top-view 默认分辨率 256²、水平视场角 60°，观察目标固定为手部 **BASE** 坐标下的
`target_local=[0.005,-0.0715,0.11]` m，与物体初始位置/姿态无关。
本手的掌面法线为 BASE 局部 **-Y**（当前安装后对应 world +Z），相机沿该法线
偏移 `distance=0.25` m，因此安装位置为 BASE 下 `[0.005,-0.3215,0.11]` m，
朝向固定目标。相机随 BASE 整体运动，修改 `objectPositionLocal`、物体尺寸或
初始姿态不会改变相机位置和朝向；旧配置中 null/缺失的 target 也使用这一固定
BASE 参考点。已有采集数据中的相机位姿和图像保持原样。
可修改 `task.dataset.top_camera.distance`、`horizontal_fov`、`target_local` 和
`up_local`；修改 target_local 会同时平移相机与观察目标。默认视角覆盖原先
65 mm 立方体的旋转包络，移动物体后应重新检查画幅覆盖；手指造成的遮挡会保留。

所有 RGB 在完成物理步后一次渲染，和同一行的 q/qdot/物体 pose 对应，终止帧在
环境重置前保存。触觉保存的是刚完成控制区间内各 pad 的峰值；不同 pad 的峰值
可能来自不同物理子步，且净接触力包含与其他手部 link/地面接触，不是压力图。
采集默认关闭触觉红绿变色，并在渲染前清掉坐标轴与选框，避免标签泄漏到 RGB。
五指相机沿用原 CAD 近似挂载方向，不会自动追踪物体；当前 65 mm 方块初始抓姿下，
多个指尖视角只看到背景或手部，物体不在画幅内。用于正式感知训练前，应查看六路
预览并标定 `task.camera.mounts` 或调整抓姿；采集器会如实保存这些遮挡/不可见样本。

同一行 RGB 是动作后的状态 **s[t+1]**；`transition/action` 是把 s[t] 推到
s[t+1] 的 a[t]。感知训练可直接使用同一行 RGB 和 pose。视觉行为克隆在
`capture_every=1` 时，应按 `(env_id, episode_id, episode_step)` 分组，将当前
行 RGB 配到下一连续行的 action，跳过 terminal 和不连续样本；不能把同一行的
前序 action 当作该 RGB 的下一步动作标签。末次采集限制也可能在 episode 中间结束。

查看数据与导出六路预览（directory 换成终端打印的实际时间目录）：

```bash
python scripts/inspect_sensor_dataset.py "datasets/rapidhand/cube_65mm/实际时间目录" \
  --sample 0 --preview /tmp/sensor_preview.png
```

按时间顺序回放已经采集的六路图像和传感器数据：

```bash
conda activate robosyn
python scripts/replay_sensor_dataset.py "datasets/rapidhand/block_corner_75mm/实际时间目录"
```

窗口显示 top-view、五指 RGB，以及同一条样本的触觉状态、关节角/速度和物体在
世界坐标系、手部 BASE 坐标系下的位置/四元数。top-view 默认叠加真实世界原点
`W0=(0,0,0)` 的 XYZ 虚线轴，每根轴长 10 cm；同时在每个物体自身原点绘制
长 5 cm 的 XYZ 实线轴，标记为 `O0`、`O1` 等，随记录的物体位姿平移、旋转。
两组轴均以红/绿/蓝表示 +X/+Y/+Z，轴端分别标为 `WX`、`O0X` 等。
叠加使用该帧保存的相机世界位姿和内参投影，不进行遮挡判断；俯视时 Z 轴投影
可能很短。世界原点不在画面内时会明确提示，不会平移原点。叠加仅用于回放显示
及手动导出的视频，原始 HDF5/RGB 保持不变。`w` 切换世界轴，`o` 切换物体轴；
启动参数 `--no-world-axes`、`--no-object-axes` 可分别默认隐藏，数值位姿始终显示。
空格暂停/继续，左右方向键逐帧查看，`r` 从头播放，`q` 或 Esc 退出。
默认按记录的仿真时间播放，`--speed 0.5` 为半速。
多环境数据默认只播放最小编号的环境，可用 `--env-id 1` 切换；
`--episode-id 2` 只播放该 episode。读取范围以各分片 `_committed_rows` 为准。

在回放窗口按 **S**，后台保存当前所选环境/episode 的**完整回放**为 MP4，包含
六路图像、传感器数值，以及按键时的世界/物体坐标轴显示设置，速度沿用 `--speed`。
输出位于当前采集目录的 `videos/replay_env<编号>_<时间>_<唯一后缀>.mp4`，
不会覆盖已有文件。窗口标题显示保存中/完成状态，终端打印完整路径；保存期间可
继续播放、暂停或逐帧查看，重复按 S 不会同时启动第二次导出。关闭窗口时会等待
正在保存的视频完成。不按 S、也不传 `--export` 时不生成视频。

仅导出视频而不打开窗口（默认不会保存视频，已有输出文件不会覆盖）：

```bash
python scripts/replay_sensor_dataset.py "datasets/rapidhand/block_corner_75mm/实际时间目录" \
  --export /tmp/sensor_replay.mp4 --no-display
```

交互回放使用现有 Matplotlib/Tk，导出使用 robosyn 环境中的 FFmpeg；不需要启动
Isaac Gym 或加载教师模型。这里播放的是采集时保存的图像，因此相机当时拍到的
背景或遮挡也会原样显示，回放不会产生新的视角或修改原始 HDF5 数据。

HDF5 可直接用于 PyTorch 感知模型，无需解码视频：

```python
import h5py
import torch

with h5py.File("<时间目录>/shard_00000.h5", "r", swmr=True) as f:
    n = int(f["_committed_rows"][()])
    finger = torch.from_numpy(f["rgb/finger"][0]).permute(0, 3, 1, 2).float() / 255
    top = torch.from_numpy(f["rgb/top"][0]).permute(2, 0, 1).float() / 255
    q = torch.from_numpy(f["state/joint_position"][0])
    pose = torch.from_numpy(f["object/pose_hand"][0])
```

训练/验证集应按完整 episode 或采集运行划分，避免将相邻帧随机拆分到不同集合。

### 新手截角方块：六个主面连续循环展示

75 mm 版本使用 `+object=block_corner_75mm_six_faces`，完整继承下面的六面循环奖励，
仅将原 60 mm 几何缩放为 `1.25`。训练和数据分别保存至
`runs/rapidhand/block_corner_75mm_six_faces_cycle/` 和
`datasets/rapidhand/block_corner_75mm_six_faces_cycle/`，保留 85 mm 版本的配置与模型：

```bash
conda activate robosyn
bash scripts/teacher_rapidhand.sh 0 +object=block_corner_75mm_six_faces \
  task.env.numEnvs=8192 max_iterations=4000
```

独立配置 `block_corner_85mm_six_faces` 复用 85 mm 截角方块、手模型和 PPO 参数，
通过当前目标面与本圈已完成面状态学习换面。对准目标主面即可登记，
不要求停留或低速度；六面完成后继续下一圈，不重置手和物体。启动训练：

```bash
conda activate robosyn
bash scripts/teacher_rapidhand.sh 0 +object=block_corner_85mm_six_faces \
  task.env.numEnvs=8192 max_iterations=4000
```

模型独立保存至
`runs/rapidhand/block_corner_85mm_six_faces_cycle/set_obj6_block_corner/six-faces/nn/`，
最佳模型名为 `set_obj6_block_corner_six_faces.pth`。之前要求停留的六面任务保留在
`runs/rapidhand/block_corner_85mm_six_faces/`；固定轴模型目录也保持不变。
本任务保留 261/157 维策略/价值输入，旧六面模型权重可加载；奖励和第 13 个目标附加
特征的含义改变，旧收益不可直接比较。旧固定轴 checkpoint 输入维度不同，不能直接加载。
`axis=z` 只保留配置兼容性，不定义此任务的旋转轴。已运行的进程不会自动加载修改。

每个 episode 从最接近顶视相机的主面开始，完成后选择尚未完成且角度最近的主面；
六个面的编号依次为物体坐标系 `+X,-X,+Y,-Y,+Z,-Z`，并非固定执行顺序。
完成一圈后清空本圈覆盖记录，下一目标排除刚完成的面；每一圈仍需六个不同主面。
奖励默认如下：

| 项目 | 默认规则 |
| --- | --- |
| 接近目标面 | `2 × (上一步角误差 − 当前角误差)`，弧度；退远相应扣分，切换目标不制造角度差奖励 |
| 展示主面 | 目标面朝向相机误差不超过 15°，满足几何可见性并且物体未掉落，一个控制步即可登记；不要求低速度或 15 mm 内的位置门槛 |
| 停留奖励 | 已删除，始终为 0；`hold_seconds=0`、`hold_reward=0` |
| 完成主面 | 真实 reset 后的初始面仅登记、不给 `+5`；其余每个新主面 `+5`，包括之后每圈的首面 |
| 完成一圈 | 六面全部完成额外 `+10`，继续下一圈，不重置物理状态；本圈同一面不重复计分 |
| 无进展 / 超时 | 每个目标先有 2 秒缓冲；达到 3 秒无至少 5° 的最佳误差改善且未展示成功，或单目标达到 12 秒，扣 `2` 并重置 |
| 掌心位置与掉落 | 超出 15 mm 后，位置惩罚为 `0.2 × ((偏移 − 0.015)/0.01)²`；掉落扣 `15` 并重置。保留手指距离、动作与关节速度惩罚 |

“主面可见”还要求物体包围盒在固定顶视相机画幅内、主面投影面积至少 64 像素。
这是不检测手指遮挡的几何可见性指标；完成六面不保证 26 个面均有无遮挡图像，
正式数据应结合实际 RGB 检查覆盖情况。训练时无需为每个环境创建相机。
物理回合仍受 500 控制步上限、掉落与停滞规则约束。训练日志新增完成圈数；
episode 成功率表示结束前是否完成过至少一圈。`stable` 只作为低速稳定性诊断，
不决定是否登记主面。旧配置 `hold_seconds>0`、`repeat_cycles=False` 仍支持原来的停留任务。

沿用 `task.dataset.enabled=True` 开启六路 RGB/触觉/状态采集，输出至
`datasets/rapidhand/block_corner_85mm_six_faces_cycle/` 下的独立时间目录。
仅此任务新增 HDF5 `six_face/*`：目标面、已完成面、停留与目标计时的动作前后快照，
以及本步实际评估的目标、展示/稳定/几何可见标记、完成圈数、整圈完成事件、
停滞/超时和分项奖励。无停留模式下，16 维附加目标特征中的原停留进度槽改为
`initial_face_pending`（仅真实 reset 后尚未登记首面时为 1）。
目标切换前后的状态分别保存，终止帧在自动重置前保存；其他任务的数据格式保持不变。

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
