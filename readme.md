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

Activate `robosyn`, then train from scratch (default: 256 environments):

```bash
bash scripts/teacher_rapidhand.sh 0 max_iterations=2000
```

The single RapidHand task entry point is `task=RapidHandRotation`, selected by
`scripts/teacher_rapidhand.sh`. Its default is `cross4_0`, Z rotation.
Use `task.env.objSet` to choose `cross`, `C`, `ball` or `custom`, `task.env.objectName`
to choose a specific object (empty means the whole set), and `task.env.axis`
to select the rotation axis. Without an explicit object name, `cross` selects
`cross4_0`; `C` selects its full set and `ball` selects its sole sphere type.
For example:

| Selection | Command |
| --- | --- |
| Single cross | `bash scripts/teacher_rapidhand.sh 0 task.env.objectName=cross4_1` |
| Single C object | `bash scripts/teacher_rapidhand.sh 0 task.env.objSet=C task.env.objectName=set_obj1_regular_block` |
| All cross shapes | `bash scripts/teacher_rapidhand.sh 0 task.env.objectName=` |
| All C shapes | `bash scripts/teacher_rapidhand.sh 0 task.env.objSet=C task.env.objectName=` |
| Two balls | `bash scripts/teacher_rapidhand.sh 0 task.env.objSet=ball` |

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
bash scripts/teacher_rapidhand.sh 0 task.env.objSet=ball max_iterations=10 headless=False force_render=True
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

For the source 60 mm chamfered block, select
`isaacgymenvs/cfg/object/block_corner_60mm.yaml`:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=block_corner_60mm \
  max_iterations=10 headless=False force_render=True
```

This preset selects `set_obj6_block_corner`, the simplified-collision hand URDF
and self-collision. It uses the previous 60 mm cube test's open-hand joint pose,
with the object moved 30 mm toward the fingers along hand-local +Z to
`[0.005, -0.0715, 0.14]`. Its spin reward coefficient is 2.0; other reward weights
inherit the common defaults.
After a 2 s settling period, each environment resets if its signed orientation
increments about the target axis sum to less than 5 degrees in magnitude over
the last 3 s. Back-and-forth motion cancels in this check; a short pause for
regrasping is allowed. Stalls terminate the episode without an additional penalty.
The same reset rule applies in training and replay. `rotationStallTimeout=0`
disables it, which remains the default for other objects and the two-ball task.
It sets PhysX buffers to `1.0 / 4194304` and inherits 256 environments and both
PPO minibatches of 1024. Outputs stay in
`runs/rapidhand/set_obj6_block_corner/z-axis/{nn,summaries,config.yaml}`.
Use the same object preset during replay. The short run is intended to check
the initial grasp, tactile response and training pipeline for this combination.

The ideal `box_50x50x50` is a 50 x 50 x 50 mm cube, with one rigid body and
one primitive box collision shape. It belongs to the RapidHand `custom` set.
Select its object preset to use a matching initial position and open-hand pose:

```bash
bash scripts/teacher_rapidhand.sh 0 +object=box_50x50x50 \
  task.env.numEnvs=256 max_iterations=10 headless=True force_render=False \
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

Object files and settings are stored here:

| Content | Location |
| --- | --- |
| Box geometry, reference mass and inertia | `assets/urdf/objects/box_50x50x50.urdf` |
| All object URDFs, including `ball.urdf`, `cross4_*.urdf` and `set_obj*.urdf` | `assets/urdf/objects/` |
| Cross visual/collision meshes | `assets/urdf/objects/meshes/non-convex-param/` |
| C-set visual/collision meshes | `assets/urdf/objects/meshes/set2/` |
| Object lists and output-directory routing | `isaacgymenvs/utils/object_tasks.py` |
| Common RapidHand physics, reward and control settings | `isaacgymenvs/cfg/task/RapidHandRotation.yaml` |
| Common RapidHand PPO training settings | `isaacgymenvs/cfg/train/RapidHandRotationPPO.yaml` |

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
