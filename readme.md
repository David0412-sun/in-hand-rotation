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
