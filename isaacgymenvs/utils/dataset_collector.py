"""Synchronized post-step sensor snapshots, captured before automatic resets."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul

from isaacgymenvs.utils.dataset_writer import HDF5DatasetWriter


def resolve_dataset_env_ids(config, num_envs):
    ids = tuple(int(value) for value in config.get("env_ids", [0]))
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= num_envs for i in ids):
        raise ValueError("dataset.env_ids must contain unique, existing environment IDs")
    for key in ("capture_every", "max_samples", "shard_size", "queue_size", "flush_every"):
        value = config.get(key, {"capture_every": 1, "max_samples": 10000,
                                 "shard_size": 1000, "queue_size": 8, "flush_every": 16}[key])
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError("dataset.{} must be a positive integer".format(key))
    return ids


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=True)
    return np.array(value, dtype=np.float32, copy=True)


def _plain_config(value):
    # The training launcher leaves ListConfig objects nested in ordinary dicts.
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def _relative_poses(poses, reference):
    """T_reference_object from T_parent_object and T_parent_reference (xyzw)."""
    poses = torch.as_tensor(np.array(poses, dtype=np.float32, copy=True))
    reference = torch.as_tensor(np.array(reference, dtype=np.float32, copy=True))
    inverse = quat_conjugate(reference[3:7].unsqueeze(0)).expand(poses.shape[0], -1)
    return torch.cat((quat_apply(inverse, poses[:, :3] - reference[:3]),
                      quat_mul(inverse, poses[:, 3:7])), dim=-1).numpy().copy()


class SensorDatasetCollector:
    """Small CPU snapshots plus a bounded, asynchronous HDF5 writer.

    Each row describes s[t+1] and six RGB views after a completed control step.
    transition/action drove s[t] -> s[t+1]; transition/*_before describes s[t].
    Terminal rows are submitted before reset_idx can overwrite the tensors.
    """

    def __init__(self, task, config):
        self.task, self.config = task, dict(config)
        self.env_ids = tuple(task.dataset_env_ids)
        self.capture_every = int(config.get("capture_every", 1))
        self.max_samples = int(config.get("max_samples", 10000))
        self.control_dt = float(task.dt) * int(task.control_freq_inv)
        self.six_faces = task.cfg["env"].get("rotationRewardMode", "") == "six_faces"
        self.control_step = 0
        self.samples = 0
        self.active = True
        self.complete = False
        self._closed = False
        self._before = {}
        self._episodes = {env_id: 0 for env_id in self.env_ids}
        self._episode_samples = {env_id: 0 for env_id in self.env_ids}
        self._origins = {}
        for env_id in self.env_ids:
            origin = task.gym.get_env_origin(task.envs[env_id])
            self._origins[env_id] = np.asarray([origin.x, origin.y, origin.z], dtype=np.float32)
        self.metadata = self._metadata()
        self.writer = HDF5DatasetWriter(
            config.get("output_dir", "datasets/rapidhand"), self.metadata,
            shard_size=int(config.get("shard_size", 1000)),
            queue_size=int(config.get("queue_size", 8)),
            compression=config.get("compression", "lzf"),
            flush_every=int(config.get("flush_every", 16)))
        print("Sensor dataset: {} | environments {} | limit {} rows".format(
            self.writer.directory, self.env_ids, self.max_samples))

    def _metadata(self):
        task = self.task
        checkpoint = str(self.config.get("checkpoint", ""))
        metadata = dict(
            schema_name="rapidhand_synchronized_sensors_v1", schema_version=1,
            timing={"row": "post-control-step state s[t+1], before automatic reset",
                    "transition_action": "action applied from s[t] to this row's s[t+1]",
                    "transition_before": "state and teacher input before the action",
                    "rgb": "all six sensors rendered together after completed physics",
                    "tactile": "per-pad peak net-force vector over the preceding control interval; different pads may peak at different substeps",
                    "timestamps": "simulation seconds since collection initialization; first completed step at control_dt",
                    "sequence_index": ["index/env_id", "index/episode_id", "index/episode_step"],
                    "rows": "control-step-major then configured environment order; never infer a trajectory from adjacent rows alone",
                    "capture_terminal": True, "capture_every": self.capture_every,
                    "behaviour_cloning": "With capture_every=1, the next row of the same environment/episode supplies the next action for this row's RGB. transition/action belongs to the preceding state, not this RGB."},
            units={"position": "m", "joint_position": "rad", "joint_velocity": "rad/s",
                   "linear_velocity": "m/s", "angular_velocity": "rad/s", "force": "N"},
            pose_layout="position_xyz_m,quaternion_xyzw",
            pose_convention="T_parent_child: maps child-frame points into parent coordinates",
            frames={"env": "Isaac Gym environment-local XYZ; world axes with environment translation removed",
                    "world": "simulation world, pose_env.position + env_origin_world",
                    "hand": "fixed hand BASE link",
                    "camera": "OpenCV optical frame: +X right, +Y down, +Z forward"},
            camera=task.hand_camera.dataset_camera_metadata(),
            joint_names=list(task.dof_names), tactile_link_names=list(task.tactile_link_names),
            tactile_threshold_newtons=task.contact_threshold,
            tactile_force_definition="net contact force, includes object, other hand links and ground; not a pressure image or object-only contact label",
            static_tactile_colours=bool(self.config.get("static_tactile_colors", True)),
            control_dt=self.control_dt, physics_dt=float(task.dt),
            control_frequency_inv=int(task.control_freq_inv),
            selected_env_ids=list(self.env_ids), num_sim_envs=task.num_envs,
            num_objects=task.num_objects, object_scale=float(task.cfg["env"].get("objectScale", 1.0)),
            objects_by_env={str(i): [task.used_training_objects[int(task.object_class_indices[i, 0].item())]] * task.num_objects
                            for i in self.env_ids},
            env_origins_world={str(i): origin.tolist() for i, origin in self._origins.items()},
            object_texture=task.object_texture_metadata,
            hand_urdf=str(task.hand_asset_root / task.hand_asset_file),
            hand_urdf_sha256=file_sha256(task.hand_asset_root / task.hand_asset_file),
            checkpoint=checkpoint, checkpoint_sha256=file_sha256(checkpoint) if checkpoint else None,
            policy_mode=str(self.config.get("mode", "replay")),
            policy_note="During training the policy evolves; actions and its input are saved for every row.",
            seed=int(self.config.get("seed", 42)),
            resolved_task_config=_plain_config(task.cfg),
            image_layout="uint8 RGB, HWC; finger tensor [5,H,W,3], ordered thumb,index,middle,ring,little",
        )
        if self.six_faces:
            metadata["six_face"] = {
                "schema_version": 2,
                "reward_mode": "six_faces",
                "goal": "present each of the six principal object faces to the fixed top camera",
                "repeat_cycles": task.six_face_task.repeat_cycles,
                "cycle_boundary": "completed is a one-step cycle event; in repeat mode covered_after is cleared and a different next target is selected without a physics reset",
                "axis": "env.axis is a compatibility field only; this task has no fixed rotation axis",
                "face_names": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
                "face_normals_object": [[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                                        [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                "timing": "*_before is the action's input goal/state; *_after is the post-step goal/state, including goal switches and before episode reset",
                "step_metrics": "target_error_rad, stable, presented and geometrically_visible refer to target_face_evaluated, the target before a possible switch; stable is diagnostic only when hold_seconds=0",
                "visibility": "geometric proxy based on orientation, camera frustum and projected face area; does not test occlusion by the hand or other objects",
                "counter_units": "hold_steps, max_paid_hold_steps, goal_age_steps and no_progress_steps count control steps; multiply by control_dt for seconds",
                "observation_extension": "16 appended goal-state values; use transition/teacher_observation_before with goal_face_before for action supervision",
                "observation_phase_slot": ("initial_face_pending (1 only before the reset's first face)"
                    if task.six_face_task.hold_required == 0 else "continuous stable hold ratio"),
                "reward_config": _plain_config(task.six_face_config),
                "camera_pose_hand_by_env": {
                    str(i): _array(task.six_face_camera_pose[i]).tolist() for i in self.env_ids
                },
            }
        return metadata

    def _six_face_snapshot(self, env_id):
        state = self.task.six_face_task
        return {
            "goal_face": np.int64(state.goal_face[env_id].item()),
            "covered": _array(state.covered[env_id]).astype(np.uint8),
            "cycles_completed": np.int64(state.cycles_completed[env_id].item()),
            "hold_steps": np.int64(state.hold_steps[env_id].item()),
            "max_paid_hold_steps": np.int64(state.max_paid_hold_steps[env_id].item()),
            "goal_age_steps": np.int64(state.goal_age[env_id].item()),
            "no_progress_steps": np.int64(state.no_progress_steps[env_id].item()),
            "best_error_rad": np.float32(state.best_error[env_id].item()),
            "stall_reference_error_rad": np.float32(state.stall_reference_error[env_id].item()),
        }

    def _snapshot(self, env_id):
        task = self.task
        states = _array(task.root_state_tensor[task.object_indices[env_id]].reshape(-1, 13))
        base_sim = _array(torch.cat((task.hand_base_position[env_id], task.hand_base_quat[env_id])))
        base_env = np.asarray(list(task.cfg["env"].get("handPosition", [0, 0, .30]))
                              + [float(v) for v in base_sim[3:7]], dtype=np.float32)
        root_to_env = base_sim[:3] - base_env[:3]
        pose_env = states[:, :7].copy()
        pose_env[:, :3] -= root_to_env
        pose_world = pose_env.copy()
        pose_world[:, :3] += self._origins[env_id]
        return dict(q=_array(task.dof_pos[env_id]), qdot=_array(task.dof_vel[env_id]),
                    targets=_array(task.cur_targets[env_id]), pose_env=pose_env,
                    pose_world=pose_world, pose_hand=_relative_poses(pose_env, base_env),
                    velocity_env=states[:, 7:13], base_env=base_env)

    def before_step(self):
        if not self.active:
            return
        self.control_step += 1
        self._before = {env_id: self._snapshot(env_id) for env_id in self.env_ids}
        for env_id in self.env_ids:
            self._before[env_id]["teacher_obs"] = _array(torch.clamp(
                self.task.obs_buf[env_id], -self.task.clip_obs, self.task.clip_obs))
            if self.six_faces:
                self._before[env_id]["six_face"] = self._six_face_snapshot(env_id)

    def requested_env_ids(self):
        if not self.active:
            return ()
        remaining = self.max_samples - self.samples
        return tuple(i for i in self.env_ids
                     if self.control_step % self.capture_every == 0
                     or bool(self.task.reset_buf[i].item()))[:remaining]

    def record(self, frames):
        """Receive synchronized RGB and snapshot tensors before any reset."""
        if not self.active:
            return
        task = self.task
        for env_id in self.requested_env_ids():
            if env_id not in frames:
                raise RuntimeError("Missing synchronized dataset cameras for env {}".format(env_id))
            current, before = self._snapshot(env_id), self._before[env_id]
            poses_env = task.hand_camera.get_dataset_camera_poses(env_id)
            poses_world = poses_env.copy()
            poses_world[:, :3] += self._origins[env_id]
            forces = task.tactile_forces[env_id]
            inverse = task.hand_base_inverse_quat[env_id:env_id + 1].expand(task.num_tactile_sensors, -1)
            force_hand = quat_apply(inverse, forces)
            pad_states = _array(task.rigid_body_state_tensor[task.tactile_body_indices[env_id], :7])
            pad_states[:, :3] -= (_array(task.hand_base_position[env_id]) - current["base_env"][:3])
            done = bool(task.reset_buf[env_id].item())
            stalled = bool(task.rotation_stalled[env_id].item()) if task.rotation_stall_enabled else False
            sample = {
                "index/sample_id": np.uint64(self.samples),
                "index/env_id": np.int32(env_id),
                "index/episode_id": np.uint64(self._episodes[env_id]),
                "index/episode_step": np.uint64(task.progress_buf[env_id].item()),
                "index/control_step": np.uint64(self.control_step),
                "index/is_first_sample": np.uint8(self._episode_samples[env_id] == 0),
                "time/sim_seconds": np.float64(self.control_step * self.control_dt),
                "time/episode_seconds": np.float64(task.progress_buf[env_id].item() * self.control_dt),
                "time/tactile_window_start": np.float64((self.control_step - 1) * self.control_dt),
                "time/tactile_window_end": np.float64(self.control_step * self.control_dt),
                "rgb/finger": frames[env_id]["finger_rgb"],
                "rgb/top": frames[env_id]["top_rgb"],
                "state/joint_position": current["q"],
                "state/joint_velocity": current["qdot"],
                "state/joint_targets": current["targets"],
                "tactile/contact": _array(task.tactile_contacts[env_id]).astype(np.uint8),
                "tactile/peak_force_norm": _array(task.tactile_force_norm[env_id]),
                "tactile/peak_force_env": _array(forces),
                "tactile/peak_force_hand": _array(force_hand),
                "tactile/sensor_pose_hand": _relative_poses(pad_states, current["base_env"]),
                "object/pose_env": current["pose_env"],
                "object/pose_world": current["pose_world"],
                "object/pose_hand": current["pose_hand"],
                "object/pose_top_camera": _relative_poses(current["pose_env"], poses_env[-1]),
                "object/velocity_env": current["velocity_env"],
                "hand/pose_env": current["base_env"],
                "camera/pose_env": poses_env,
                "camera/pose_world": poses_world,
                "camera/pose_hand": _relative_poses(poses_env, current["base_env"]),
                "camera/view_matrix": task.hand_camera.get_dataset_camera_views(env_id),
                "transition/action": _array(task.actions[env_id]),
                "transition/teacher_observation_before": before["teacher_obs"],
                "transition/joint_position_before": before["q"],
                "transition/joint_velocity_before": before["qdot"],
                "transition/joint_targets_before": before["targets"],
                "transition/object_pose_hand_before": before["pose_hand"],
                "transition/reward": np.float32(task.rew_buf[env_id].item()),
                "transition/done": np.uint8(done),
                "transition/timeout": np.uint8(bool(task.timeout_buf[env_id].item()) and not stalled),
                "transition/stalled": np.uint8(stalled),
                "quality/finite_state": np.uint8(all(np.isfinite(value).all()
                    for value in (current["q"], current["qdot"], current["pose_env"],
                                  _array(forces), poses_env))),
            }
            if self.six_faces:
                after_goal = self._six_face_snapshot(env_id)
                for suffix, state in (("before", before["six_face"]), ("after", after_goal)):
                    for name, value in state.items():
                        sample["six_face/{}_{}".format(name, suffix)] = value
                result = task.six_face_result
                for name in ("goal_changed", "completed", "stalled", "goal_timeout",
                             "stable", "presented", "terminal_success", "geometrically_visible"):
                    sample["six_face/" + name] = np.uint8(result[name][env_id].item())
                sample["six_face/target_face_evaluated"] = np.int64(result["target_face_before"][env_id].item())
                sample["six_face/target_error_rad"] = np.float32(result["target_error"][env_id].item())
                for name in ("progress_reward", "hold_reward", "face_bonus",
                             "completion_bonus", "timeout_penalty"):
                    sample["six_face/reward/" + name] = np.float32(result[name][env_id].item())
                sample["transition/stalled"] = sample["six_face/stalled"]
                sample["six_face/terminal"] = np.uint8(task.six_face_terminal[env_id].item())
            self.writer.submit(sample)
            self.samples += 1
            self._episode_samples[env_id] += 1
        self._before.clear()
        if self.samples >= self.max_samples:
            self.active = False
            self.complete = True
            self.writer.close()
            print("Sensor dataset complete: {} rows in {}".format(self.samples, self.writer.directory))

    def on_reset(self, env_ids):
        # Called for automatic, reset_done and explicit user resets alike.
        for value in env_ids.detach().cpu().tolist():
            if value in self._episodes:
                self._episodes[value] += 1
                self._episode_samples[value] = 0

    def close(self):
        if not self._closed:
            self.active = False
            self._before.clear()
            self.writer.close()
            self._closed = True
