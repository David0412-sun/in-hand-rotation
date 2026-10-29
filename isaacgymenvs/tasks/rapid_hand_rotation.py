"""GPU teacher environment for the fixed-base RapidHand Vision hand.

The actor observes four frames of [normalised q, normalised position targets,
binary pad contacts, local spin axis], followed by each current 13-D object
state in the hand base frame.  The central critic observes current
[q, qdot, targets, actions, object state, local axis, peak pad force vectors].
objSet selects one cross/C/custom object or two Baoding balls.
"""

import json
import math
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul
import numpy as np
import torch

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.object_tasks import OBJECT_SETS, select_objects


class RapidHandRotation(VecTask):
    """Independent hand task; it does not inherit Allegro-specific control."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        self.cfg = cfg
        self.hand_camera = None
        self.camera_config = cfg.get("camera", {})
        self.camera_env_ids = ()
        if self.camera_config.get("enabled", False):
            from isaacgymenvs.utils.hand_camera import resolve_env_ids
            if self.camera_config.get("display", False) and headless:
                raise ValueError("camera.display=True requires headless=False for selecting a hand in the Viewer")
            if graphics_device_id < 0:
                raise ValueError("Finger cameras require graphics_device_id >= 0, including headless recording")
            self.camera_env_ids = resolve_env_ids(self.camera_config, int(cfg["env"]["numEnvs"]))
            # Env reads this top-level flag before creating the graphics context.
            cfg["enableCameraSensors"] = True
        env = cfg["env"]
        if env.get("handName", "vision_hand") != "vision_hand":
            raise ValueError("RapidHandRotation currently supports vision_hand only")
        if not env.get("legacy_obs", True) or env.get("is_distillation", False):
            raise ValueError("RapidHandRotation currently requires legacy_obs=True and teacher mode")
        if env.get("ablation_mode", "no-pc") != "no-pc":
            raise ValueError("RapidHandRotation currently supports ablation_mode=no-pc only")
        if not env.get("asymmetric_observations", True):
            raise ValueError("RapidHandRotation requires asymmetric_observations=True")
        if cfg["sim"]["up_axis"] != "z":
            raise ValueError("RapidHandRotation requires a z-up simulation")

        self.project_root = Path(__file__).resolve().parents[2]
        self.hand_asset_root = self._asset_path(env.get("handAssetRoot", "assets/rapidhand"))
        self.object_asset_root = self._asset_path(env.get("assetRoot", "assets"))
        with (self.hand_asset_root / "manifest.json").open() as manifest_file:
            manifest = json.load(manifest_file)["hands"]["vision_hand"]
        self.hand_asset_file = env.get("handAssetFile", manifest["asset_file"])
        hand_xml = ET.parse(str(self.hand_asset_root / self.hand_asset_file)).getroot()
        self.expected_dof_names = list(manifest["joint_names"])
        self.num_hand_dofs = int(manifest["num_dofs"])
        self.tactile_link_names = [link.get("name") for link in hand_xml.findall("link")
                                   if "tactile" in link.get("name", "").lower()]
        self.hand_link_colors = self._link_visual_colors(hand_xml)
        self.self_collision = bool(env.get("selfCollision", True))
        self.hand_collision_filters = self._self_collision_filters(
            hand_xml, env.get("selfCollisionIgnorePairs", []))
        self.num_tactile_sensors = len(self.tactile_link_names)
        self.fingertip_link_names = list(manifest["fingertip_links"])
        if self.num_hand_dofs != 20 or self.num_tactile_sensors != 15:
            raise ValueError("Vision hand must contain 20 DOFs and 15 tactile links")
        for link in hand_xml.findall("link"):
            if link.get("name") in self.tactile_link_names and link.find("collision") is None:
                raise ValueError("Tactile link has no collision geometry: " + link.get("name"))

        self._configure_objects(env)
        self.axis = str(env.get("axis", "z"))
        if self.axis not in ("x", "y", "z"):
            raise ValueError("axis must be x, y or z")
        self.frame_stack = int(env.get("frameStack", 4))
        self.max_episode_length = int(env.get("episodeLength", 500))
        if self.frame_stack < 1 or self.max_episode_length < 2:
            raise ValueError("frameStack must be positive and episodeLength at least two")
        self.frame_obs_dim = 2 * self.num_hand_dofs + self.num_tactile_sensors + 3
        self.frame_obs_slices = self._slices(("q", self.num_hand_dofs),
                                           ("targets", self.num_hand_dofs),
                                           ("touch", self.num_tactile_sensors), ("axis", 3))
        self.obs_slices = dict(self.frame_obs_slices)
        history_dim = self.frame_stack * self.frame_obs_dim
        self.obs_slices.update(history=slice(0, history_dim),
                               object=slice(history_dim, history_dim + self.object_state_dim))
        self.state_slices = self._slices(("q", self.num_hand_dofs), ("velocity", self.num_hand_dofs),
                                       ("targets", self.num_hand_dofs), ("actions", self.num_hand_dofs),
                                       ("object", self.object_state_dim), ("axis", 3),
                                       ("tactile_forces", 3 * self.num_tactile_sensors))
        env["numActions"] = self.num_hand_dofs
        env["numObservations"] = history_dim + self.object_state_dim
        env["numStates"] = 4 * self.num_hand_dofs + self.object_state_dim + 3 + 3 * self.num_tactile_sensors
        self.is_distillation = False
        self.ablation_mode = "no-pc"
        self.debug_viz = bool(env.get("enableDebugVis", False))
        self.contact_threshold = float(env.get("contactThreshold", 0.05))
        self.action_scale = float(env.get("actionScale", 0.1))
        self.target_smoothing = float(env.get("targetSmoothing", 0.8))
        if self.contact_threshold <= 0 or self.action_scale <= 0 or not 0 < self.target_smoothing <= 1:
            raise ValueError("Invalid contact threshold, action scale or target smoothing")
        self.rotation_reward_mode = str(env.get("rotationRewardMode", "angular_velocity"))
        if self.rotation_reward_mode not in ("angular_velocity", "pose_delta"):
            raise ValueError("rotationRewardMode must be angular_velocity or pose_delta")
        self.pose_rotation_reward = self.rotation_reward_mode == "pose_delta"
        if self.pose_rotation_reward and self.num_objects != 1:
            raise ValueError("pose_delta reward is only supported for single objects")
        self.centered_rotation_reward = bool(env.get("centeredRotationReward", False))
        if self.centered_rotation_reward:
            if not self.pose_rotation_reward:
                raise ValueError("centeredRotationReward requires rotationRewardMode=pose_delta")
            self.positive_spin_reward_limit = float(env.get("positiveSpinRewardLimit", 0.6))
            self.position_tolerance = float(env.get("positionTolerance", 0.005))
            self.position_penalty_distance = float(env.get("positionPenaltyDistance", 0.01))
            self.position_reward_falloff = float(env.get("positionRewardFalloff", 0.015))
            positive_scales = (self.positive_spin_reward_limit, self.position_penalty_distance,
                               self.position_reward_falloff)
            if (not all(math.isfinite(value) and value > 0 for value in positive_scales)
                    or not math.isfinite(self.position_tolerance) or self.position_tolerance < 0):
                raise ValueError("Invalid centered rotation reward limits or distances")
        self.rotation_control_dt = float(cfg["sim"]["dt"]) * int(env.get("controlFrequencyInv", 1))
        if not math.isfinite(self.rotation_control_dt) or self.rotation_control_dt <= 0:
            raise ValueError("Rotation control interval must be positive and finite")
        self.rotation_stall_forward_only = bool(env.get("rotationStallForwardOnly", False))
        stall_timeout = float(env.get("rotationStallTimeout", 0.0))
        stall_grace = float(env.get("rotationStallGracePeriod", 2.0))
        stall_angle = float(env.get("rotationStallMinAngleDeg", 5.0))
        if (not all(math.isfinite(v) for v in (stall_timeout, stall_grace, stall_angle))
                or stall_timeout < 0 or stall_grace < 0 or stall_angle <= 0):
            raise ValueError("Invalid rotation stall timeout, grace period or angle")
        self.rotation_stall_enabled = stall_timeout > 0
        if self.rotation_stall_enabled:
            if self.num_objects != 1:
                raise ValueError("Rotation stall detection currently supports single objects only")
            self.rotation_stall_steps = max(1, math.ceil(stall_timeout / self.rotation_control_dt))
            self.rotation_stall_grace_steps = math.ceil(stall_grace / self.rotation_control_dt)
            self.rotation_stall_min_angle = math.radians(stall_angle)

        try:
            super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless,
                             virtual_screen_capture, force_render)
            self._acquire_tensors()
            self.reset()
        except BaseException:
            try:
                self.close()
            except Exception as error:
                warnings.warn("Cleanup after initialization failure: {}".format(error))
            raise
        print("{}: {} DOFs, {} pads, actor {}, critic {}, objects {}".format(
            type(self).__name__, self.num_actions, self.num_tactile_sensors, self.num_obs, self.num_states,
            ", ".join(self.used_training_objects)))

    def _configure_objects(self, env):
        self.object_set_id = str(env.get("objSet", "cross"))
        if self.object_set_id not in ("cross", "C", "ball", "custom"):
            raise ValueError("RapidHandRotation supports objSet=cross, C, ball or custom")
        if self.object_set_id == "ball" and str(env.get("axis", "z")) != "z":
            raise ValueError("The two-ball task currently supports axis=z only")
        self.used_training_objects = select_objects(self.object_set_id, env.get("objectName", ""))
        self.num_training_objects = len(self.used_training_objects)
        self.object_sets = OBJECT_SETS
        self.num_objects = 2 if self.object_set_id == "ball" else 1
        self.object_state_dim = 13 * self.num_objects

    def _asset_path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.project_root / path

    @staticmethod
    def _slices(*fields):
        result, start = {}, 0
        for name, width in fields:
            result[name] = slice(start, start + width)
            start += width
        return result

    @staticmethod
    def _quat_from_euler(euler):
        roll, pitch, yaw = (float(value) for value in euler)
        # Isaac Gym's binding takes (roll, pitch, yaw), despite the method name.
        return gymapi.Quat.from_euler_zyx(roll, pitch, yaw)

    @staticmethod
    def _link_visual_colors(hand_xml):
        # Alternate along the URDF tree, including fixed links. Red and green
        # are reserved for tactile state so ordinary links cannot mimic a pad.
        palette = ((0.15, 0.40, 0.95), (1.0, 0.65, 0.12),
                   (0.65, 0.25, 0.90), (0.15, 0.80, 0.90))
        children = {link.get("name"): [] for link in hand_xml.findall("link")}
        child_links = set()
        for joint in hand_xml.findall("joint"):
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            children[parent].append(child)
            child_links.add(child)
        pending = [(name, 0) for name in children if name not in child_links]
        colors = {}
        while pending:
            name, depth = pending.pop()
            colors[name] = (1.0, 0.0, 0.0) if "tactile" in name.lower() else palette[depth % len(palette)]
            pending.extend((child, depth + 1) for child in children[name])
        return colors

    @staticmethod
    def _self_collision_filters(hand_xml, ignored_pairs):
        # Keep tactile bodies separate for sensing, but filter contacts within
        # each fixed assembly and across each directly connected joint. One
        # unique bit per moving joint excludes exactly those assembly pairs.
        joints = hand_xml.findall("joint")
        fixed_parent = {joint.find("child").get("link"): joint.find("parent").get("link")
                        for joint in joints if joint.get("type") == "fixed"}

        def assembly(name):
            while name in fixed_parent:
                name = fixed_parent[name]
            return name

        link_assembly = {link.get("name"): assembly(link.get("name"))
                         for link in hand_xml.findall("link")}
        masks = {name: 0 for name in link_assembly.values()}
        edges = [(assembly(joint.find("parent").get("link")),
                  assembly(joint.find("child").get("link")))
                 for joint in joints if joint.get("type") != "fixed"]
        # The two-axis finger mounts also overlap the palm across their small
        # intermediate link. Explicit assembly pairs handle these exceptions.
        for first, second in ignored_pairs:
            if first not in link_assembly or second not in link_assembly:
                raise ValueError("Unknown self-collision exclusion: " + str((first, second)))
            edges.append((assembly(first), assembly(second)))
        if len(edges) > 31:
            raise ValueError("Hand self-collision filtering exceeds the available mask bits")
        for bit, (parent, child) in enumerate(edges):
            masks[parent] |= 1 << bit
            masks[child] |= 1 << bit
        if any(mask == 0 for mask in masks.values()):
            raise ValueError("Hand contains an isolated assembly without a collision filter")
        return {name: masks[group] for name, group in link_assembly.items()}

    def allocate_buffers(self):
        # VecTask's default allocator contains Allegro observation assumptions.
        shape = self.num_envs
        self.obs_buf = torch.zeros((shape, self.num_obs), device=self.device)
        self.states_buf = torch.zeros((shape, self.num_states), device=self.device)
        self.rew_buf = torch.zeros(shape, device=self.device)
        self.reset_buf = torch.ones(shape, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.actions = torch.zeros((shape, self.num_actions), device=self.device)
        self.prev_targets = torch.zeros_like(self.actions)
        self.cur_targets = torch.zeros_like(self.actions)
        self.obs_history = torch.zeros((shape, self.frame_stack, self.frame_obs_dim), device=self.device)
        self._history_needs_reset = torch.ones(shape, device=self.device, dtype=torch.bool)
        self.tactile_forces = torch.zeros((shape, self.num_tactile_sensors, 3), device=self.device)
        self.tactile_force_norm = torch.zeros((shape, self.num_tactile_sensors), device=self.device)
        self.tactile_contacts = torch.zeros_like(self.tactile_force_norm)
        self.spin_axis = torch.zeros((shape, 3), device=self.device)
        self.spin_axis[:, ("x", "y", "z").index(self.axis)] = 1.0
        if self.num_objects == 2:
            self.previous_ball_vector = torch.zeros((shape, 3), device=self.device)
            self.orbit_angle_accumulator = torch.zeros(shape, device=self.device)
        if self.rotation_stall_enabled or self.pose_rotation_reward:
            self.previous_object_quat = torch.zeros((shape, 4), device=self.device)
        if self.pose_rotation_reward:
            self.rotation_episode_sums = {
                key: torch.zeros(shape, device=self.device)
                for key in ("angle", "sampled_spin", "spin_reward", "reward", "reverse_steps")
            }
        if self.rotation_stall_enabled:
            self.rotation_step_angles = torch.zeros((self.rotation_stall_steps, shape), device=self.device)
            self.rotation_window_index = 0
            self.rotation_stalled = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.extras = {}
        self._physics_substep = 0

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)
        ground = gymapi.PlaneParams()
        ground.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, ground)
        self._create_envs()
        if self.camera_config.get("enabled", False):
            from isaacgymenvs.utils.hand_camera import HandCameraMonitor
            self.hand_camera = HandCameraMonitor(self, self.camera_config)
        return self.sim

    def _create_envs(self):
        env_cfg = self.cfg["env"]
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        options.collapse_fixed_joints = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        options.armature = 0.001
        options.thickness = 0.001
        options.angular_damping = 0.01
        hand_asset = self.gym.load_asset(self.sim, str(self.hand_asset_root), self.hand_asset_file, options)
        if hand_asset is None:
            raise RuntimeError("Failed to load Vision hand")
        if self.self_collision:
            shape_props = self.gym.get_asset_rigid_shape_properties(hand_asset)
            shape_ranges = self.gym.get_asset_rigid_body_shape_indices(hand_asset)
            for name, shape_range in zip(self.gym.get_asset_rigid_body_names(hand_asset), shape_ranges):
                for shape_index in range(shape_range.start, shape_range.start + shape_range.count):
                    shape_props[shape_index].filter = self.hand_collision_filters[name]
            self.gym.set_asset_rigid_shape_properties(hand_asset, shape_props)
        self.dof_names = list(self.gym.get_asset_dof_names(hand_asset))
        if len(self.dof_names) != self.num_actions or set(self.dof_names) != set(self.expected_dof_names):
            raise RuntimeError("Imported DOFs differ from the Vision manifest: " + str(self.dof_names))
        self.dof_name_to_index = {name: index for index, name in enumerate(self.dof_names)}
        props = self.gym.get_asset_dof_properties(hand_asset)
        if not np.all(np.isfinite(props["lower"])) or not np.all(props["lower"] < props["upper"]):
            raise ValueError("Vision hand has invalid DOF limits")
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(float(env_cfg.get("stiffness", 3.0)))
        props["damping"].fill(float(env_cfg.get("damping", 0.5)))
        props["friction"].fill(0.01)
        # Keep the URDF's 1 Nm effort limits; do not apply Allegro's 20 Nm controller.
        if not np.all(np.isfinite(props["effort"])) or not np.all(props["effort"] > 0):
            raise ValueError("Vision hand requires positive URDF effort limits")
        default_q = np.clip(np.zeros(self.num_actions, dtype=np.float32), props["lower"], props["upper"])
        for name, value in env_cfg.get("defaultJointPositions", {}).items():
            if name not in self.dof_name_to_index:
                raise ValueError("Unknown initial joint name: " + name)
            index = self.dof_name_to_index[name]
            default_q[index] = np.clip(float(value), props["lower"][index], props["upper"][index])
        self.dof_lower_limits = torch.tensor(props["lower"].copy(), device=self.device)
        self.dof_upper_limits = torch.tensor(props["upper"].copy(), device=self.device)
        self.default_dof_pos = torch.tensor(default_q, device=self.device)
        self.num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)

        object_options = gymapi.AssetOptions()
        object_options.disable_gravity = False
        object_options.fix_base_link = False
        object_options.collapse_fixed_joints = True
        object_options.angular_damping = float(env_cfg.get("objectAngularDamping", 0.01))
        self.object_asset_dict = {}
        for name in self.used_training_objects:
            filename = "urdf/objects/" + name + ".urdf"
            asset = self.gym.load_asset(self.sim, str(self.object_asset_root), filename, object_options)
            if asset is None or self.gym.get_asset_dof_count(asset) != 0:
                raise RuntimeError("Object must be a successfully imported rigid asset: " + name)
            self.object_asset_dict[name] = asset

        aggregate_mode = int(env_cfg.get("aggregateMode", 1))
        hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        aggregate_sizes = {
            name: (self.num_hand_bodies + self.num_objects * self.gym.get_asset_rigid_body_count(asset),
                   hand_shapes + self.num_objects * self.gym.get_asset_rigid_shape_count(asset))
            for name, asset in self.object_asset_dict.items()
        }

        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(*env_cfg.get("handPosition", [0, 0, 0.30]))
        hand_pose.r = self._quat_from_euler(env_cfg.get("handRotationEuler", [-math.pi / 2, 0, 0]))
        object_poses = self._object_start_poses(hand_pose)
        if len(object_poses) != self.num_objects:
            raise ValueError("Object pose count does not match the task")
        spacing = float(env_cfg.get("envSpacing", 0.4))
        lower, upper = gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing)
        columns = max(1, int(math.sqrt(self.num_envs)))
        self.envs, self.hands, self.objects = [], [], []
        hand_indices, object_indices, class_indices = [], [], []
        tactile_indices, fingertip_indices, initial_object_states = [], [], []
        self.tactile_body_handles = []
        self._visual_env_ids = tuple(range(self.num_envs)) if not self.headless else self.camera_env_ids
        self._tactile_visual_state = (np.zeros((len(self._visual_env_ids), self.num_tactile_sensors), dtype=bool)
                                      if self._visual_env_ids else None)
        friction = float(env_cfg.get("friction", 1.0))
        mass = float(env_cfg.get("objectMass", 0.05))
        scale = float(env_cfg.get("objectScale", 1.0))
        if mass <= 0 or scale <= 0:
            raise ValueError("objectMass and objectScale must be positive")
        for index in range(self.num_envs):
            class_index = index % self.num_training_objects
            name = self.used_training_objects[class_index]
            env = self.gym.create_env(self.sim, lower, upper, columns)
            if aggregate_mode >= 1:
                # Group the hand and all manipulated objects, as in Allegro.
                # True keeps intra-aggregate collisions; shape filters still apply.
                bodies, shapes = aggregate_sizes[name]
                if not self.gym.begin_aggregate(env, bodies, shapes, True):
                    raise RuntimeError("Failed to begin aggregate for environment {}".format(index))
            # -1 preserves our per-shape assembly filters; 1 disables all self
            # collisions for reproducing older runs. Object filter 0 contacts both.
            hand_filter = -1 if self.self_collision else 1
            hand = self.gym.create_actor(env, hand_asset, hand_pose, "hand", index, hand_filter)
            self.gym.set_actor_dof_properties(env, hand, props)
            if not self.headless or index in self.camera_env_ids:
                for link_name, color in self.hand_link_colors.items():
                    body = self.gym.find_actor_rigid_body_handle(env, hand, link_name)
                    self.gym.set_rigid_body_color(env, hand, body, gymapi.MESH_VISUAL_AND_COLLISION,
                                                  gymapi.Vec3(*color))
            env_objects, env_object_indices, env_initial_states = [], [], []
            for slot, object_pose in enumerate(object_poses):
                actor_name = "object" if self.num_objects == 1 else "object_{}".format(slot)
                obj = self.gym.create_actor(env, self.object_asset_dict[name], object_pose, actor_name, index, 0)
                self.gym.set_actor_scale(env, obj, scale)
                if self.num_objects > 1 and (not self.headless or index in self.camera_env_ids):
                    color = (0.95, 0.95, 0.95) if slot == 0 else (1.0, 0.72, 0.10)
                    for body in range(self.gym.get_actor_rigid_body_count(env, obj)):
                        self.gym.set_rigid_body_color(env, obj, body, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(*color))
                body_props = self.gym.get_actor_rigid_body_properties(env, obj)
                total_mass = sum(body.mass for body in body_props)
                if total_mass <= 0:
                    raise ValueError("Imported object has nonpositive mass: " + name)
                for body in body_props:
                    body.mass *= mass / total_mass
                self.gym.set_actor_rigid_body_properties(env, obj, body_props, True)
                env_objects.append(obj)
                env_object_indices.append(self.gym.get_actor_index(env, obj, gymapi.DOMAIN_SIM))
                env_initial_states.append([object_pose.p.x, object_pose.p.y, object_pose.p.z,
                                           object_pose.r.x, object_pose.r.y, object_pose.r.z, object_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            for actor in [hand] + env_objects:
                shape_props = self.gym.get_actor_rigid_shape_properties(env, actor)
                for shape_prop in shape_props:
                    shape_prop.friction = friction
                self.gym.set_actor_rigid_shape_properties(env, actor, shape_props)
            if aggregate_mode >= 1:
                if not self.gym.end_aggregate(env):
                    raise RuntimeError("Failed to end aggregate for environment {}".format(index))
            hand_indices.append(self.gym.get_actor_index(env, hand, gymapi.DOMAIN_SIM))
            object_indices.append(env_object_indices[0] if self.num_objects == 1 else env_object_indices)
            class_indices.append([class_index])
            tactile_handles = [self.gym.find_actor_rigid_body_handle(env, hand, name)
                               for name in self.tactile_link_names]
            tip_handles = [self.gym.find_actor_rigid_body_handle(env, hand, name)
                           for name in self.fingertip_link_names]
            if min(tactile_handles + tip_handles) < 0:
                raise RuntimeError("Imported Vision hand is missing a tactile or fingertip body")
            tactile_indices.append([self.gym.get_actor_rigid_body_index(env, hand, handle, gymapi.DOMAIN_SIM)
                                    for handle in tactile_handles])
            fingertip_indices.append([self.gym.get_actor_rigid_body_index(env, hand, handle, gymapi.DOMAIN_SIM)
                                      for handle in tip_handles])
            self.tactile_body_handles.append(tactile_handles)
            initial_object_states.append(env_initial_states[0] if self.num_objects == 1 else env_initial_states)
            self.envs.append(env)
            self.hands.append(hand)
            self.objects.append(env_objects[0] if self.num_objects == 1 else env_objects)
        self.hand_indices = torch.tensor(hand_indices, device=self.device, dtype=torch.long)
        self.object_indices = torch.tensor(object_indices, device=self.device, dtype=torch.long)
        self.object_class_indices = torch.tensor(class_indices, device=self.device, dtype=torch.long)
        self.tactile_body_indices = torch.tensor(tactile_indices, device=self.device, dtype=torch.long)
        self.fingertip_body_indices = torch.tensor(fingertip_indices, device=self.device, dtype=torch.long)
        self.object_init_state = torch.tensor(initial_object_states, device=self.device)
        self.hand_base_position = torch.tensor([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z],
                                               device=self.device).repeat(self.num_envs, 1)
        self.hand_base_quat = torch.tensor([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w],
                                           device=self.device).repeat(self.num_envs, 1)
        self.hand_base_inverse_quat = quat_conjugate(self.hand_base_quat)
        normal = torch.tensor(env_cfg.get("palmNormalLocal", [0, -1, 0]), device=self.device, dtype=torch.float)
        if torch.linalg.vector_norm(normal).item() <= 0:
            raise ValueError("palmNormalLocal cannot be zero")
        self.palm_normal_local = normal / torch.linalg.vector_norm(normal)
        self.axis_local = quat_apply(self.hand_base_inverse_quat, self.spin_axis)

    def _object_start_poses(self, hand_pose):
        env = self.cfg["env"]
        if self.num_objects == 2:
            positions = env.get("ballPositionsLocal", [[-0.020, -0.055, 0.11], [0.030, -0.055, 0.11]])
            if len(positions) != 2 or any(len(position) != 3 for position in positions):
                raise ValueError("ballPositionsLocal must contain exactly two xyz centres")
            if not all(math.isfinite(float(value)) for position in positions for value in position):
                raise ValueError("ballPositionsLocal must contain finite coordinates")
            radius = float(env.get("ballRadius", 0.022)) * float(env.get("objectScale", 1.0))
            separation = math.sqrt(sum((float(positions[1][i]) - float(positions[0][i])) ** 2 for i in range(3)))
            if radius <= 0 or separation < 2 * radius:
                raise ValueError("Initial balls must have a positive radius and non-overlapping centres")
        else:
            positions = [env.get("objectPositionLocal", [0.005, -0.055, 0.11])]
        poses = []
        for position in positions:
            pose = gymapi.Transform()
            pose.p = hand_pose.p + hand_pose.r.rotate(gymapi.Vec3(*(float(value) for value in position)))
            pose.r = self._quat_from_euler(env.get("objectRotationEuler", [0, 0, 0]))
            poses.append(pose)
        return poses

    def _object_states_in_hand_frame(self, states):
        # Handle one object [N,13] and several objects [N,K,13] without changing
        # the single-object observation layout or ordering.
        flat = states.reshape(-1, 13)
        inverse = self.hand_base_inverse_quat[:, None, :].expand(-1, self.num_objects, -1).reshape(-1, 4)
        base = self.hand_base_position[:, None, :].expand(-1, self.num_objects, -1).reshape(-1, 3)
        local = torch.cat((quat_apply(inverse, flat[:, :3] - base),
                           quat_mul(inverse, flat[:, 3:7]),
                           quat_apply(inverse, flat[:, 7:10]),
                           quat_apply(inverse, flat[:, 10:13])), dim=-1)
        return local.reshape(states.shape)

    def set_viewer(self):
        super().set_viewer()
        if self.viewer is not None:
            env = self.cfg["env"]
            self.gym.viewer_camera_look_at(
                self.viewer, self.envs[0],
                gymapi.Vec3(*env.get("viewerPosition", [0.45, -0.45, 0.65])),
                gymapi.Vec3(*env.get("viewerTarget", [0.0, 0.08, 0.34])))
        if self.hand_camera is not None:
            self.hand_camera.bind_viewer(self.viewer)

    def on_viewer_event(self, event):
        if self.hand_camera is not None:
            self.hand_camera.handle_event(event)

    def before_viewer_draw(self):
        if self.hand_camera is not None:
            self.hand_camera.poll()
            self.hand_camera.draw_selection()

    def _acquire_tensors(self):
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.rigid_body_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.net_contact_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        if self.dof_state_tensor.shape[0] != self.num_envs * self.num_actions:
            raise RuntimeError("Expected only Vision hand DOFs in every environment")
        self.dof_state = self.dof_state_tensor.view(self.num_envs, self.num_actions, 2)
        self.dof_pos, self.dof_vel = self.dof_state[..., 0], self.dof_state[..., 1]
        self._refresh_state()
        # Capture the simulator's coordinates rather than assuming whether the
        # tensor API includes each environment's grid origin.
        self.object_init_state = self.root_state_tensor[self.object_indices].clone()
        hand_roots = self.root_state_tensor[self.hand_indices]
        self.hand_base_position = hand_roots[:, :3].clone()
        self.hand_base_quat = hand_roots[:, 3:7].clone()
        self.hand_base_inverse_quat = quat_conjugate(self.hand_base_quat)
        self.object_reference_local = self._object_states_in_hand_frame(self.object_init_state)[..., :3].clone()
        self.axis_local = quat_apply(self.hand_base_inverse_quat, self.spin_axis)

    def _refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _collect_tactile(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        forces = self.net_contact_force_tensor[self.tactile_body_indices]
        norms = torch.linalg.vector_norm(forces, dim=-1)
        stronger = norms > self.tactile_force_norm
        self.tactile_forces.copy_(torch.where(stronger.unsqueeze(-1), forces, self.tactile_forces))
        self.tactile_force_norm.copy_(torch.maximum(self.tactile_force_norm, norms))
        self.tactile_contacts.copy_((self.tactile_force_norm >= self.contact_threshold).float())

    def _update_tactile_colors(self):
        if self._tactile_visual_state is None:
            return
        contacts = self.tactile_contacts[list(self._visual_env_ids)].to(device="cpu", dtype=torch.bool).numpy()
        for row_id, pad_id in np.argwhere(contacts != self._tactile_visual_state):
            env_id = self._visual_env_ids[row_id]
            color = gymapi.Vec3(0, 1, 0) if contacts[row_id, pad_id] else gymapi.Vec3(1, 0, 0)
            self.gym.set_rigid_body_color(
                self.envs[env_id], self.hands[env_id], self.tactile_body_handles[env_id][pad_id],
                gymapi.MESH_VISUAL_AND_COLLISION, color)
        self._tactile_visual_state[:] = contacts

    def pre_physics_step(self, actions):
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self.reset_idx(reset_ids)
        self.actions.copy_(torch.clamp(actions.to(self.device), -1, 1))
        desired = self.prev_targets + self.action_scale * self.actions
        desired = torch.maximum(torch.minimum(desired, self.dof_upper_limits), self.dof_lower_limits)
        self.cur_targets.copy_(self.target_smoothing * desired + (1 - self.target_smoothing) * self.prev_targets)
        self.prev_targets.copy_(self.cur_targets)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        self.tactile_forces.zero_()
        self.tactile_force_norm.zero_()
        self.tactile_contacts.zero_()
        self._physics_substep = 0

    def update_controller(self):
        # Called before each simulate().  The first sample belongs to the previous
        # control step (or episode), so collect only after a new physics step.
        if self._physics_substep > 0:
            self._collect_tactile()
        self._physics_substep += 1

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.compute_observations()
        self.compute_reward()

    def step(self, actions):
        observations, rewards, dones, infos = super().step(actions)
        if self.hand_camera is not None:
            # Capture completed physics, including terminal frames, before reset.
            if self.camera_config.get("display", False) and not self.force_render:
                self.render()
            self.hand_camera.capture()
        if self.rotation_stall_enabled:
            # A stalled episode is a task termination, even near the time limit.
            infos["time_outs"] = infos["time_outs"] & ~self.rotation_stalled.to(self.rl_device)
        # PPO samples its next action directly from the returned observation;
        # it does not call reset_done() between steps. Return a fresh initial
        # observation for done environments while preserving terminal results.
        done_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            rewards, dones = rewards.clone(), dones.clone()
            infos = dict(infos)
            infos["time_outs"] = infos["time_outs"].clone()
            self.reset_idx(done_ids)
            self.compute_observations(update_history=False, capture_contacts=False, refresh_state=False)
            observations = self._pack_observations()
        return observations, rewards, dones, infos

    def _normalise_q(self, q):
        return 2 * (q - self.dof_lower_limits) / (self.dof_upper_limits - self.dof_lower_limits) - 1

    def compute_observations(self, update_history=True, capture_contacts=True, refresh_state=True):
        if refresh_state:
            self._refresh_state()
        if capture_contacts:
            # Include the last physics step as well as the intermediate steps.
            self._collect_tactile()
        self.object_state = self.root_state_tensor[self.object_indices]
        self.object_pos = self.object_state[..., :3]
        self.object_rot = self.object_state[..., 3:7]
        self.object_linvel = self.object_state[..., 7:10]
        self.object_angvel = self.object_state[..., 10:13]
        local_states = self._object_states_in_hand_frame(self.object_state)
        self.object_pos_local = local_states[..., :3]
        local_object = local_states.reshape(self.num_envs, self.object_state_dim)
        q, targets = self._normalise_q(self.dof_pos), self._normalise_q(self.cur_targets)
        frame = torch.cat((q, targets, self.tactile_contacts, self.axis_local), dim=-1)
        if update_history:
            self.obs_history[:, 1:] = self.obs_history[:, :-1].clone()
            self.obs_history[:, 0] = frame
        new_ids = self._history_needs_reset.nonzero(as_tuple=False).flatten()
        if new_ids.numel() > 0:
            self.obs_history[new_ids] = frame[new_ids].unsqueeze(1).expand(-1, self.frame_stack, -1)
            self._history_needs_reset[new_ids] = False
        self.obs_buf.copy_(torch.nan_to_num(torch.cat((self.obs_history.flatten(1), local_object), dim=-1)))
        # Forces are peak-per-pad vectors in the hand frame, with pad order fixed
        # by tactile_link_names.  The actor uses binary contact only.
        raw_tactile_local = quat_apply(
            self.hand_base_inverse_quat[:, None, :].expand(-1, self.num_tactile_sensors, -1).reshape(-1, 4),
            self.tactile_forces.reshape(-1, 3)).reshape(self.num_envs, -1)
        state = torch.cat((q, self.dof_vel, targets, self.actions, local_object,
                           self.axis_local, raw_tactile_local), dim=-1)
        self.states_buf.copy_(torch.nan_to_num(state))
        # Display the same completed control-step contact flags as the actor.
        # Do not update on pre_physics_step's temporary accumulator clear.
        self._update_tactile_colors()
        return self.obs_buf

    def compute_reward(self):
        if self.num_objects == 2:
            return self._compute_baoding_reward()
        # Episode dictionaries must describe only newly finished episodes.
        self.extras.pop("episode", None)
        env = self.cfg["env"]
        sampled_spin = torch.sum(self.object_angvel * self.spin_axis, dim=-1)
        pose_angle = None
        spin = sampled_spin
        if self.pose_rotation_reward:
            pose_angle = self.projected_rotation_delta(
                self.previous_object_quat, self.object_rot, self.spin_axis)
            # Update once, sharing this exact increment with reward and stall checks.
            self.previous_object_quat.copy_(self.object_rot)
            spin = pose_angle / self.rotation_control_dt
        spin_clipped = torch.clamp(spin, -float(env.get("maxSpinReward", 10)), float(env.get("maxSpinReward", 10)))
        position_error = torch.linalg.vector_norm(self.object_pos_local - self.object_reference_local, dim=-1)
        fingertip_positions = self.rigid_body_state_tensor[self.fingertip_body_indices, :3]
        fingertip_distance = torch.linalg.vector_norm(fingertip_positions - self.object_pos[:, None, :], dim=-1).mean(-1)
        contact_fraction = self.tactile_contacts.mean(-1)
        spin_reward = float(env.get("spinRewardScale", 1)) * spin_clipped
        position_penalty = float(env.get("positionPenaltyScale", 2)) * position_error
        if self.centered_rotation_reward:
            excess_distance = (position_error - self.position_tolerance).clamp_min(0)
            position_gate = torch.exp(-0.5 * (excess_distance / self.position_reward_falloff).square())
            # Only forward progress saturates and fades away from the palm target.
            # Reverse progress stays linear, so large backtracking cannot be hidden
            # by the positive speed limit or reduced by moving away from the palm.
            spin_reward = float(env.get("spinRewardScale", 1)) * (
                position_gate * spin.clamp(min=0, max=self.positive_spin_reward_limit)
                + spin.clamp(max=0))
            position_penalty = float(env.get("positionPenaltyScale", 2)) * (
                excess_distance / self.position_penalty_distance).square()
        reward = (spin_reward
                  + float(env.get("contactRewardScale", 0.2)) * contact_fraction
                  - position_penalty
                  - float(env.get("fingerDistancePenaltyScale", 0.1)) * fingertip_distance
                  - float(env.get("actionPenaltyScale", 0.002)) * self.actions.square().mean(-1)
                  - float(env.get("velocityPenaltyScale", 0.0001)) * self.dof_vel.square().mean(-1))
        local_height = (self.object_pos_local * self.palm_normal_local).sum(-1)
        finite = (torch.isfinite(self.object_state).all(-1) & torch.isfinite(self.dof_state).all(-1).all(-1)
                  & torch.isfinite(self.tactile_forces).all(-1).all(-1)
                  & torch.isfinite(self.tactile_force_norm).all(-1)
                  & torch.isfinite(fingertip_positions).all(-1).all(-1))
        dropped = ((local_height < float(env.get("dropHeightLocal", -0.02)))
                   | (position_error > float(env.get("maxObjectDistance", 0.18))))
        if self.pose_rotation_reward:
            # A terminal fall must not earn a positive rotation impulse.
            held_spin_reward = torch.where(finite & ~dropped, spin_reward, torch.zeros_like(spin_reward))
            reward = reward - spin_reward + held_spin_reward
            spin_reward = held_spin_reward
        reward = reward + float(env.get("fallPenalty", -15.0)) * dropped.float()
        self.rew_buf.copy_(torch.where(finite, torch.nan_to_num(reward), torch.zeros_like(reward)))
        self.reset_buf.copy_((dropped | ~finite | (self.progress_buf >= self.max_episode_length)).long())
        if self.rotation_stall_enabled:
            self._update_rotation_stall(pose_angle)
            self.rotation_stalled.logical_and_(finite & ~dropped)
            self.reset_buf.logical_or_(self.rotation_stalled)
            self.extras["rotation/stall_rate"] = self.rotation_stalled.float().mean()
        self.extras["tactile/contact_fraction"] = contact_fraction.mean()
        self.extras["tactile/contact_env_rate"] = (self.tactile_contacts.sum(-1) > 0).float().mean()
        self.extras["tactile/peak_force"] = torch.nan_to_num(self.tactile_force_norm).mean()
        self.extras["rotation/spin"] = torch.nan_to_num(spin).mean()
        self.extras["rotation/position_error"] = torch.nan_to_num(position_error).mean()
        self.extras["rotation/drop_rate"] = dropped.float().mean()
        self.extras["rotation/nonfinite_rate"] = (~finite).float().mean()
        if self.centered_rotation_reward:
            self.extras["rotation/position_reward_gate"] = torch.nan_to_num(position_gate).mean()
            self.extras["rotation/position_penalty"] = torch.nan_to_num(position_penalty).mean()
        if self.pose_rotation_reward:
            self.extras["rotation/sampled_spin"] = torch.nan_to_num(sampled_spin).mean()
            self._log_rotation_episodes(pose_angle, sampled_spin, spin_reward, finite, dropped)

    @staticmethod
    def projected_rotation_delta(previous, current, axis):
        """Shortest signed quaternion increment projected onto a fixed world axis."""
        delta = quat_mul(current, quat_conjugate(previous))
        delta = delta / torch.clamp(torch.linalg.vector_norm(delta, dim=-1, keepdim=True), min=1e-8)
        # q and -q describe the same pose; use the shortest incremental turn.
        delta = torch.where(delta[:, 3:4] < 0, -delta, delta)
        vector_norm = torch.linalg.vector_norm(delta[:, :3], dim=-1)
        angle = 2 * torch.atan2(vector_norm, delta[:, 3])
        projected = angle * (delta[:, :3] * axis).sum(-1) / vector_norm.clamp_min(1e-8)
        return torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)

    def _update_rotation_stall(self, projected_angle=None):
        """Detect negligible net rotation over a full window, after reset grace.

        Use actual orientation increments rather than sampled angular velocity.
        Signed increments cancel back-and-forth jitter. Legacy presets accept
        either direction; forward-only presets require positive net progress.
        """
        if projected_angle is None:
            projected_angle = self.projected_rotation_delta(
                self.previous_object_quat, self.object_rot, self.spin_axis)
            self.previous_object_quat.copy_(self.object_rot)
        self.rotation_step_angles[self.rotation_window_index].copy_(projected_angle)
        self.rotation_window_index = (self.rotation_window_index + 1) % self.rotation_stall_steps
        net_angle = self.rotation_step_angles.sum(0)
        ready = self.progress_buf >= self.rotation_stall_grace_steps + self.rotation_stall_steps
        progress = net_angle if self.rotation_stall_forward_only else net_angle.abs()
        self.rotation_stalled.copy_(ready & (progress < self.rotation_stall_min_angle))
        self.extras["rotation/window_angle_deg"] = net_angle.abs().mean() * (180.0 / math.pi)
        if self.rotation_stall_forward_only:
            self.extras["rotation/window_signed_angle_deg"] = net_angle.mean() * (180.0 / math.pi)

    def _log_rotation_episodes(self, angle, sampled_spin, spin_reward, finite, dropped):
        """Log whole completed episodes, independently of last-step direct metrics.

        Angles are unclipped projected pose increments, not certified full
        fixed-axis revolutions. Invalid/fallen terminal motion is excluded.
        """
        valid = finite & ~dropped
        sums = self.rotation_episode_sums
        sums["angle"].add_(torch.where(valid, angle, torch.zeros_like(angle)))
        sums["sampled_spin"].add_(torch.where(valid, sampled_spin, torch.zeros_like(sampled_spin)))
        sums["spin_reward"].add_(spin_reward)
        sums["reward"].add_(self.rew_buf)
        sums["reverse_steps"].add_((valid & (angle < -1e-6)).float())
        done_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return
        steps = self.progress_buf[done_ids].clamp_min(1).float()
        duration = steps * self.rotation_control_dt
        stalled = (self.rotation_stalled[done_ids] if self.rotation_stall_enabled
                   else torch.zeros_like(done_ids, dtype=torch.bool))
        fell = finite[done_ids] & dropped[done_ids]
        invalid = ~finite[done_ids]
        timed_out = ~invalid & ~fell & ~stalled
        # Advanced indexing creates snapshots that survive reset_idx in step().
        self.extras["episode"] = {
            "rotation/net_angle_deg": sums["angle"][done_ids] * (180.0 / math.pi),
            "rotation/mean_actual_spin": sums["angle"][done_ids] / duration,
            "rotation/mean_sampled_spin": sums["sampled_spin"][done_ids] / steps,
            "rotation/reverse_step_fraction": sums["reverse_steps"][done_ids] / steps,
            "rotation/spin_return": sums["spin_reward"][done_ids],
            "rotation/return": sums["reward"][done_ids],
            "rotation/duration_seconds": duration,
            "rotation/stall_termination_rate": stalled.float(),
            "rotation/drop_termination_rate": fell.float(),
            "rotation/nonfinite_termination_rate": invalid.float(),
            "rotation/timeout_termination_rate": timed_out.float(),
        }

    @staticmethod
    def orbit_delta(previous, current, axis):
        """Signed angle between projected centre-to-centre vectors, in radians.

        A positive increment follows the right-hand rule about ``axis``.  Using
        centre differences removes common translation; sphere orientations and
        individual angular velocities do not enter this calculation.  A
        degenerate projection or a nonfinite input produces a zero increment.
        """
        epsilon = 1e-8
        axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        unit_axis = axis / torch.clamp(axis_norm, min=epsilon)
        old = previous - (previous * unit_axis).sum(-1, keepdim=True) * unit_axis
        new = current - (current * unit_axis).sum(-1, keepdim=True) * unit_axis
        sine = (torch.cross(old, new, dim=-1) * unit_axis).sum(-1)
        cosine = (old * new).sum(-1)
        angle = torch.atan2(sine, cosine)
        valid = ((axis_norm.squeeze(-1) > epsilon)
                 & (torch.linalg.vector_norm(old, dim=-1) > epsilon)
                 & (torch.linalg.vector_norm(new, dim=-1) > epsilon)
                 & torch.isfinite(previous).all(-1) & torch.isfinite(current).all(-1)
                 & torch.isfinite(axis).all(-1))
        return torch.where(valid, angle, torch.zeros_like(angle))

    def _compute_baoding_reward(self):
        env = self.cfg["env"]
        ball_vector = self.object_pos[:, 1] - self.object_pos[:, 0]
        angle = self.orbit_delta(self.previous_ball_vector, ball_vector, self.spin_axis)
        # Use the actual decision interval, rather than the old task's fixed
        # theta*20 multiplier.  Preserve the sign so backtracking is penalised.
        control_dt = max(float(self.dt) * self.control_freq_inv, 1e-8)
        orbital_speed = angle / control_dt
        self.previous_ball_vector.copy_(ball_vector)

        centroid_local = self.object_pos_local.mean(1)
        initial_centroid = self.object_reference_local.mean(1)
        centroid_error = torch.linalg.vector_norm(centroid_local - initial_centroid, dim=-1)
        centroid_velocity = self.object_linvel.mean(1)
        centroid_speed = torch.linalg.vector_norm(centroid_velocity, dim=-1)
        separation = torch.linalg.vector_norm(ball_vector, dim=-1)
        projected_vector = ball_vector - (ball_vector * self.spin_axis).sum(-1, keepdim=True) * self.spin_axis
        projected_separation = torch.linalg.vector_norm(projected_vector, dim=-1)
        local_height = (self.object_pos_local * self.palm_normal_local).sum(-1)
        # Distance to the fixed initial pair centre permits the balls to exchange
        # places; comparing each ball to its own start position would oppose it.
        ball_distance = torch.linalg.vector_norm(self.object_pos_local - initial_centroid[:, None, :], dim=-1)
        radius = float(env.get("ballRadius", 0.022)) * float(env.get("objectScale", 1.0))
        minimum_separation = float(env.get("minBallSeparation", 1.8 * radius))
        maximum_separation = float(env.get("maxBallSeparation", 0.10))
        maximum_distance = float(env.get("maxObjectDistance", 0.18))
        finite = (torch.isfinite(self.object_state).all(-1).all(-1)
                  & torch.isfinite(self.dof_state).all(-1).all(-1)
                  & torch.isfinite(self.tactile_forces).all(-1).all(-1)
                  & torch.isfinite(self.tactile_force_norm).all(-1))
        fingertip_positions = self.rigid_body_state_tensor[self.fingertip_body_indices, :3]
        finite = finite & torch.isfinite(fingertip_positions).all(-1).all(-1)
        dropped = ((local_height < float(env.get("dropHeightLocal", -0.02))).any(-1)
                   | (ball_distance > maximum_distance).any(-1)
                   | (separation < minimum_separation) | (separation > maximum_separation))
        held = (finite & ~dropped
                & (local_height >= float(env.get("minBallHeightLocal", 0.0))).all(-1)
                & (projected_separation >= 0.5 * minimum_separation))
        # Both signed directions remain in the reward.  No abs(), positive-only
        # clipping, ball quaternion or individual angular-speed reward is used.
        rewarded_speed = torch.where(held, orbital_speed, torch.zeros_like(orbital_speed))
        self.orbit_angle_accumulator.add_(torch.where(held, angle, torch.zeros_like(angle)))
        contact_fraction = self.tactile_contacts.mean(-1)
        tip_to_ball = torch.linalg.vector_norm(
            fingertip_positions[:, :, None, :] - self.object_pos[:, None, :, :], dim=-1)
        fingertip_distance = tip_to_ball.min(-1).values.mean(-1)
        tilt = (ball_vector * self.spin_axis).sum(-1).abs() / torch.clamp(separation, min=1e-8)
        reward = (float(env.get("spinRewardScale", 1.0)) * rewarded_speed
                  + float(env.get("contactRewardScale", 0.2)) * contact_fraction
                  - float(env.get("positionPenaltyScale", 2.0)) * centroid_error
                  - float(env.get("centroidVelocityPenaltyScale", 0.1)) * centroid_speed
                  - float(env.get("orbitTiltPenaltyScale", 0.1)) * tilt
                  - float(env.get("fingerDistancePenaltyScale", 0.1)) * fingertip_distance
                  - float(env.get("actionPenaltyScale", 0.002)) * self.actions.square().mean(-1)
                  - float(env.get("velocityPenaltyScale", 0.0001)) * self.dof_vel.square().mean(-1)
                  + float(env.get("fallPenalty", -15.0)) * dropped.float())
        self.rew_buf.copy_(torch.where(finite, torch.nan_to_num(reward), torch.zeros_like(reward)))
        self.reset_buf.copy_((dropped | ~finite | (self.progress_buf >= self.max_episode_length)).long())
        self.extras["tactile/contact_fraction"] = contact_fraction.mean()
        self.extras["tactile/contact_env_rate"] = (self.tactile_contacts.sum(-1) > 0).float().mean()
        self.extras["tactile/peak_force"] = torch.nan_to_num(self.tactile_force_norm).mean()
        self.extras["rotation/spin"] = torch.nan_to_num(rewarded_speed).mean()
        self.extras["rotation/position_error"] = torch.nan_to_num(centroid_error).mean()
        self.extras["rotation/drop_rate"] = dropped.float().mean()
        self.extras["rotation/nonfinite_rate"] = (~finite).float().mean()
        self.extras["baoding/orbit_delta"] = torch.nan_to_num(angle).mean()
        self.extras["baoding/orbit_turns"] = torch.nan_to_num(self.orbit_angle_accumulator).mean() / (2 * math.pi)
        self.extras["baoding/separation"] = torch.nan_to_num(separation).mean()
        self.extras["baoding/centroid_speed"] = torch.nan_to_num(centroid_speed).mean()
        self.extras["baoding/held_rate"] = held.float().mean()

    def reset_idx(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        cfg = self.cfg["env"]
        count = env_ids.numel()
        noise = float(cfg.get("resetDofPositionNoise", 0.03))
        q = self.default_dof_pos + noise * (2 * torch.rand((count, self.num_actions), device=self.device) - 1)
        q = torch.maximum(torch.minimum(q, self.dof_upper_limits), self.dof_lower_limits)
        self.dof_pos[env_ids] = q
        self.dof_vel[env_ids] = 0
        self.prev_targets[env_ids] = q
        self.cur_targets[env_ids] = q
        self.actions[env_ids] = 0
        object_states = self.object_init_state[env_ids].clone()
        # Every episode starts at rest, regardless of the imported GPU snapshot.
        object_states[..., 7:13] = 0
        position_noise = float(cfg.get("objectPositionNoise", 0.002))
        offset_local = position_noise * (2 * torch.rand((count, 3), device=self.device) - 1)
        offset_world = quat_apply(self.hand_base_quat[env_ids], offset_local)
        if self.num_objects > 1:
            # A shared translation preserves the separation between the balls.
            offset_world = offset_world[:, None, :]
        object_states[..., :3] += offset_world
        self.root_state_tensor[self.object_indices[env_ids]] = object_states
        if self.num_objects == 2:
            self.previous_ball_vector[env_ids] = object_states[:, 1, :3] - object_states[:, 0, :3]
            self.orbit_angle_accumulator[env_ids] = 0
        if self.rotation_stall_enabled or self.pose_rotation_reward:
            self.previous_object_quat[env_ids] = object_states[:, 3:7]
        if self.pose_rotation_reward:
            for total in self.rotation_episode_sums.values():
                total[env_ids] = 0
        if self.rotation_stall_enabled:
            self.rotation_step_angles[:, env_ids] = 0
            self.rotation_stalled[env_ids] = False
        object_ids = self.object_indices[env_ids].to(dtype=torch.int32).flatten()
        hand_ids = self.hand_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                    gymtorch.unwrap_tensor(object_ids), object_ids.numel())
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor),
                                             gymtorch.unwrap_tensor(hand_ids), hand_ids.numel())
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.cur_targets),
                                                       gymtorch.unwrap_tensor(hand_ids), hand_ids.numel())
        self.tactile_forces[env_ids] = 0
        self.tactile_force_norm[env_ids] = 0
        self.tactile_contacts[env_ids] = 0
        self.obs_history[env_ids] = 0
        self._history_needs_reset[env_ids] = True
        self.progress_buf[env_ids] = 0
        self.randomize_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.timeout_buf[env_ids] = 0

    def _pack_observations(self):
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        self.obs_dict["states"] = self.get_state()
        return self.obs_dict

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # GPU setters do not make a refresh valid until a simulation step.  Use
        # the root/DOF cache just written by reset_idx, without advancing others.
        self.compute_observations(update_history=False, capture_contacts=False, refresh_state=False)
        return self._pack_observations()

    def reset_done(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self.reset_idx(env_ids)
            self.compute_observations(update_history=False, capture_contacts=False, refresh_state=False)
        return self._pack_observations(), env_ids

    def get_number_of_agents(self):
        return 1

    def get_internal_info(self, key):
        if key == "obj":
            return self.object_class_indices
        if key == "tactile":
            return self.tactile_forces
        raise KeyError(key)

    def get_internal_state(self):
        # Match the original teacher's object-orientation inspection interface.
        return self.object_rot.clone()

    def close(self):
        monitor = getattr(self, "hand_camera", None)
        try:
            if monitor is not None:
                monitor.close()
        finally:
            self.hand_camera = None
            try:
                if getattr(self, "viewer", None) is not None:
                    self.gym.destroy_viewer(self.viewer)
                    self.viewer = None
            finally:
                try:
                    if getattr(self, "sim", None) is not None:
                        self.gym.destroy_sim(self.sim)
                        self.sim = None
                finally:
                    if getattr(self, "virtual_display", None) is not None:
                        self.virtual_display.stop()
                        self.virtual_display = None
