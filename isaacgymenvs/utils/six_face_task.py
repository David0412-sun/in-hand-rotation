"""Stateful six-main-face presentation reward, independent of Isaac Gym.

Poses are [position, quaternion_xyzw] in the same coordinate frame. Camera
quaternions map optical coordinates (+X right, +Y down, +Z forward) to that
frame. Visibility is a geometric proxy; it does not test hand occlusion.
"""

import math

import torch


FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def _rotate(quaternion, vectors):
    xyz, w = quaternion[..., :3], quaternion[..., 3:4]
    first = torch.cross(xyz.expand_as(vectors), vectors, dim=-1)
    return vectors + 2.0 * (w * first + torch.cross(xyz.expand_as(vectors), first, dim=-1))


def _safe_pose(pose):
    pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
    quaternion = pose[..., 3:7]
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    return pose[..., :3], quaternion


class SixFaceTask:
    """Present six distinct faces, optionally repeating without a physical reset."""

    OBSERVATION_DIM = 16
    observation_size = OBSERVATION_DIM
    face_names = FACE_NAMES

    def __init__(self, config, num_envs, device, dt):
        self.num_envs, self.device = int(num_envs), torch.device(device)
        if self.num_envs <= 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("SixFaceTask requires positive num_envs and finite dt")
        self.dt = float(dt)

        def number(name, default, positive=True):
            value = float(config.get(name, default))
            if not math.isfinite(value) or (positive and value <= 0):
                raise ValueError("sixFace.{} must be finite{}".format(name, " and positive" if positive else ""))
            return value

        self.alignment_tolerance = math.radians(number("alignment_tolerance_deg", 15.0))
        hold_seconds = number("hold_seconds", 0.5, False)
        if hold_seconds < 0:
            raise ValueError("sixFace.hold_seconds must be nonnegative")
        self.hold_required = math.ceil(hold_seconds / self.dt)
        self.repeat_cycles = bool(config.get("repeat_cycles", False))
        self.max_angular_speed = number("max_angular_speed", 0.4)
        self.max_linear_speed = number("max_linear_speed", 0.03)
        self.position_tolerance = number("position_tolerance", 0.015)
        self.progress_scale = number("progress_scale", 2.0, False)
        self.hold_scale = number("hold_reward", 0.05, False)
        self.face_scale = number("face_bonus", 5.0, False)
        self.initial_face_scale = number("initial_face_bonus", 0.0, False)
        self.completion_scale = number("completion_bonus", 10.0, False)
        self.timeout_scale = number("timeout_penalty", -2.0, False)
        self.stall_steps = max(1, math.ceil(number("stall_seconds", 3.0) / self.dt))
        grace_seconds = number("stall_grace_seconds", 2.0, False)
        if grace_seconds < 0:
            raise ValueError("sixFace.stall_grace_seconds must be nonnegative")
        self.grace_steps = math.ceil(grace_seconds / self.dt)
        self.stall_progress = math.radians(number("stall_progress_deg", 5.0))
        self.timeout_steps = max(1, math.ceil(number("target_timeout_seconds", 12.0) / self.dt))
        self.minimum_area = number("min_projected_area_pixels", 64.0)
        self.half_extent = number("object_half_extent", 0.0425)
        face_ratio = number("face_half_width_ratio", 0.58)
        self.width = number("camera_width", 256)
        self.height = number("camera_height", 256)
        fov = number("camera_horizontal_fov", 60.0)
        self.near = number("camera_near", 0.01)
        self.far = number("camera_far", 2.0)
        if not 0 < fov < 180 or self.far <= self.near or not 0 < self.alignment_tolerance < math.pi / 2:
            raise ValueError("Invalid sixFace camera frustum or alignment tolerance")
        if not 0 < face_ratio <= 1 or self.timeout_scale > 0:
            raise ValueError("Invalid sixFace face ratio or timeout penalty")
        if min(self.progress_scale, self.hold_scale, self.face_scale, self.initial_face_scale, self.completion_scale) < 0:
            raise ValueError("sixFace reward coefficients must be nonnegative")
        self.tan_x = math.tan(math.radians(fov) / 2)
        self.tan_y = self.tan_x * self.height / self.width
        self.focal = self.width / (2 * self.tan_x)

        normals = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        self.normals = torch.tensor(normals, device=self.device, dtype=torch.float32)
        self.box_corners = self.half_extent * torch.tensor(
            [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
            device=self.device, dtype=torch.float32)
        corners = []
        for normal in normals:
            axis = next(i for i in range(3) if normal[i])
            tangents = [i for i in range(3) if i != axis]
            face = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                point = [float(x) for x in normal]
                point[tangents[0]], point[tangents[1]] = a * face_ratio, b * face_ratio
                face.append(point)
            corners.append(face)
        self.face_corners = self.half_extent * torch.tensor(corners, device=self.device, dtype=torch.float32)
        self.goal_face = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.cycles_completed = torch.zeros_like(self.goal_face)
        self.covered = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.bool)
        self.hold_steps = torch.zeros_like(self.goal_face)
        self.max_paid_hold_steps = torch.zeros_like(self.goal_face)
        self.goal_age = torch.zeros_like(self.goal_face)
        self.no_progress_steps = torch.zeros_like(self.goal_face)
        self.previous_error = torch.zeros(self.num_envs, device=self.device)
        self.best_error = torch.zeros_like(self.previous_error)
        self.stall_reference_error = torch.zeros_like(self.previous_error)

    def _geometry(self, object_pose, camera_pose):
        position, quaternion = _safe_pose(object_pose)
        camera_position, camera_quaternion = _safe_pose(camera_pose)
        inverse_camera = torch.cat((-camera_quaternion[:, :3], camera_quaternion[:, 3:4]), dim=-1)
        normals = _rotate(quaternion[:, None, :], self.normals[None, :, :].expand(self.num_envs, -1, -1))
        centers = position[:, None, :] + self.half_extent * normals
        rays = camera_position[:, None, :] - centers
        rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True).clamp_min(1e-8)
        errors = torch.acos(torch.sum(normals * rays, dim=-1).clamp(-1, 1))

        box = _rotate(quaternion[:, None, :], self.box_corners[None, :, :].expand(self.num_envs, -1, -1))
        box = _rotate(inverse_camera[:, None, :], box + position[:, None, :] - camera_position[:, None, :])
        x, y, z = box.unbind(dim=-1)
        envelope_visible = ((z >= self.near) & (z <= self.far)
                            & (x.abs() <= z * self.tan_x) & (y.abs() <= z * self.tan_y)).all(dim=-1)

        face = self.face_corners[None, :, :, :].expand(self.num_envs, -1, -1, -1)
        face = _rotate(quaternion[:, None, None, :], face) + position[:, None, None, :]
        face = _rotate(inverse_camera[:, None, None, :], face - camera_position[:, None, None, :])
        pixels = self.focal * face[..., :2] / face[..., 2:3].clamp_min(self.near)
        px, py = pixels.unbind(dim=-1)
        area = 0.5 * (px * torch.roll(py, -1, dims=-1) - py * torch.roll(px, -1, dims=-1)).sum(dim=-1).abs()
        visible = envelope_visible[:, None] & (area >= self.minimum_area) & (errors <= self.alignment_tolerance)
        return errors, visible

    def reset(self, env_ids, object_pose, camera_pose):
        """Reset selected envs; both pose arrays contain all environments."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        errors, _ = self._geometry(object_pose, camera_pose)
        goals = errors[env_ids].argmin(dim=-1)
        initial_error = errors[env_ids].gather(1, goals[:, None]).squeeze(1)
        self.goal_face[env_ids] = goals
        self.covered[env_ids] = False
        self.cycles_completed[env_ids] = 0
        for buffer in (self.hold_steps, self.max_paid_hold_steps, self.goal_age, self.no_progress_steps):
            buffer[env_ids] = 0
        for buffer in (self.previous_error, self.best_error, self.stall_reference_error):
            buffer[env_ids] = initial_error

    def update(self, object_pose, linear_velocity, angular_velocity, camera_pose, position_error, valid):
        """Advance once per control step; returned labels describe the old target.

        State buffers and observation() describe the next target after success.
        Invalid/dropped envs earn no positive reward. The caller applies ordinary
        penalties and resets on stalled, goal_timeout, terminal_success, or its
        own done. completed is a one-step cycle event, not necessarily terminal.
        """
        errors, visible = self._geometry(object_pose, camera_pose)
        old_goal = self.goal_face.clone()
        index = old_goal[:, None]
        error = errors.gather(1, index).squeeze(1)
        geometrically_visible = visible.gather(1, index).squeeze(1)
        finite = (torch.isfinite(object_pose).all(dim=-1) & torch.isfinite(camera_pose).all(dim=-1)
                  & torch.isfinite(linear_velocity).all(dim=-1) & torch.isfinite(angular_velocity).all(dim=-1)
                  & torch.isfinite(position_error)
                  & (torch.linalg.vector_norm(object_pose[:, 3:7], dim=-1) > 1e-8)
                  & (torch.linalg.vector_norm(camera_pose[:, 3:7], dim=-1) > 1e-8))
        eligible = valid.bool() & finite & ~self.covered.gather(1, index).squeeze(1)
        stable = (eligible & geometrically_visible & (position_error <= self.position_tolerance)
                  & (torch.linalg.vector_norm(linear_velocity, dim=-1) <= self.max_linear_speed)
                  & (torch.linalg.vector_norm(angular_velocity, dim=-1) <= self.max_angular_speed))
        progress_reward = torch.where(eligible, self.progress_scale * (self.previous_error - error), 0.0)
        self.previous_error.copy_(error)
        self.goal_age.add_(eligible.long())
        if self.hold_required:
            self.hold_steps.copy_(torch.where(stable, self.hold_steps + 1, 0).clamp_max(self.hold_required))
            newly_paid = (self.hold_steps - self.max_paid_hold_steps).clamp_min(0)
            hold_reward = self.hold_scale * newly_paid.float()
            self.max_paid_hold_steps.copy_(torch.maximum(self.max_paid_hold_steps, self.hold_steps))
            achieved = stable & (self.hold_steps >= self.hold_required)
        else:
            # A sampled, visible face is enough. Do not require stopping, a
            # dwell period, low velocity, or the old 15 mm presentation gate.
            self.hold_steps.zero_()
            self.max_paid_hold_steps.zero_()
            hold_reward = torch.zeros_like(error)
            achieved = eligible & geometrically_visible

        self.best_error.copy_(torch.where(eligible, torch.minimum(self.best_error, error), self.best_error))
        improved = eligible & (self.stall_reference_error - self.best_error >= self.stall_progress)
        self.stall_reference_error.copy_(torch.where(improved, self.best_error, self.stall_reference_error))
        self.no_progress_steps.copy_(torch.where(improved | stable, 0, self.no_progress_steps + eligible.long()))
        first_face = ~self.covered.any(dim=-1) & (self.cycles_completed == 0)
        face_bonus = achieved.float() * torch.where(first_face, self.initial_face_scale, self.face_scale)
        self.covered.scatter_(1, index, self.covered.gather(1, index) | achieved[:, None])
        completed = achieved & self.covered.all(dim=-1)
        covered_count = self.covered.sum(dim=-1)
        self.cycles_completed.add_(completed.long())
        terminal_success = completed & (not self.repeat_cycles)
        completion_bonus = self.completion_scale * completed.float()
        stalled = (eligible & ~achieved & ~stable & (self.no_progress_steps >= self.stall_steps)
                   & (self.goal_age >= self.grace_steps + self.stall_steps))
        goal_timeout = eligible & ~achieved & (self.goal_age >= self.timeout_steps)
        timeout_penalty = self.timeout_scale * (stalled | goal_timeout).float()

        # Switch after accounting for this step. New-goal angular error is never
        # compared to old-goal error, preventing artificial switch rewards.
        restart_cycle = completed & self.repeat_cycles
        self.covered.masked_fill_(restart_cycle[:, None], False)
        goal_changed = achieved & ~terminal_success
        unavailable = self.covered.clone()
        # The last face of a completed cycle cannot immediately count again as
        # the next cycle's first face. All six must still be visited next time.
        unavailable.scatter_(1, index, unavailable.gather(1, index) | restart_cycle[:, None])
        next_goals = errors.masked_fill(unavailable, float("inf")).argmin(dim=-1)
        self.goal_face.copy_(torch.where(goal_changed, next_goals, old_goal))
        next_error = errors.gather(1, self.goal_face[:, None]).squeeze(1)
        for buffer in (self.previous_error, self.best_error, self.stall_reference_error):
            buffer.copy_(torch.where(goal_changed, next_error, buffer))
        for buffer in (self.hold_steps, self.max_paid_hold_steps, self.goal_age, self.no_progress_steps):
            buffer.masked_fill_(goal_changed, 0)

        reward = progress_reward + hold_reward + face_bonus + completion_bonus + timeout_penalty
        return {
            "reward": reward, "progress_reward": progress_reward, "hold_reward": hold_reward,
            "face_bonus": face_bonus, "completion_bonus": completion_bonus,
            "timeout_penalty": timeout_penalty, "stalled": stalled, "goal_timeout": goal_timeout,
            "completed": completed, "terminal_success": terminal_success,
            "cycles_completed": self.cycles_completed.clone(), "covered_count": covered_count,
            "goal_changed": goal_changed, "stable": stable, "presented": achieved,
            "geometrically_visible": geometrically_visible, "target_error": error,
            "target_face_before": old_goal,
            "metrics": {
                "six_face/covered": covered_count.float().mean(),
                "six_face/cycles_completed": self.cycles_completed.float().mean(),
                "six_face/presented": achieved.float().mean(),
                "six_face/target_error_deg": torch.rad2deg(error).mean(),
                "six_face/stable": stable.float().mean(),
                "six_face/geometrically_visible": geometrically_visible.float().mean(),
                "six_face/completed": completed.float().mean(),
                "six_face/stalled": stalled.float().mean(),
                "six_face/goal_timeout": goal_timeout.float().mean(),
            },
        }

    def observation(self):
        goal = torch.nn.functional.one_hot(self.goal_face, num_classes=6).float()
        # Preserve the 16-D layout. The unused dwell slot identifies the sole
        # unbonused initial face in immediate-presentation mode; later cycles
        # have the same empty coverage mask but pay their first new face.
        phase = (self.hold_steps.float() / self.hold_required if self.hold_required else
                 (~self.covered.any(dim=-1) & (self.cycles_completed == 0)).float())
        auxiliary = torch.stack((phase,
                                 self.best_error / math.pi,
                                 self.no_progress_steps.float() / self.stall_steps,
                                 self.goal_age.float() / self.timeout_steps), dim=-1).clamp(0, 1)
        return torch.cat((goal, self.covered.float(), auxiliary), dim=-1)
