"""Optional finger-camera preview and bounded, streaming video recording.

Images are diagnostic outputs only: this module never changes observations,
actions, or physics state. All Gym and Tk calls stay on the simulation thread.
"""

from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import warnings

import numpy as np
from isaacgym import gymapi


FINGERS = ("thumb", "index", "middle", "ring", "little")


def resolve_env_ids(config, num_envs):
    """Resolve the small camera pool before creating actors or cameras."""
    if not config.get("enabled", False):
        return ()
    selected = int(config.get("env_id", 0))
    if not 0 <= selected < num_envs:
        raise ValueError("camera.env_id must refer to an existing environment")
    if config.get("display", False):
        requested = config.get("env_ids", [])
        ids = tuple(int(value) for value in requested) if requested else tuple(range(min(4, num_envs)))
        if selected not in ids:
            if requested:
                raise ValueError("camera.env_id must be included in camera.env_ids")
            ids = ids[:-1] + (selected,)
    else:
        ids = (selected,)
    if len(set(ids)) != len(ids) or any(value < 0 or value >= num_envs for value in ids):
        raise ValueError("camera.env_ids must contain unique, valid environment IDs")
    return ids


def mouse_ray(mouse_xy, viewer_pose, horizontal_fov, viewer_size):
    """Ray in env coordinates. Preview4 Viewer axes are left/up/forward.

    Use the public Viewer pose API: its camera handle is not a sensor handle,
    and get_viewer_camera_handle itself crashes in some Preview4 builds.
    """
    width, height = viewer_size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid viewer dimensions")
    tangent = math.tan(math.radians(horizontal_fov) / 2)
    x, y = float(mouse_xy[0]), float(mouse_xy[1])
    local = gymapi.Vec3(-(2 * x - 1) * tangent,
                       (1 - 2 * y) * tangent * height / width, 1)
    ray = viewer_pose.r.rotate(local)
    direction = np.asarray([ray.x, ray.y, ray.z], dtype=np.float64)
    origin = np.asarray([viewer_pose.p.x, viewer_pose.p.y, viewer_pose.p.z])
    return origin, direction / np.linalg.norm(direction)


def ray_box_distance(origin, direction, lower, upper):
    """Nearest nonnegative ray/AABB intersection, or None."""
    near, far = 0.0, float("inf")
    for start, delta, lo, hi in zip(origin, direction, lower, upper):
        if abs(delta) < 1e-12:
            if start < lo or start > hi:
                return None
            continue
        a, b = (lo - start) / delta, (hi - start) / delta
        near, far = max(near, min(a, b)), min(far, max(a, b))
        if far < near:
            return None
    return near


class _VideoClip:
    """Five synchronous pipes: memory use is independent of recording duration."""

    def __init__(self, directory, width, height, fps, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.processes, self.logs = {}, []
        self.frames = None
        self.closed = False
        self.frame_count = 0
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Finger-camera recording requires ffmpeg on PATH")
        try:
            (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
            self.frames = (self.directory / "frames.jsonl").open("w", buffering=1)
            for finger in FINGERS:
                log = (self.directory / (finger + ".ffmpeg.log")).open("wb")
                self.logs.append(log)
                command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin",
                           "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
                           "{}x{}".format(width, height), "-framerate", "{:.10g}".format(fps),
                           "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1",
                           "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(self.directory / (finger + ".mp4"))]
                self.processes[finger] = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        except BaseException:
            self.close()
            raise

    def write(self, images, frame):
        for finger, process in self.processes.items():
            if process.poll() is not None:
                raise RuntimeError("Camera encoder exited; see " + str(self.directory))
            process.stdin.write(np.ascontiguousarray(images[finger], dtype=np.uint8).tobytes())
        self.frames.write(json.dumps(dict(frame, frame=self.frame_count), allow_nan=False) + "\n")
        self.frame_count += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Close every input first so all encoders can finalize concurrently.
        for process in self.processes.values():
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        for finger, process in self.processes.items():
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            if code:
                warnings.warn("{} camera encoder failed ({}); see {}".format(
                    finger, code, self.directory))
        if self.frames is not None:
            self.frames.close()
        for log in self.logs:
            log.close()


class HandCameraMonitor:
    """Create before prepare_sim; bind the viewer once it exists; capture pre-reset."""

    def __init__(self, task, config):
        self.task, self.gym, self.sim = task, task.gym, task.sim
        self.config = dict(config)
        self.env_ids = resolve_env_ids(config, task.num_envs)
        self.selected_id = int(config.get("env_id", 0))
        self.display = bool(config.get("display", False))
        self.record = bool(config.get("record", False))
        if self.display and task.headless:
            raise ValueError("camera.display=True requires headless=False; use camera.record for headless video")
        self.width, self.height = int(config.get("width", 112)), int(config.get("height", 112))
        self.horizontal_fov = float(config.get("horizontal_fov", 90))
        self.capture_every = int(config.get("capture_every", 1))
        self.segment_steps = int(config.get("segment_steps", 300))
        self.interval_steps = int(config.get("record_interval_steps", 0))
        self.record_steps = int(config.get("record_steps", 200))
        near, far = float(config.get("near_plane", .001)), float(config.get("far_plane", 1))
        if (min(self.width, self.height) < 2 or self.width % 2 or self.height % 2
                or not 0 < self.horizontal_fov < 180 or not 0 < near < far
                or min(self.capture_every, self.segment_steps, self.record_steps) < 1
                or self.interval_steps < 0
                or (self.interval_steps and self.record_steps > self.interval_steps)):
            raise ValueError("Invalid camera dimensions, FOV, clipping planes or recording cadence")
        self.control_dt = float(task.dt) * int(task.control_freq_inv)
        self.fps = 1 / (self.control_dt * self.capture_every)
        self.viewer = None
        self.window = None
        self.cameras = {}
        self.selecting = False
        self._lines_drawn = False
        self._mouse_available = True
        self._closed = False
        self.control_step = 0
        self._episode_ids = {env_id: 0 for env_id in self.env_ids}
        self._clip = None
        self._clip_start = 0
        self._clip_index = 0
        self._record_origin = 0
        self._output_directory = None
        self._photos = {}
        self._labels = {}
        self._status = None
        self._env_var = self._record_var = None
        mounts = config.get("mounts", {})
        if set(mounts) != set(FINGERS):
            raise ValueError("camera.mounts must define thumb, index, middle, ring and little")
        # Hydra can leave nested ListConfig values inside a regular dict.
        self._mounts = {finger: dict(link=str(mount["link"]),
            position=[float(v) for v in mount.get("position", [0, 0, 0])],
            rotation=[float(v) for v in mount.get("rotation", [0, 0, 0])])
            for finger, mount in mounts.items()}

        # Fixed-base hand proxy, expressed in each environment's coordinates.
        env_cfg = task.cfg["env"]
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*env_cfg.get("handPosition", [0, 0, .30]))
        pose.r = gymapi.Quat.from_euler_zyx(*env_cfg.get("handRotationEuler", [-math.pi / 2, 0, 0]))
        corners = [pose.p + pose.r.rotate(gymapi.Vec3(x, y, z))
                   for x in (-.105, .105) for y in (-.10, .025) for z in (-.025, .20)]
        corners = np.asarray([[v.x, v.y, v.z] for v in corners], dtype=np.float64)
        self._box_lower, self._box_upper = corners.min(axis=0), corners.max(axis=0)
        self._box_lines = self._make_box_lines(self._box_lower, self._box_upper)
        props = gymapi.CameraProperties()
        props.width, props.height = self.width, self.height
        props.horizontal_fov, props.near_plane, props.far_plane = self.horizontal_fov, near, far
        props.enable_tensors = False
        try:
            for env_id in self.env_ids:
                env, hand = task.envs[env_id], task.hands[env_id]
                self.cameras[env_id] = {}
                for finger in FINGERS:
                    mount = self._mounts[finger]
                    body = self.gym.find_actor_rigid_body_handle(env, hand, str(mount["link"]))
                    if body < 0:
                        raise ValueError("Camera link not found: " + str(mount["link"]))
                    transform = gymapi.Transform()
                    transform.p = gymapi.Vec3(*(float(v) for v in mount.get("position", [0, 0, 0])))
                    transform.r = gymapi.Quat.from_euler_zyx(
                        *(float(v) for v in mount.get("rotation", [0, 0, 0])))
                    handle = self.gym.create_camera_sensor(env, props)
                    if handle < 0:
                        raise RuntimeError("Failed to create finger camera; a graphics device is required")
                    self.cameras[env_id][finger] = handle
                    self.gym.attach_camera_to_body(handle, env, body, transform, gymapi.FOLLOW_TRANSFORM)
        except Exception:
            self.close()
            raise
        print("Finger cameras: {} environments, {} sensors, {}x{}, {:.3f} FPS".format(
            len(self.env_ids), 5 * len(self.env_ids), self.width, self.height, self.fps))

    @staticmethod
    def _make_box_lines(lower, upper):
        vertices = np.asarray([[x, y, z] for x in (lower[0], upper[0])
                               for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
        edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]
        return np.asarray([[vertices[i], vertices[j]] for i, j in edges], dtype=np.float32)

    def bind_viewer(self, viewer):
        self.viewer = viewer
        if self.display:
            self._create_window()
            self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "hand_camera_pick")
            self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT_BRACKET, "hand_camera_previous")
            self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT_BRACKET, "hand_camera_next")
            self.gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "hand_camera_click")

    def _create_window(self):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        self._Image, self._ImageTk = Image, ImageTk
        self.window = tk.Tk()
        self.window.title("Finger cameras — env {}".format(self.selected_id))
        self.window.protocol("WM_DELETE_WINDOW", self._close_preview)
        controls = ttk.Frame(self.window, padding=6)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(controls, text="Hand:").pack(side="left")
        self._env_var = tk.StringVar(value=str(self.selected_id))
        selector = ttk.Combobox(controls, textvariable=self._env_var,
                                values=[str(i) for i in self.env_ids], width=6, state="readonly")
        selector.pack(side="left", padx=4)
        selector.bind("<<ComboboxSelected>>", lambda _event: self.select_env(int(self._env_var.get())))
        ttk.Button(controls, text="[", command=lambda: self._cycle(-1), width=3).pack(side="left")
        ttk.Button(controls, text="]", command=lambda: self._cycle(1), width=3).pack(side="left")
        self._record_var = tk.BooleanVar(value=self.record)
        ttk.Checkbutton(controls, text="Record", variable=self._record_var,
                        command=lambda: self.set_recording(self._record_var.get())).pack(side="right")
        for i, finger in enumerate(FINGERS):
            frame = ttk.LabelFrame(self.window, text=finger.title(), padding=3)
            frame.grid(row=1 + i // 2, column=i % 2, padx=4, pady=4)
            label = ttk.Label(frame)
            label.pack()
            self._labels[finger] = label
        self._status = tk.StringVar(value="Viewer: P + click a hand; [ / ] to switch")
        ttk.Label(self.window, textvariable=self._status, padding=6).grid(
            row=4, column=0, columnspan=2, sticky="w")
        self.window.update_idletasks()
        self.window.geometry("+{}+40".format(max(0, self.window.winfo_screenwidth() - 490)))

    def _close_preview(self):
        self.display = False
        self.selecting = False
        if self.window is not None:
            self.window.destroy()
            self.window = None
        self._photos.clear()
        if self.record:
            print("Camera preview closed; recording continues. Stop the run to finalize remaining video.")

    def poll(self):
        if self.window is not None:
            self.window.update_idletasks()
            self.window.update()

    def select_env(self, env_id):
        if env_id not in self.env_ids:
            raise ValueError("This environment has no cameras: {}".format(env_id))
        if env_id == self.selected_id:
            return
        self._finish_clip()
        self.selected_id = env_id
        self._record_origin = self.control_step
        if self.window is not None:
            self._env_var.set(str(env_id))
            self.window.title("Finger cameras — env {}".format(env_id))
            # Clear stale images immediately: never label the previous hand as this one.
            for label in self._labels.values():
                label.configure(image="")
            self._photos.clear()

    def _cycle(self, offset):
        index = (self.env_ids.index(self.selected_id) + offset) % len(self.env_ids)
        self.select_env(self.env_ids[index])

    def set_recording(self, enabled):
        enabled = bool(enabled)
        if self.record != enabled:
            self._finish_clip()
            self.record = enabled
            self._record_origin = self.control_step
        if self.window is not None:
            self._record_var.set(enabled)

    def handle_event(self, event):
        if event.value <= 0 or not self.display:
            return
        if event.action == "hand_camera_pick":
            self.selecting = not self.selecting
        elif event.action == "hand_camera_previous":
            self._cycle(-1)
        elif event.action == "hand_camera_next":
            self._cycle(1)
        elif event.action == "hand_camera_click" and self.selecting and self._mouse_available:
            self._pick_hand()

    def _pick_hand(self):
        mouse = self.gym.get_viewer_mouse_position(self.viewer)
        if not (0 <= mouse.x <= 1 and 0 <= mouse.y <= 1):
            return
        matches = []
        try:
            size = self.gym.get_viewer_size(self.viewer)
            fov = self.task.viewer_camera_properties.horizontal_fov
            for env_id in self.env_ids:
                env = self.task.envs[env_id]
                pose = self.gym.get_viewer_camera_transform(self.viewer, env)
                origin, direction = mouse_ray((mouse.x, mouse.y), pose, fov, (size.x, size.y))
                distance = ray_box_distance(origin, direction, self._box_lower, self._box_upper)
                if distance is not None:
                    matches.append((distance, env_id))
        except (RuntimeError, ValueError) as error:
            self._mouse_available = False
            warnings.warn("Viewer projection unavailable; use [ / ] or the camera selector: " + str(error))
            return
        if matches:
            self.select_env(min(matches)[1])
            self.selecting = False

    def draw_selection(self):
        if self.viewer is None or (not self.display and not self._lines_drawn):
            return
        self.gym.clear_lines(self.viewer)
        self._lines_drawn = False
        if not self.display:
            return
        for env_id in self.env_ids:
            if env_id == self.selected_id or self.selecting:
                rgb = (1., .85, 0.) if env_id == self.selected_id else (.3, .8, 1.)
                colors = np.tile(np.asarray(rgb, dtype=np.float32), (len(self._box_lines), 1))
                self.gym.add_lines(self.viewer, self.task.envs[env_id], len(self._box_lines),
                                   self._box_lines, colors)
                self._lines_drawn = True

    def _record_now(self):
        if not self.record:
            return False
        return (not self.interval_steps or
                (self.control_step - 1 - self._record_origin) % self.interval_steps < self.record_steps)

    def _start_clip(self):
        if self._output_directory is None:
            root = str(self.config.get("output_dir", ""))
            if not root:
                raise ValueError("camera.output_dir must be set to the current task's run directory")
            mode = str(self.config.get("mode", "train"))
            if mode not in ("train", "replay"):
                raise ValueError("camera.mode must be train or replay")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns())
            self._output_directory = Path(root).expanduser() / "videos" / mode / stamp
        self._clip_index += 1
        directory = self._output_directory / "env_{:04d}".format(self.selected_id) / (
            "segment_{:04d}".format(self._clip_index))
        metadata = dict(environment=self.selected_id, width=self.width, height=self.height,
                        horizontal_fov_degrees=self.horizontal_fov, fps=self.fps,
                        control_dt=self.control_dt, capture_every=self.capture_every,
                        start_control_step=self.control_step,
                        checkpoint=str(self.config.get("checkpoint", "")),
                        mounts=self._mounts, camera_model="pinhole; approximate optical mounts",
                        timing="simulation time; synchronized five views; captured before reset")
        self._clip = _VideoClip(directory, self.width, self.height, self.fps, metadata)
        self._clip_start = self.control_step
        print("Recording finger cameras: " + str(directory))

    def _finish_clip(self):
        if self._clip is not None:
            clip, self._clip = self._clip, None
            clip.close()

    def capture(self):
        """Call exactly once after each completed control step, before auto-reset."""
        if self._closed:
            return
        self.control_step += 1
        self.poll()
        recording = self._record_now()
        if not recording:
            self._finish_clip()
        sampled = self.control_step % self.capture_every == 0
        if sampled and (self.display or recording):
            # Debug lines also appear in sensor renders. Keep selection overlays
            # in the Viewer only; draw_selection restores them on its next frame.
            if self.viewer is not None and self._lines_drawn:
                self.gym.clear_lines(self.viewer)
                self._lines_drawn = False
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            images = {}
            for finger, handle in self.cameras[self.selected_id].items():
                rgba = self.gym.get_camera_image(self.sim, self.task.envs[self.selected_id],
                                                  handle, gymapi.IMAGE_COLOR)
                images[finger] = np.asarray(rgba, dtype=np.uint8).reshape(
                    self.height, self.width, 4)[:, :, :3].copy()
            if self.window is not None:
                for finger, rgb in images.items():
                    image = self._Image.fromarray(rgb).resize((self.width * 2, self.height * 2))
                    self._photos[finger] = self._ImageTk.PhotoImage(image, master=self.window)
                    self._labels[finger].configure(image=self._photos[finger])
                self._status.set("env {} | step {} | {}{}".format(
                    self.selected_id, self.control_step, "REC" if recording else "preview",
                    " | click a cyan/yellow box" if self.selecting else " | P: select; [ / ]: switch"))
            if recording:
                try:
                    if self._clip is None:
                        self._start_clip()
                    env_id = self.selected_id
                    reward = float(self.task.rew_buf[env_id].item())
                    self._clip.write(images, dict(control_step=self.control_step,
                        sim_time=self.control_step * self.control_dt,
                        episode=self._episode_ids[env_id], done=bool(self.task.reset_buf[env_id].item()),
                        reward=reward if math.isfinite(reward) else None))
                    if self.control_step - self._clip_start + self.capture_every >= self.segment_steps:
                        self._finish_clip()
                except (OSError, RuntimeError, ValueError) as error:
                    self.set_recording(False)
                    warnings.warn("Finger-camera recording stopped: " + str(error))
        # Count episode boundaries even during an interval with no video frames.
        for env_id in self.env_ids:
            self._episode_ids[env_id] += int(bool(self.task.reset_buf[env_id].item()))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._finish_clip()
        if self.window is not None:
            self.window.destroy()
            self.window = None
        # The Sim owns sensor render targets. Release them with destroy_sim,
        # after destroy_viewer, matching the SDK graphics example's lifecycle.
        self.cameras.clear()
        self._photos.clear()
