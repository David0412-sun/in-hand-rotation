#!/usr/bin/env python3
"""Replay saved sensor samples, without Isaac Gym, a policy or GPU allocation."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def quaternion_rotation(quaternion):
    """Rotation from local coordinates to the parent frame, using xyzw."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or norm < 1e-12:
        raise ValueError("Invalid pose quaternion")
    x, y, z, w = quaternion / norm
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + 2 * (w * skew + skew @ skew)


def pose_axis_points(pose, length):
    """World positions of the local origin and positive X/Y/Z axis endpoints."""
    pose = np.asarray(pose, dtype=np.float64)
    rotation = quaternion_rotation(pose[3:])
    return np.vstack((pose[:3], pose[:3] + length * rotation.T))


def project_world_points(points, camera_pose, intrinsic):
    """Project world points using a camera-to-world OpenCV optical pose (xyzw)."""
    pose = np.asarray(camera_pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Invalid camera world pose")
    rotation = quaternion_rotation(pose[3:])
    camera_points = (np.asarray(points, dtype=np.float64) - pose[:3]) @ rotation
    pixels = np.full((len(camera_points), 2), np.nan)
    valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-6)
    projected = camera_points[valid] @ np.asarray(intrinsic, dtype=np.float64).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, camera_points[:, 2]


class DatasetReplay:
    """Index committed rows once; keep only one HDF5 file and one image row open."""

    def __init__(self, directory, env_id=None, episode_id=None, world_axes=True, object_axes=True):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.world_axes = world_axes
        self.object_axes = object_axes
        self.top_index = self.metadata["camera"]["order"].index("top")
        self.top_calibration = self.metadata["camera"]["cameras"][self.top_index]
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.paths = [self.directory / s["file"] for s in manifest["shards"]]
        dtype = [("shard", "i4"), ("row", "i8"), ("env", "i4"),
                 ("episode", "u8"), ("time", "f8")]
        parts = []
        for shard, path in enumerate(self.paths):
            with h5py.File(path, "r", swmr=True) as handle:
                rows = int(handle["_committed_rows"][()])
                part = np.empty(rows, dtype=dtype)
                part["shard"], part["row"] = shard, np.arange(rows)
                part["env"] = handle["index/env_id"][:rows]
                part["episode"] = handle["index/episode_id"][:rows]
                part["time"] = handle["time/sim_seconds"][:rows]
                parts.append(part)
        records = np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
        if not len(records):
            raise ValueError("Dataset contains no committed samples")
        available = np.unique(records["env"])
        self.env_id = int(available.min()) if env_id is None else env_id
        records = records[records["env"] == self.env_id]
        if episode_id is not None:
            records = records[records["episode"] == episode_id]
        if not len(records):
            raise ValueError("No samples match env/episode selection; available env IDs: %s"
                             % available.tolist())
        self.records = records
        self.times = records["time"].copy()
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("Selected environment timestamps must be finite and strictly increasing")
        config = self.metadata.get("resolved_task_config", {}).get("dataset", {})
        self.last_duration = float(self.metadata["control_dt"]) * int(config.get("capture_every", 1))
        if not math.isfinite(self.last_duration) or self.last_duration <= 0:
            raise ValueError("Invalid control_dt/capture_every in metadata")
        self._handle = None
        self._shard = None
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
        self.font = ImageFont.truetype(str(font_path), 11) if font_path.exists() else ImageFont.load_default()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._shard = None

    def sample(self, index):
        record = self.records[index]
        shard, row = int(record["shard"]), int(record["row"])
        if self._shard != shard:
            self.close()
            self._handle = h5py.File(self.paths[shard], "r", swmr=True)
            self._shard = shard
        keys = ("rgb/top", "rgb/finger", "index/episode_id", "index/episode_step",
                "time/sim_seconds", "transition/done", "transition/reward",
                "tactile/contact", "state/joint_position", "state/joint_velocity",
                "object/pose_hand", "object/pose_world", "camera/pose_world")
        return {key: self._handle[key][row] for key in keys}

    def top_image(self, sample, tile):
        """Display-only world and object frames, without depth testing."""
        image = Image.fromarray(sample["rgb/top"]).resize((tile, tile))
        if self.world_axes:
            points = np.vstack((np.zeros(3), np.eye(3) * 0.10))
            self._draw_axes(image, sample, points, "W0", "W", dashed=True)
        if self.object_axes:
            for i, pose in enumerate(np.atleast_2d(sample["object/pose_world"])):
                name = "O%d" % i
                self._draw_axes(image, sample, pose_axis_points(pose, 0.05), name, name,
                                notice_row=i + 1)
        return image

    def _draw_axes(self, image, sample, points, origin_name, axis_prefix, dashed=False, notice_row=0):
        tile = image.width
        uv, depth = project_world_points(points, sample["camera/pose_world"][self.top_index],
                                         self.top_calibration["K"])
        # Match PIL's pixel-centre convention when resizing the source image.
        scale = np.asarray([tile / self.top_calibration["width"],
                            tile / self.top_calibration["height"]])
        uv = (uv + 0.5) * scale - 0.5
        near = float(self.metadata["camera"].get("top_camera", {}).get("near_plane", 0.01))
        visible = np.isfinite(uv).all(axis=1) & (depth >= near)
        inside = visible & (uv >= 0).all(axis=1) & (uv < tile).all(axis=1)
        draw = ImageDraw.Draw(image)

        def text_at(position, text, colour=(255, 255, 255)):
            draw.text(tuple(position), text, font=self.font, fill=colour,
                      stroke_width=1, stroke_fill=(0, 0, 0))

        if not inside[0]:
            text_at((6, 6 + 15 * notice_row), origin_name + " origin outside view")
            return
        origin = uv[0]
        for i, (name, colour) in enumerate(zip("XYZ", ((255, 65, 65), (45, 230, 80), (75, 145, 255))), 1):
            if not visible[i]:
                continue
            tip = uv[i]
            delta = tip - origin
            length = np.linalg.norm(delta)
            segments = [(start, start + 0.16) for start in (0., 0.24, 0.48, 0.72)] if dashed else [(0., 1.)]
            for start, end in segments:
                line = [tuple(origin + start * delta), tuple(origin + end * delta)]
                draw.line(line, fill=(0, 0, 0), width=4)
                draw.line(line, fill=colour, width=2)
            axis_name = axis_prefix + name
            if inside[i] and length >= 2:
                direction = delta / length
                side = np.asarray([-direction[1], direction[0]])
                arrow = min(7.0, length * 0.35)
                draw.polygon([tuple(tip), tuple(tip - arrow * direction + arrow * 0.5 * side),
                              tuple(tip - arrow * direction - arrow * 0.5 * side)], fill=colour)
                text_at(tip + [4, 2 if name == "Z" else -14], axis_name, colour)
            elif inside[i]:
                # An axis parallel to the optical axis can project to a point.
                draw.ellipse((tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3), outline=colour, width=2)
                text_at(origin + [4, 15 * (i - 1)], axis_name + (" toward camera" if depth[i] < depth[0] else " away from camera"), colour)
        draw.ellipse((origin[0] - 2, origin[1] - 2, origin[0] + 2, origin[1] + 2), fill=(255, 255, 255))
        text_at(origin + [-23, 4], origin_name)

    def panel(self, index):
        sample = self.sample(index)
        width, tile, title, header = 768, 256, 24, 32
        contacts = sample["tactile/contact"]
        q, qdot = sample["state/joint_position"], sample["state/joint_velocity"]
        poses = np.atleast_2d(sample["object/pose_hand"])
        world_poses = np.atleast_2d(sample["object/pose_world"])
        tactile_rows = int(math.ceil(len(contacts) / 3))
        joint_rows = int(math.ceil(len(q) / 4))
        height = header + 2 * (tile + title) + 24 + tactile_rows * 18 + 24 + joint_rows * 18 + 24 + len(poses) * 36 + 18 + 26
        height += height % 2  # yuv420p video dimensions must be even.
        canvas = Image.new("RGB", (width, height), (25, 25, 28))
        draw = ImageDraw.Draw(canvas)

        def label(x, y, text, fill=(235, 235, 235)):
            draw.text((x, y), str(text), font=self.font, fill=fill)

        label(8, 8, "Frame %d/%d | env %d | episode %d step %d | t %.3fs | done %d | r %.3f" % (
            index + 1, len(self.records), self.env_id, sample["index/episode_id"],
            sample["index/episode_step"], sample["time/sim_seconds"],
            sample["transition/done"], sample["transition/reward"]))
        names = ["top"] + self.metadata["camera"]["order"][:5]
        for i, (name, rgb) in enumerate(zip(names, [sample["rgb/top"]] + list(sample["rgb/finger"]))):
            x, y = (i % 3) * tile, header + (i // 3) * (tile + title)
            if i == 0:
                name += (" | W:10cm dashed" if self.world_axes else "")
                name += (" | O:5cm solid" if self.object_axes else "")
            label(x + 8, y + 5, name)
            frame = self.top_image(sample, tile) if i == 0 else Image.fromarray(rgb).resize((tile, tile))
            canvas.paste(frame, (x, y + title))
        y = header + 2 * (tile + title)
        label(8, y + 4, "Tactile: RED = no contact, GREEN = contact (recorded labels)")
        y += 24
        tactile_names = self.metadata["tactile_link_names"]
        for i, contact in enumerate(contacts):
            x, dy = (i // tactile_rows) * 256 + 8, y + (i % tactile_rows) * 18
            draw.rectangle((x, dy + 1, x + 11, dy + 12), fill=(45, 210, 80) if contact else (230, 55, 55))
            label(x + 18, dy, tactile_names[i])
        y += tactile_rows * 18
        label(8, y + 4, "Joint position q [rad] / velocity dq [rad/s]")
        y += 24
        for i, (name, position, velocity) in enumerate(zip(self.metadata["joint_names"], q, qdot)):
            x, dy = (i // joint_rows) * 192 + 8, y + (i % joint_rows) * 18
            label(x, dy, "%s  %+.3f / %+.3f" % (name, position, velocity))
        y += joint_rows * 18
        label(8, y + 4, "Object in WORLD: xyz [m], quat xyzw")
        label(392, y + 4, "Object in hand BASE: xyz [m], quat xyzw")
        y += 24
        for i, pose in enumerate(poses):
            for x, value in ((8, world_poses[i]), (392, pose)):
                label(x, y, "object %d xyz [%+.4f %+.4f %+.4f]" % (i, *value[:3]))
                label(x, y + 16, "quat [%+.4f %+.4f %+.4f %+.4f]" % tuple(value[3:]))
            y += 36
        origin = self.metadata["env_origins_world"][str(self.env_id)]
        label(8, y, "WORLD origin [0,0,0] | env origin [%+.3f %+.3f %+.3f] m | axes overlay, no depth test" % tuple(origin))
        y += 18
        label(8, y + 5, "SPACE pause | LEFT/RIGHT step | R restart | W world | O object | S video | Q/ESC close")
        return np.asarray(canvas)


def export_video(dataset, destination, speed):
    destination = Path(destination)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("--export must use an .mp4 filename")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing video: %s" % destination)
    executable = shutil.which("ffmpeg")
    if executable is None:
        sibling = Path(sys.executable).with_name("ffmpeg")
        executable = str(sibling) if sibling.is_file() else None
    if executable is None:
        raise RuntimeError("MP4 export requires ffmpeg; activate the robosyn environment")
    times = dataset.times - dataset.times[0]
    intervals = np.diff(times)
    # Use the smallest recorded interval so short terminal intervals are not discarded.
    interval = float(np.min(intervals)) if len(intervals) else dataset.last_duration
    fps = speed / interval
    count = int(math.ceil((times[-1] + dataset.last_duration) / interval - 1e-7))
    if len(intervals) and not np.allclose(intervals, interval, rtol=1e-5, atol=1e-8):
        print("Uneven timestamps: export holds each saved frame until the next sample; "
              "constant-rate video quantizes times to %.6f s." % interval)
    frame = dataset.panel(0)
    height, width = frame.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", "%dx%d" % (width, height),
               "-framerate", "%.12g" % fps, "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(destination)]
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
        try:
            previous = 0
            for frame_index in range(count):
                index = int(np.searchsorted(times, frame_index * interval + 1e-8, side="right") - 1)
                if index != previous:
                    frame = dataset.panel(index)
                    previous = index
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        except BaseException as exc:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            if process.poll() is None:
                process.terminate()
            process.wait()
            error_log.seek(0)
            details = error_log.read().decode("utf-8", errors="replace")
            if isinstance(exc, (BrokenPipeError, RuntimeError)):
                raise RuntimeError("Video export failed: " + details.strip()) from exc
            raise
    print("Exported %d frames, %.6f FPS, %.3f seconds: %s" % (count, fps, count / fps, destination))


class VideoSaveJob:
    """Export a selection snapshot with a separate reader, leaving the GUI free."""

    def __init__(self):
        self.thread = None
        self.results = queue.Queue()

    def start(self, dataset, speed):
        if self.thread is not None:
            return None
        directory, env_id = dataset.directory, dataset.env_id
        records, times = dataset.records.copy(), dataset.times.copy()
        paths, metadata = list(dataset.paths), dataset.metadata
        last_duration = dataset.last_duration
        world_axes, object_axes = dataset.world_axes, dataset.object_axes
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = directory / "videos" / ("replay_env%d_%s_%s.mp4" % (
            env_id, stamp, uuid.uuid4().hex[:6]))

        def run():
            reader, error = None, None
            try:
                reader = DatasetReplay(directory, env_id, world_axes=world_axes, object_axes=object_axes)
                # Preserve the exact episode selection and committed prefix seen
                # when S was pressed, even if collection is still appending rows.
                reader.records, reader.times = records, times
                reader.paths, reader.metadata = paths, metadata
                reader.last_duration = last_duration
                export_video(reader, destination, speed)
            except Exception as exc:
                error = str(exc)
            finally:
                if reader is not None:
                    try:
                        reader.close()
                    except Exception as exc:
                        error = error or str(exc)
                self.results.put((destination, error))

        self.thread = threading.Thread(target=run, name="sensor-video-export", daemon=False)
        try:
            self.thread.start()
        except Exception:
            self.thread = None
            raise
        return destination

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.thread.join()
        self.thread = None
        return result

    def wait(self):
        if self.thread is not None:
            self.thread.join()
        return self.poll()


def show_player(dataset, speed):
    import matplotlib

    matplotlib.use("TkAgg")
    # Reserve S for video; keep Ctrl+S available for Matplotlib's image dialog.
    matplotlib.rcParams["keymap.save"] = [key for key in matplotlib.rcParams["keymap.save"]
                                          if key.lower() != "s"]
    import matplotlib.pyplot as plt

    try:
        fig, ax = plt.subplots(figsize=(9, 10))
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Cannot open playback window. Run on a desktop, or use "
                           "--export replay.mp4 --no-display") from exc
    title = "Recorded sensor data | %.2fx | S: save full video" % speed
    fig.canvas.manager.set_window_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")
    artist = ax.imshow(dataset.panel(0))
    state = {"index": 0, "paused": False, "deadline": time.monotonic()}
    video = VideoSaveJob()

    def report_save(result, window=True):
        path, error = result
        if error is None:
            message = "Saved video: %s" % path.resolve()
            status = "Saved: " + path.name
        else:
            message = "Video save failed: %s (%s)" % (error, path)
            status = "Video save failed; see terminal. Press S to retry."
        print(message, flush=True)
        if window:
            fig.canvas.manager.set_window_title(status)

    def schedule():
        index = state["index"]
        delay = dataset.times[index + 1] - dataset.times[index] if index + 1 < len(dataset.records) else dataset.last_duration
        state["deadline"] = time.monotonic() + delay / speed

    def update(index):
        state["index"] = index
        artist.set_data(dataset.panel(index))
        fig.canvas.draw_idle()
        schedule()

    def tick():
        result = video.poll()
        if result is not None:
            report_save(result)
        if not state["paused"] and time.monotonic() >= state["deadline"]:
            if state["index"] + 1 < len(dataset.records):
                update(state["index"] + 1)
            else:
                state["paused"] = True

    def key(event):
        if event.key in ("q", "escape"):
            plt.close(fig)
        elif event.key == " ":
            state["paused"] = not state["paused"]
            schedule()
        elif event.key in ("left", "right"):
            state["paused"] = True
            update(max(0, min(len(dataset.records) - 1, state["index"] + (1 if event.key == "right" else -1))))
        elif event.key == "r":
            state["paused"] = False
            update(0)
        elif event.key in ("w", "o"):
            if event.key == "w":
                dataset.world_axes = not dataset.world_axes
            else:
                dataset.object_axes = not dataset.object_axes
            artist.set_data(dataset.panel(state["index"]))
            fig.canvas.draw_idle()
        elif event.key in ("s", "S"):
            if video.thread is not None:
                print("Video save already in progress; please wait.", flush=True)
                return
            try:
                destination = video.start(dataset, speed)
            except Exception as exc:
                print("Cannot start video save: %s" % exc, flush=True)
                fig.canvas.manager.set_window_title("Video save failed; see terminal. Press S to retry.")
                return
            print("Saving full selected replay to %s" % destination.resolve(), flush=True)
            fig.canvas.manager.set_window_title("Saving video... " + destination.name)

    timer = fig.canvas.new_timer(interval=10)
    timer.add_callback(tick)
    fig.canvas.mpl_connect("key_press_event", key)
    fig.canvas.mpl_connect("close_event", lambda event: timer.stop())
    schedule()
    timer.start()
    try:
        plt.show()
    finally:
        timer.stop()
        if video.thread is not None:
            print("Finishing video export before exiting...", flush=True)
            result = video.wait()
            if result is not None:
                report_save(result, window=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Timestamp directory containing manifest.json")
    parser.add_argument("--env-id", type=int, help="Replay one environment (default: smallest recorded ID)")
    parser.add_argument("--episode-id", type=int, help="Only replay this episode within the selected environment")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1)")
    parser.add_argument("--no-world-axes", action="store_true", help="Hide the top-view world-axis overlay; keep world pose values")
    parser.add_argument("--no-object-axes", action="store_true", help="Hide the top-view object-axis overlay")
    parser.add_argument("--export", type=Path, help="Optionally write an MP4; existing files are never overwritten")
    parser.add_argument("--no-display", action="store_true", help="Export without a desktop window; requires --export")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.no_display and args.export is None:
        parser.error("--no-display requires --export")
    dataset = None
    try:
        dataset = DatasetReplay(args.directory, args.env_id, args.episode_id,
                                world_axes=not args.no_world_axes, object_axes=not args.no_object_axes)
        print("Replaying %d saved samples from env %d; RGB/state are read from HDF5." % (
            len(dataset.records), dataset.env_id))
        if args.export:
            export_video(dataset, args.export, args.speed)
        if not args.no_display:
            show_player(dataset, args.speed)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, "Replay error: %s\n" % exc)
    except KeyboardInterrupt:
        pass
    finally:
        if dataset is not None:
            dataset.close()


if __name__ == "__main__":
    main()
