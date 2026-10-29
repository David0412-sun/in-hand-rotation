"""Persistent GPU-memory diagnostics without changing the CUDA workload.

Device/process memory comes from nvidia-smi at each recorded endpoint; these
values are not peaks between records. PyTorch peaks cover its allocator only,
not PhysX or rendering allocations. Physical GPUs are all recorded, while
``torch_device`` and ``cuda_visible_devices`` identify the separate CUDA view.
Device totals include other applications and desktop graphics; process memory
contains only this logger's PID, when nvidia-smi makes that value available.
"""

import csv
from datetime import datetime
import importlib
import os
from pathlib import Path
import subprocess
import warnings


class GPUMemoryLogger:
    FIELDS = (
        "session_id", "pid", "timestamp", "epoch", "phase", "frame",
        "gpu_index", "gpu_uuid", "device_used_mib", "device_free_mib",
        "device_total_mib", "process_used_mib", "torch_device",
        "cuda_visible_devices", "torch_allocated_mib", "torch_reserved_mib",
        "torch_peak_allocated_mib", "torch_peak_reserved_mib", "error",
    )

    def __init__(self, output_dir, device="cuda:0"):
        self._path = Path(output_dir) / "summaries" / "gpu_memory.csv"
        self.device = str(device)
        self.pid = os.getpid()
        self.session_id = "{}-{}".format(
            datetime.now().astimezone().strftime("%Y%m%dT%H%M%S.%f%z"), self.pid
        )
        self._visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._file = None
        self._writer = None
        self._torch = None
        self._warned = False
        self._disabled = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
            if self._file.tell() == 0:
                self._writer.writeheader()
                self._flush()
        except Exception as exc:
            self._disable(exc)

    @property
    def path(self):
        return str(self._path)

    @staticmethod
    def _message(exc):
        return " ".join(str(exc).split())[:1000]

    def _warn(self, message, force=False):
        if force or not self._warned:
            self._warned = True
            # Warning filters must not turn optional monitoring into a failure.
            try:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            except Exception:
                pass

    def _disable(self, exc):
        self._disabled = True
        self._warn("GPU memory logging disabled: {}".format(self._message(exc)),
                   force=True)
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def _flush(self):
        self._file.flush()
        os.fsync(self._file.fileno())

    @staticmethod
    def _number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return ""

    def _query(self, query, notes):
        try:
            result = subprocess.run(
                ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
            )
            return [[cell.strip() for cell in row]
                    for row in csv.reader(result.stdout.splitlines()) if row]
        except Exception as exc:
            note = "nvidia-smi {}: {}".format(query.split("=")[0], self._message(exc))
            notes.append(note)
            self._warn(note)
            return []

    def _gpu_rows(self, notes):
        devices = self._query(
            "--query-gpu=index,uuid,memory.used,memory.free,memory.total", notes
        )
        processes = self._query(
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", notes
        )
        process_memory = {}
        for fields in processes:
            if len(fields) == 3 and fields[1] == str(self.pid):
                value = self._number(fields[2])
                if value != "":
                    process_memory[fields[0]] = process_memory.get(fields[0], 0) + value
        rows = []
        for fields in devices:
            if len(fields) != 5:
                note = "nvidia-smi returned an unexpected GPU record"
                notes.append(note)
                self._warn(note)
                continue
            rows.append(dict(
                gpu_index=fields[0], gpu_uuid=fields[1],
                device_used_mib=self._number(fields[2]),
                device_free_mib=self._number(fields[3]),
                device_total_mib=self._number(fields[4]),
                process_used_mib=process_memory.get(fields[1], ""),
            ))
        return rows or [{}]

    def _torch_cuda(self):
        if self._torch is None:
            self._torch = importlib.import_module("torch")
        return self._torch.cuda

    def _torch_stats(self, notes):
        stats = {}
        if not self.device.startswith("cuda"):
            return stats
        try:
            cuda = self._torch_cuda()
            if not cuda.is_initialized():
                return stats
            for column, method in (
                ("torch_allocated_mib", "memory_allocated"),
                ("torch_reserved_mib", "memory_reserved"),
                ("torch_peak_allocated_mib", "max_memory_allocated"),
                ("torch_peak_reserved_mib", "max_memory_reserved"),
            ):
                stats[column] = getattr(cuda, method)(self.device) / (1024 ** 2)
        except Exception as exc:
            note = "PyTorch memory statistics: {}".format(self._message(exc))
            notes.append(note)
            self._warn(note)
        return stats

    def record(self, epoch, phase, frame=0, error=None, include_torch=True):
        if self._disabled:
            return
        notes = [self._message(error)] if error is not None else []
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        rows = self._gpu_rows(notes)
        stats = self._torch_stats(notes) if include_torch else {}
        common = dict(
            session_id=self.session_id, pid=self.pid, timestamp=timestamp,
            epoch=epoch, phase=phase, frame=frame, torch_device=self.device,
            cuda_visible_devices=self._visible_devices, error="; ".join(notes),
            **stats
        )
        try:
            for row in rows:
                self._writer.writerow(dict(common, **row))
            self._flush()
        except Exception as exc:
            self._disable(exc)

    def begin_epoch(self, epoch, frame=0):
        if self._disabled:
            return
        error = None
        if self.device.startswith("cuda"):
            try:
                cuda = self._torch_cuda()
                if cuda.is_initialized():
                    cuda.reset_peak_memory_stats(self.device)
            except Exception as exc:
                error = "PyTorch peak reset: {}".format(self._message(exc))
                self._warn(error)
        self.record(epoch, "epoch_start", frame=frame, error=error)

    def close(self):
        if self._disabled:
            return
        try:
            if self._file is not None:
                self._flush()
                self._file.close()
                self._file = None
            self._disabled = True
        except Exception as exc:
            self._disable(exc)
