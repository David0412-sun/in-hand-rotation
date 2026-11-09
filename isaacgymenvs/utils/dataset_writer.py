"""Bounded, lossless streaming of synchronized samples to HDF5 shards.

Only the writer thread accesses HDF5. ``submit`` takes an owned CPU copy of
every field, so neither simulator tensors nor camera buffers outlive a step.
Each row is a post-control-step observation paired with its preceding action;
the caller records exact field meanings and camera order in ``metadata``.
"""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import uuid

import numpy as np


SCHEMA_VERSION = 1
_STOP = object()
_COMMITTED = "_committed_rows"


class _FlushRequest:
    def __init__(self):
        self.done = threading.Event()


class HDF5DatasetWriter:
    """Write fixed-schema samples without unbounded memory or silent drops.

    ``output_dir`` is a parent directory; each instance creates a unique child
    directory. Numeric arrays retain their supplied dtype. HDF5 dataset paths
    may contain ``/``. For example ``rgb/fingertip`` can have shape
    ``[5,112,112,3]`` and dtype ``uint8``. Flush/close propagate background errors.
    """

    def __init__(self, output_dir, metadata, shard_size=1000, queue_size=8,
                 compression="lzf", flush_every=16):
        try:
            import h5py
        except ImportError as error:
            raise ImportError("Dataset collection requires h5py; install it in the active Python environment") from error
        for name, value in (("shard_size", shard_size), ("queue_size", queue_size),
                            ("flush_every", flush_every)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        if compression == "none":
            compression = None
        if compression not in (None, "lzf", "gzip"):
            raise ValueError("compression must be lzf, gzip or none")
        self._h5py = h5py
        self.shard_size = int(shard_size)
        self.flush_every = int(flush_every)
        self.compression = compression
        # A JSON round trip both isolates metadata and rejects tensor/NaN values.
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self._metadata_json = json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory = Path(output_dir).expanduser().resolve() / (stamp + "_" + uuid.uuid4().hex[:8])
        self.directory.mkdir(parents=True, exist_ok=False)
        self._queue = queue.Queue(maxsize=int(queue_size))
        self._submit_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._error = None
        self._closing = False
        self._closed = False
        self._submitted_rows = 0
        self._committed_rows = 0
        self._shards = []
        self._schema = None
        self._file = None
        self._datasets = None
        self._rows_in_shard = 0
        self._last_flush_rows = 0
        self._write_json("metadata.json", dict(self.metadata, schema_version=SCHEMA_VERSION))
        self._write_manifest(closed=False)
        self._thread = threading.Thread(target=self._run, name="hdf5-dataset-writer", daemon=True)
        self._thread.start()

    @property
    def submitted_rows(self):
        with self._state_lock:
            return self._submitted_rows

    @property
    def committed_rows(self):
        with self._state_lock:
            return self._committed_rows

    @property
    def shards(self):
        with self._state_lock:
            return copy.deepcopy(self._shards)

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError("Dataset writer failed: {}".format(self._error)) from self._error

    @staticmethod
    def _copy_sample(sample):
        if not isinstance(sample, dict) or not sample:
            raise ValueError("sample must be a nonempty dictionary of field paths")
        values = {}
        for name, value in sample.items():
            if (not isinstance(name, str) or not name or "\x00" in name
                    or name.split("/")[0] == _COMMITTED
                    or any(part in ("", ".", "..") for part in name.split("/"))):
                raise ValueError("Invalid or reserved HDF5 field path: {!r}".format(name))
            if hasattr(value, "detach") and hasattr(value, "device"):
                raise TypeError("{} is a tensor; pass an explicit CPU NumPy copy".format(name))
            array = np.array(value, copy=True, order="C")
            if array.dtype.kind not in "biuf":
                raise TypeError("{} must contain numeric or boolean data, got {}".format(name, array.dtype))
            if any(size == 0 for size in array.shape):
                raise ValueError("{} contains an empty dimension".format(name))
            values[name] = array
        paths = set(values)
        for name in paths:
            parts = name.split("/")
            if any("/".join(parts[:index]) in paths for index in range(1, len(parts))):
                raise ValueError("A field path cannot also be another field's parent: {}".format(name))
        return values

    def _put(self, item):
        while True:
            self._raise_error()
            if not self._thread.is_alive():
                raise RuntimeError("Dataset writer thread is no longer running")
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def submit(self, sample):
        """Queue an owned sample; a full queue pauses the caller instead of dropping it."""
        values = self._copy_sample(sample)
        with self._submit_lock:
            self._raise_error()
            if self._closing or self._closed:
                raise RuntimeError("Cannot submit to a closing dataset writer")
            self._put(values)
            with self._state_lock:
                self._submitted_rows += 1

    def flush(self):
        """Wait until all previously submitted samples are committed to disk."""
        request = _FlushRequest()
        with self._submit_lock:
            self._raise_error()
            if self._closed:
                return
            if self._closing:
                raise RuntimeError("Dataset writer is closing")
            self._put(request)
        while not request.done.wait(timeout=0.1):
            self._raise_error()
        self._raise_error()

    def close(self):
        """Drain the queue and close every shard, reporting any background failure."""
        with self._submit_lock:
            if self._closed:
                self._raise_error()
                return
            if not self._closing:
                self._closing = True
                try:
                    self._put(_STOP)
                except RuntimeError:
                    # An errored worker still closes its own file in finally.
                    if self._error is None:
                        raise
        self._thread.join()
        self._closed = True
        self._raise_error()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _write_json(self, name, value):
        path = self.directory / name
        temporary = self.directory / ("." + name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _write_manifest(self, closed=False):
        with self._state_lock:
            manifest = dict(schema_version=SCHEMA_VERSION, closed=closed,
                            committed_rows=self._committed_rows,
                            shards=copy.deepcopy(self._shards))
        if self._error is not None:
            manifest["error"] = "{}: {}".format(type(self._error).__name__, self._error)
        self._write_json("manifest.json", manifest)

    def _open_shard(self, sample):
        schema = {name: dict(shape=list(value.shape), dtype=value.dtype.str)
                  for name, value in sorted(sample.items())}
        if self._schema is None:
            self._schema = schema
        elif schema != self._schema:
            raise ValueError("Sample fields, shapes or dtypes differ from the first sample")
        index = len(self._shards)
        filename = "shard_{:05d}.h5".format(index)
        self._file = self._h5py.File(str(self.directory / filename), "x", libver="latest")
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["metadata_json"] = self._metadata_json
        self._file.attrs["sample_schema_json"] = json.dumps(self._schema, sort_keys=True)
        self._file.attrs["committed_rows"] = np.uint64(0)
        self._file.create_dataset(_COMMITTED, data=np.uint64(0))
        self._datasets = {}
        for name, value in sorted(sample.items()):
            image = value.dtype == np.uint8 and value.ndim >= 3 and value.shape[-1] in (3, 4)
            chunks = (1 if image else min(128, self.shard_size),) + value.shape
            options = dict(shape=(0,) + value.shape, maxshape=(None,) + value.shape,
                           chunks=chunks, dtype=value.dtype, compression=self.compression)
            if self.compression == "gzip":
                options["compression_opts"] = 1
            if self.compression is not None and value.dtype.itemsize > 1:
                options["shuffle"] = True
            self._datasets[name] = self._file.create_dataset(name, **options)
        self._file.flush()
        self._file.swmr_mode = True
        self._rows_in_shard = 0
        self._last_flush_rows = 0
        with self._state_lock:
            self._shards.append(dict(file=filename, rows=0, closed=False))

    def _append(self, sample):
        schema = {name: dict(shape=list(value.shape), dtype=value.dtype.str)
                  for name, value in sorted(sample.items())}
        if self._schema is not None and schema != self._schema:
            raise ValueError("Sample fields, shapes or dtypes differ from the first sample")
        if self._file is None:
            self._open_shard(sample)
        row = self._rows_in_shard
        for name, value in sample.items():
            dataset = self._datasets[name]
            dataset.resize(row + 1, axis=0)
            dataset[row] = value
        self._rows_in_shard += 1
        if self._rows_in_shard - self._last_flush_rows >= self.flush_every:
            self._commit()
        if self._rows_in_shard >= self.shard_size:
            self._finish_shard()

    def _commit(self):
        if self._file is None or self._rows_in_shard == self._last_flush_rows:
            return
        # Flush every field before advancing the marker used by SWMR readers.
        self._file.flush()
        self._file[_COMMITTED][()] = np.uint64(self._rows_in_shard)
        self._file.attrs.modify("committed_rows", np.uint64(self._rows_in_shard))
        self._file.flush()
        with self._state_lock:
            self._committed_rows += self._rows_in_shard - self._last_flush_rows
            self._shards[-1]["rows"] = self._rows_in_shard
        self._last_flush_rows = self._rows_in_shard
        self._write_manifest(closed=False)

    def _finish_shard(self):
        if self._file is None:
            return
        self._commit()
        self._file.close()
        self._file = None
        self._datasets = None
        with self._state_lock:
            self._shards[-1]["closed"] = True
        self._write_manifest(closed=False)

    def _run(self):
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        break
                    if isinstance(item, _FlushRequest):
                        self._commit()
                        item.done.set()
                    else:
                        self._append(item)
                finally:
                    self._queue.task_done()
            self._finish_shard()
        except BaseException as error:
            self._error = error
        finally:
            if self._file is not None:
                # A schema/IO failure can leave a partially extended next row.
                # Only _rows_in_shard contains fully written rows, so preserve
                # those valid preceding samples without committing the partial row.
                try:
                    self._commit()
                except BaseException as error:
                    if self._error is None:
                        self._error = error
                try:
                    self._file.close()
                except BaseException as error:
                    if self._error is None:
                        self._error = error
                self._file = None
            try:
                self._write_manifest(closed=self._error is None)
            except BaseException as error:
                if self._error is None:
                    self._error = error
            # Wake a pending flush if an earlier queued sample failed.
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending, _FlushRequest):
                    pending.done.set()
                self._queue.task_done()


def inspect_dataset(directory):
    """Read a collection manifest and only the committed extent of each shard."""
    import h5py

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    report = dict(directory=str(directory), closed=manifest["closed"], committed_rows=0, shards=[])
    for shard in manifest["shards"]:
        with h5py.File(str(directory / shard["file"]), "r", swmr=True) as handle:
            rows = int(handle[_COMMITTED][()])
            schema = json.loads(handle.attrs["sample_schema_json"])
            if any(handle[name].shape[0] < rows for name in schema):
                raise ValueError("Shard commit marker exceeds a field's extent: " + shard["file"])
            report["shards"].append(dict(file=shard["file"], rows=rows, fields=schema))
            report["committed_rows"] += rows
    return report


def read_sample(directory, shard_index, row):
    """Return one committed row as a flat dictionary of NumPy field values."""
    import h5py

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    shard = manifest["shards"][shard_index]
    with h5py.File(str(directory / shard["file"]), "r", swmr=True) as handle:
        rows = int(handle[_COMMITTED][()])
        if not 0 <= row < rows:
            raise IndexError("Requested row is outside the committed shard extent")
        schema = json.loads(handle.attrs["sample_schema_json"])
        return {name: np.array(handle[name][row], copy=True) for name in schema}
