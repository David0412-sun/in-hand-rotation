#!/usr/bin/env python3
"""Inspect committed HDF5 samples without importing Isaac Gym or PyTorch."""

import argparse
import json
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Collection directory containing manifest.json")
    parser.add_argument("--sample", type=int, default=0, help="Global committed sample index")
    parser.add_argument("--preview", type=Path, help="Optionally export a labelled six-camera preview")
    args = parser.parse_args()
    if args.sample < 0:
        parser.error("--sample must be nonnegative")
    manifest = json.loads((args.directory / "manifest.json").read_text())
    metadata = json.loads((args.directory / "metadata.json").read_text())
    shards, selected, offset, total = [], None, args.sample, 0
    for shard in manifest["shards"]:
        path = args.directory / shard["file"]
        with h5py.File(path, "r", swmr=True) as handle:
            rows = int(handle["_committed_rows"][()])
        shards.append({"file": shard["file"], "committed_rows": rows})
        if selected is None:
            if offset < rows:
                selected = (path, offset)
            else:
                offset -= rows
        total += rows
    summary = {"directory": str(args.directory), "closed": manifest["closed"],
               "committed_samples": total, "shards": shards,
               "camera_order": metadata["camera"]["order"],
               "pose_layout": metadata["pose_layout"], "control_dt": metadata["control_dt"]}
    if selected is None:
        print(json.dumps(summary, indent=2))
        parser.error("--sample is outside the committed dataset")
    with h5py.File(selected[0], "r", swmr=True) as handle:
        schema = json.loads(handle.attrs["sample_schema_json"])
        summary["sample"] = {
            "env_id": int(handle["index/env_id"][selected[1]]),
            "episode_id": int(handle["index/episode_id"][selected[1]]),
            "episode_step": int(handle["index/episode_step"][selected[1]]),
            "done": bool(handle["transition/done"][selected[1]]),
            "fields": schema}
        if args.preview:
            from PIL import Image, ImageDraw

            finger = handle["rgb/finger"][selected[1]]
            top = handle["rgb/top"][selected[1]]
            images = [top] + list(finger)
            labels = ["top"] + metadata["camera"]["order"][:5]
            tile, title = 256, 24
            canvas = Image.new("RGB", (3 * tile, 2 * (tile + title)), (32, 32, 32))
            draw = ImageDraw.Draw(canvas)
            for i, (name, rgb) in enumerate(zip(labels, images)):
                x, y = (i % 3) * tile, (i // 3) * (tile + title)
                draw.text((x + 8, y + 5), name, fill=(255, 255, 255))
                canvas.paste(Image.fromarray(rgb).resize((tile, tile)), (x, y + title))
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(args.preview)
            summary["preview"] = str(args.preview)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
