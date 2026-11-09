#!/usr/bin/env python3
"""Build fixed texture atlases for the chamfered block and source cube.

The source OBJ and its UV coordinates are read only. Rebuilding an existing
output is allowed only when both generated files are byte-identical. The
default solid_v3 profile gives each face one colour, chosen for high contrast
across shared mesh edges. Earlier fixed_v1 and simple_v2 profiles remain
reproducible. The cube_solid_v1 profile gives the source cube six fixed colours.
"""

import argparse
import colorsys
import hashlib
import io
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
MESH_FILE = "urdf/objects/meshes/set2/set_obj6_block_corner.obj"
MESH_SHA256 = "cff753bc2cad97232d872680a04b41610a7e4690ed94376da10a0ace238b5d0c"
GENERATOR_VERSION = "chamfered_uv_atlas_v1"
DEFAULT_SEED = 20261102
DEFAULT_OUTPUT = ROOT / "assets/textures/chamfered_block_solid_v3"
BORDER = 4
DARK = (18, 22, 28, 255)
SOLID_PROFILES = {
    "solid_v3": {
        "mesh_file": MESH_FILE, "mesh_sha256": MESH_SHA256,
        "layout": (24, 43, 26), "object_name": "set_obj6_block_corner",
        "generator_version": "chamfered_uv_atlas_solid_v3",
        "texture_prefix": "set_obj6_block_corner_solid_v3",
    },
    "cube_solid_v1": {
        "mesh_file": "urdf/objects/meshes/set2/set_obj1_regular_block.obj",
        "mesh_sha256": "a2d7b41f0e90e89b38c2f04d1a140eed4e7484547a1a8f29557d57918c41b190",
        "layout": (8, 14, 6), "object_name": "set_obj1_regular_block",
        "generator_version": "cube_uv_atlas_solid_v1",
        "texture_prefix": "set_obj1_regular_block_solid_v1",
    },
}

# Built-in bitmap glyphs keep the labels independent of installed system fonts.
GLYPHS = {
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_mesh(path, expected_hash=MESH_SHA256, expected_layout=(24, 43, 26)):
    data = Path(path).read_bytes()
    actual = sha256(data)
    if actual != expected_hash:
        raise ValueError("Source mesh hash mismatch: expected {}, got {}".format(expected_hash, actual))
    vertices, uvs, normals, faces = [], [], [], []
    for line in data.decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "vt":
            uvs.append([float(v) for v in fields[1:3]])
        elif fields[0] == "vn":
            normals.append([float(v) for v in fields[1:4]])
        elif fields[0] == "f":
            entries = [[int(v) - 1 for v in token.split("/")] for token in fields[1:]]
            if any(len(entry) != 3 for entry in entries):
                raise ValueError("Every face must have vertex, UV and normal indices")
            faces.append(entries)
    if (len(vertices), len(uvs), len(faces)) != expected_layout:
        raise ValueError("Unexpected source mesh layout")
    return vertices, uvs, normals, faces


def face_normal(vertices):
    a = [vertices[1][i] - vertices[0][i] for i in range(3)]
    b = [vertices[2][i] - vertices[0][i] for i in range(3)]
    normal = [a[1] * b[2] - a[2] * b[1],
              a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - a[1] * b[0]]
    length = math.sqrt(sum(value * value for value in normal))
    if length == 0:
        raise ValueError("Degenerate face")
    return [value / length for value in normal]


def rgb(hue, saturation, value):
    return tuple(int(component * 255 + 0.5) for component in
                 colorsys.hsv_to_rgb(hue % 1, saturation, value)) + (255,)


def fit_rectangle(mask, width, height):
    """Find a label rectangle fully inside the polygon, favouring its centre."""
    if width > mask.width or height > mask.height:
        return None
    xs = sorted(range(mask.width - width + 1), key=lambda x: (abs(x + width / 2 - mask.width / 2), x))
    ys = sorted(range(mask.height - height + 1), key=lambda y: (abs(y + height / 2 - mask.height / 2), y))
    for y in ys:
        for x in xs:
            box = (x, y, x + width, y + height)
            if mask.crop(box).getextrema()[0] == 255:
                return box
    return None


def draw_glyphs(draw, text, x, y, scale):
    for character in text:
        for row, pixels in enumerate(GLYPHS[character]):
            for column, active in enumerate(pixels):
                if active == "1":
                    left, top = x + column * scale, y + row * scale
                    draw.rectangle((left, top, left + scale - 1, top + scale - 1), fill=(255, 255, 255, 255))
        x += 6 * scale


def pattern_for_face(face_id, face_seed, interior):
    rng = random.Random(face_seed)
    width, height = interior.size
    base_hue = (face_id * 0.6180339887498949 + 0.137) % 1
    base = rgb(base_hue, 0.66, 0.84)
    tile = Image.new("RGBA", interior.size, base)
    draw = ImageDraw.Draw(tile)
    columns, rows = rng.randint(3, 5), rng.randint(3, 5)
    cell_colours = []
    for row in range(rows):
        for column in range(columns):
            colour = rgb(base_hue + rng.uniform(-0.22, 0.22), rng.uniform(0.40, 0.85), rng.uniform(0.50, 0.97))
            cell_colours.append(list(colour[:3]))
            box = (column * width // columns, row * height // rows,
                   (column + 1) * width // columns, (row + 1) * height // rows)
            draw.rectangle(box, fill=colour)
    # Coarse dots and a broken diagonal provide features at multiple scales.
    for _ in range(5):
        radius = max(2, int(min(width, height) * rng.uniform(0.045, 0.095)))
        x, y = rng.randrange(width), rng.randrange(height)
        colour = rgb(base_hue + rng.uniform(0.3, 0.7), 0.75, rng.uniform(0.6, 0.98))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
    stripe = max(2, min(width, height) // 18)
    draw.line([(0, height * 3 // 4), (width * 2 // 5, height // 3),
               (width * 3 // 4, height // 2)], fill=(240, 231, 185, 255), width=stripe)

    label, label_box, label_scale = "F{:02d}".format(face_id), None, None
    for scale in (4, 3, 2, 1):
        box = fit_rectangle(interior, 17 * scale + 4, 7 * scale + 4)
        if box is not None:
            label_box, label_scale = box, scale
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=DARK)
            draw_glyphs(draw, label, box[0] + 2, box[1] + 2, scale)
            break

    available = interior.copy()
    if label_box is not None:
        ImageDraw.Draw(available).rectangle(label_box, fill=0)
    arrow_box = None
    for side in (20, 16, 12, 8):
        box = fit_rectangle(available, side, side)
        if box is None:
            continue
        arrow_box = box
        x, y, right, bottom = box
        stroke = max(1, side // 6)
        draw.line([(x + 2, bottom - 3), (x + 2, y + 3), (right - 4, y + 3)],
                  fill=(255, 255, 255, 255), width=stroke)
        draw.polygon([(right - 2, y + 3), (right - 7, y), (right - 7, y + 6)],
                     fill=(255, 74, 57, 255))
        break
    return tile, {"style": "coarse_colour_tiles_dots_asymmetric_markers",
                  "base_colour_rgb": list(base[:3]), "grid": [columns, rows],
                  "cell_colours_rgb": cell_colours, "label": label,
                  "label_bbox_local_pixels": list(label_box) if label_box else None,
                  "label_bitmap_scale": label_scale,
                  "arrow_bbox_local_pixels": list(arrow_box) if arrow_box else None}


def build_v1(seed=DEFAULT_SEED, resolution=1024):
    if not 0 <= seed <= 0xffffffff:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if resolution < 512:
        raise ValueError("resolution must be at least 512 so all 26 faces have an interior")
    vertices, uvs, normals, faces = read_mesh(ROOT / "assets" / MESH_FILE)
    atlas = Image.new("RGBA", (resolution, resolution), DARK)
    occupied = Image.new("L", atlas.size, 0)
    face_records, fingerprints = [], set()
    for face_id, entries in enumerate(faces):
        polygon = [uvs[entry[1]] for entry in entries]
        pixels = [(int(u * (resolution - 1) + 0.5), int((1 - v) * (resolution - 1) + 0.5))
                  for u, v in polygon]
        mask = Image.new("L", atlas.size, 0)
        ImageDraw.Draw(mask).polygon(pixels, fill=255)
        bounds = mask.getbbox()
        local_polygon = mask.crop(bounds)
        # Explicit black padding also erodes UV edges lying on the atlas border.
        padded = Image.new("L", (local_polygon.width + 2 * BORDER,
                                 local_polygon.height + 2 * BORDER), 0)
        padded.paste(local_polygon, (BORDER, BORDER))
        local_mask = padded.filter(ImageFilter.MinFilter(2 * BORDER + 1)).crop(
            (BORDER, BORDER, BORDER + local_polygon.width, BORDER + local_polygon.height))
        interior = Image.new("L", atlas.size, 0)
        interior.paste(local_mask, (bounds[0], bounds[1]))
        if interior.getbbox() is None:
            raise ValueError("Face {} has no interior at this resolution".format(face_id))
        if ImageChops.multiply(occupied, interior).getbbox() is not None:
            raise ValueError("UV interiors overlap at face {}".format(face_id))
        occupied = ImageChops.lighter(occupied, interior)
        material_key = "{}:{}:{}:face:{}".format(GENERATOR_VERSION, seed, MESH_SHA256, face_id)
        face_seed = int.from_bytes(hashlib.sha256(material_key.encode()).digest()[:4], "big")
        tile, pattern = pattern_for_face(face_id, face_seed, local_mask)
        atlas.paste(tile, (bounds[0], bounds[1]), local_mask)
        fingerprint = sha256(tile.tobytes() + local_mask.tobytes())
        if fingerprint in fingerprints:
            raise ValueError("Duplicate face pattern")
        fingerprints.add(fingerprint)
        face_vertices = [vertices[entry[0]] for entry in entries]
        normal = face_normal(face_vertices)
        if sum(a * b for a, b in zip(normal, normals[entries[0][2]])) < 0.99:
            raise ValueError("Face normal does not match the source winding")
        pattern.update({"atlas_bbox_pixels": list(bounds), "common_edge_band_pixels": BORDER,
                        "interior_pixels": sum(local_mask.histogram()[1:]),
                        "pattern_sha256": fingerprint})
        face_records.append({"face_id": face_id,
                             "vertex_indices0": [entry[0] for entry in entries],
                             "uv_indices0": [entry[1] for entry in entries],
                             "vertices_mesh": face_vertices, "uv_polygon": polygon,
                             "normal_object": normal, "face_seed": face_seed,
                             "pattern": pattern})
    stream = io.BytesIO()
    atlas.save(stream, format="PNG", compress_level=9, optimize=False)
    png = stream.getvalue()
    texture_hash = sha256(png)
    manifest = {
        "schema_version": 1, "generator_version": GENERATOR_VERSION,
        "object_name": "set_obj6_block_corner", "seed": seed,
        "resolution": [resolution, resolution], "mesh_file": MESH_FILE,
        "mesh_sha256": MESH_SHA256, "texture_file": "atlas.png",
        "texture_sha256": texture_hash,
        "texture_id": "set_obj6_block_corner_fixed_v1_s{}_r{}_{}".format(seed, resolution, texture_hash[:12]),
        "source_mesh_scale": [0.03, 0.03, 0.03],
        "uv_convention": {"mesh_uv_origin": "bottom_left", "png_origin": "top_left",
                          "rasterization": "x=round(u*(width-1)); y=round((1-v)*(height-1))"},
        "coordinate_frame": {"texture_attachment": "object_local_mesh_UV",
                             "pose_storage": "position_m_and_quaternion_xyzw",
                             "mapping": "One fixed object-local surface appearance, independent of pose, environment, episode and time.",
                             "image_note": "Matching object pose does not fix camera pose, lighting or hand occlusion."},
        "faces": face_records,
    }
    return png, (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def simple_face_palettes(seed, count):
    """Choose separated, high-contrast colour pairs without global RNG state."""
    rng = random.Random("chamfered_simple_palette_v2:{}".format(seed))
    pairs = []
    for red in range(48, 249, 40):
        for green in range(48, 249, 40):
            for blue in range(48, 249, 40):
                colour = (red, green, blue)
                opposite = tuple(296 - component for component in colour)
                if colour < opposite and sum((a - b) ** 2 for a, b in zip(colour, opposite)) >= 200 ** 2:
                    pairs.append((colour, opposite))
    rng.shuffle(pairs)
    selected = [pairs.pop()]
    distance = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b))
    while len(selected) < count:
        # Maximise the minimum RGB distance to every already chosen colour.
        score = lambda pair: min(distance(a, b) for chosen in selected for a in pair for b in chosen)
        index = max(range(len(pairs)), key=lambda candidate: score(pairs[candidate]))
        selected.append(pairs.pop(index))
    rng.shuffle(selected)
    # Spread the most distinct pairs over the six large faces first, instead of
    # allowing several randomly adjacent main faces to look green/purple alike.
    main_ids = (0, 1, 2, 3, 4, 25)
    if count == 26:
        main_pairs = [selected.pop(0)]
        while len(main_pairs) < len(main_ids):
            score = lambda pair: min(distance(a, b) for chosen in main_pairs for a in pair for b in chosen)
            index = max(range(len(selected)), key=lambda candidate: score(selected[candidate]))
            main_pairs.append(selected.pop(index))
        palettes = [None] * count
        for face_id, pair in zip(main_ids, main_pairs):
            palettes[face_id] = pair
        for face_id in range(count):
            if palettes[face_id] is None:
                palettes[face_id] = selected.pop(0)
    else:
        palettes = selected
    return [pair if rng.randrange(2) else pair[::-1] for pair in palettes]


def build_simple(seed=DEFAULT_SEED, resolution=1024):
    if not 0 <= seed <= 0xffffffff:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if resolution < 512:
        raise ValueError("resolution must be at least 512 so all 26 faces have an interior")
    version = "chamfered_uv_atlas_simple_v2"
    vertices, uvs, normals, faces = read_mesh(ROOT / "assets" / MESH_FILE)
    palettes = simple_face_palettes(seed, len(faces))
    all_colours = [colour for palette in palettes for colour in palette]
    if len(set(all_colours)) != 2 * len(faces):
        raise ValueError("Face palettes must be globally disjoint")
    atlas = Image.new("RGBA", (resolution, resolution), palettes[0][0] + (255,))
    occupied = Image.new("L", atlas.size, 0)
    face_records, tiles = [], []
    for face_id, entries in enumerate(faces):
        polygon = [uvs[entry[1]] for entry in entries]
        pixels = [(int(u * (resolution - 1) + 0.5), int((1 - v) * (resolution - 1) + 0.5))
                  for u, v in polygon]
        mask = Image.new("L", atlas.size, 0)
        ImageDraw.Draw(mask).polygon(pixels, fill=255)
        # Rasterised shared UV edges belong to the lowest face ID. No common
        # black edge band is painted over otherwise independently coloured faces.
        owned = ImageChops.subtract(mask, occupied)
        occupied = ImageChops.lighter(occupied, mask)
        bounds = owned.getbbox()
        local_mask = owned.crop(bounds)
        points = [(x, y) for y in range(local_mask.height) for x in range(local_mask.width)
                  if local_mask.getpixel((x, y))]
        material_key = "{}:{}:{}:face:{}".format(version, seed, MESH_SHA256, face_id)
        face_seed = int.from_bytes(hashlib.sha256(material_key.encode()).digest()[:4], "big")
        rng = random.Random(face_seed)
        angle = rng.uniform(0, 2 * math.pi)
        direction = (math.cos(angle), math.sin(angle))
        projections = sorted(x * direction[0] + y * direction[1] for x, y in points)
        fraction = rng.uniform(0.30, 0.40)
        threshold = projections[int((len(projections) - 1) * fraction)]
        tile = Image.new("RGBA", local_mask.size, palettes[face_id][0] + (255,))
        for x in range(tile.width):
            for y in range(tile.height):
                if x * direction[0] + y * direction[1] <= threshold:
                    tile.putpixel((x, y), palettes[face_id][1] + (255,))
        actual_colours = {tile.getpixel(point)[:3] for point in points}
        if actual_colours != set(palettes[face_id]):
            raise ValueError("Face {} must contain exactly two colours".format(face_id))
        atlas.paste(tile, (bounds[0], bounds[1]), local_mask)
        tiles.append((tile, local_mask, bounds))
        face_vertices = [vertices[entry[0]] for entry in entries]
        normal = face_normal(face_vertices)
        if sum(a * b for a, b in zip(normal, normals[entries[0][2]])) < 0.99:
            raise ValueError("Face normal does not match the source winding")
        face_records.append({
            "face_id": face_id,
            "vertex_indices0": [entry[0] for entry in entries],
            "uv_indices0": [entry[1] for entry in entries],
            "vertices_mesh": face_vertices, "uv_polygon": polygon,
            "normal_object": normal, "face_seed": face_seed,
            "pattern": {"style": "two_solid_colours_asymmetric_split",
                        "palette_rgb": [list(colour) for colour in palettes[face_id]],
                        "split_direction_pixels": list(direction),
                        "split_threshold_local_pixels": threshold,
                        "accent_fraction_target": fraction,
                        "atlas_bbox_pixels": list(bounds),
                        "owned_pixels": len(points),
                        "common_edge_band_pixels": 0,
                        "pattern_sha256": sha256(tile.tobytes() + local_mask.tobytes())}})
    # Extend face colours into unused UV area for two texels. The extension
    # never paints over a face or another face's explicitly assigned boundary.
    padding_occupied = occupied.copy()
    for tile, mask, bounds in tiles:
        expanded = Image.new("L", atlas.size, 0)
        expanded.paste(mask, (bounds[0], bounds[1]))
        padding = ImageChops.subtract(expanded.filter(ImageFilter.MaxFilter(5)), padding_occupied)
        padding_occupied = ImageChops.lighter(padding_occupied, padding)
        for y in range(max(0, bounds[1] - 2), min(resolution, bounds[3] + 2)):
            for x in range(max(0, bounds[0] - 2), min(resolution, bounds[2] + 2)):
                if padding.getpixel((x, y)):
                    atlas.putpixel((x, y), tile.getpixel((min(max(x - bounds[0], 0), tile.width - 1),
                                                        min(max(y - bounds[1], 0), tile.height - 1))))
    # Recheck the final image, not just the pre-composite tile.
    for face, (_, mask, bounds) in zip(face_records, tiles):
        colours = {atlas.getpixel((x + bounds[0], y + bounds[1]))[:3]
                   for y in range(mask.height) for x in range(mask.width) if mask.getpixel((x, y))}
        if colours != {tuple(colour) for colour in face["pattern"]["palette_rgb"]}:
            raise ValueError("Atlas compositing changed a face palette")
    stream = io.BytesIO()
    atlas.save(stream, format="PNG", compress_level=9, optimize=False)
    png = stream.getvalue()
    texture_hash = sha256(png)
    manifest = {
        "schema_version": 1, "generator_version": version,
        "object_name": "set_obj6_block_corner", "seed": seed,
        "resolution": [resolution, resolution], "mesh_file": MESH_FILE,
        "mesh_sha256": MESH_SHA256, "texture_file": "atlas.png",
        "texture_sha256": texture_hash,
        "texture_id": "set_obj6_block_corner_simple_v2_s{}_r{}_{}".format(seed, resolution, texture_hash[:12]),
        "source_mesh_scale": [0.03, 0.03, 0.03],
        "palette_contract": {"colours_per_face": 2, "colours_total": len(all_colours),
                             "face_palettes_disjoint": True,
                             "common_edge_colour": False,
                             "sampling_note": "Bilinear texture filtering and lighting can interpolate edge/display colours."},
        "uv_convention": {"mesh_uv_origin": "bottom_left", "png_origin": "top_left",
                          "rasterization": "x=round(u*(width-1)); y=round((1-v)*(height-1))",
                          "shared_edge_pixel_ownership": "lowest face_id first; no common border",
                          "outside_face_padding_pixels": 2},
        "coordinate_frame": {"texture_attachment": "object_local_mesh_UV",
                             "pose_storage": "position_m_and_quaternion_xyzw",
                             "mapping": "One fixed object-local surface appearance, independent of pose, environment, episode and time.",
                             "image_note": "Matching object pose does not fix camera pose, lighting or hand occlusion."},
        "faces": face_records,
    }
    return png, (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def face_adjacency(faces):
    """Use shared source-mesh edges, not neighbouring atlas pixels."""
    edge_faces = {}
    for face_id, entries in enumerate(faces):
        indices = [entry[0] for entry in entries]
        for a, b in zip(indices, indices[1:] + indices[:1]):
            edge_faces.setdefault(tuple(sorted((a, b))), []).append(face_id)
    if any(len(owners) != 2 for owners in edge_faces.values()):
        raise ValueError("The source must be a closed two-manifold mesh")
    edges = sorted(tuple(sorted(owners)) for owners in edge_faces.values())
    adjacent = [[] for _ in faces]
    for a, b in edges:
        adjacent[a].append(b)
        adjacent[b].append(a)
    return edges, [sorted(neighbours) for neighbours in adjacent]


def colour_lab(colour):
    """sRGB to CIELAB using the D65 reference white (for colour assignment)."""
    r, g, b = [component / 255.0 for component in colour]
    r, g, b = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
               for value in (r, g, b)]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    convert = lambda value: (value ** (1.0 / 3.0) if value > (6.0 / 29.0) ** 3
                             else value / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    x, y, z = map(convert, (x, y, z))
    return (116 * y - 16, 500 * (x - y), 200 * (y - z))


def solid_face_colours(seed, count, edges):
    """Choose unique face colours, then maximise the weakest adjacent contrast."""
    rng = random.Random("solid_v3:{}".format(seed))
    candidates = sorted({tuple(int(round(component * 255)) for component in
                               colorsys.hsv_to_rgb(hue / 36.0, saturation, value))
                         for hue in range(36) for saturation in (0.55, 0.75, 0.95)
                         for value in (0.65, 0.82, 1.0)})
    rng.shuffle(candidates)
    labs = {colour: colour_lab(colour) for colour in candidates}
    distance = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b))
    colours = [candidates.pop()]
    while len(colours) < count:
        index = max(range(len(candidates)), key=lambda i: min(
            distance(labs[candidates[i]], labs[colour]) for colour in colours))
        colours.append(candidates.pop(index))
    differences = [[distance(labs[a], labs[b]) for b in colours] for a in colours]
    score = lambda assignment: tuple(sorted(differences[assignment[a]][assignment[b]]
                                            for a, b in edges))
    best = None
    for _ in range(12):
        assignment = list(range(count))
        rng.shuffle(assignment)
        current = score(assignment)
        for _ in range(40):
            move, improved = None, current
            for a in range(count):
                for b in range(a + 1, count):
                    assignment[a], assignment[b] = assignment[b], assignment[a]
                    candidate = score(assignment)
                    if candidate > improved:
                        move, improved = (a, b), candidate
                    assignment[a], assignment[b] = assignment[b], assignment[a]
            if move is None:
                break
            a, b = move
            assignment[a], assignment[b] = assignment[b], assignment[a]
            current = improved
        if best is None or current > best[0]:
            best = (current, assignment[:])
    return [colours[index] for index in best[1]], math.sqrt(best[0][0])


def build_solid(seed=DEFAULT_SEED, resolution=1024, profile="solid_v3"):
    if not 0 <= seed <= 0xffffffff:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if resolution < 512:
        raise ValueError("resolution must be at least 512 so all faces have an interior")
    specification = SOLID_PROFILES[profile]
    mesh_file, mesh_hash = specification["mesh_file"], specification["mesh_sha256"]
    version = specification["generator_version"]
    vertices, uvs, normals, faces = read_mesh(
        ROOT / "assets" / mesh_file, mesh_hash, specification["layout"])
    edges, adjacent = face_adjacency(faces)
    colours, minimum_contrast = solid_face_colours(seed, len(faces), edges)
    if len(set(colours)) != len(faces) or minimum_contrast < 60:
        raise ValueError("Face colours must be unique and adjacent CIELAB contrast at least 60")
    atlas = Image.new("RGBA", (resolution, resolution), colours[0] + (255,))
    occupied = Image.new("L", atlas.size, 0)
    face_records, masks = [], []
    for face_id, entries in enumerate(faces):
        polygon = [uvs[entry[1]] for entry in entries]
        pixels = [(int(u * (resolution - 1) + 0.5), int((1 - v) * (resolution - 1) + 0.5))
                  for u, v in polygon]
        mask = Image.new("L", atlas.size, 0)
        ImageDraw.Draw(mask).polygon(pixels, fill=255)
        # Shared UV-edge texels have one fixed owner; there is no common border.
        owned = ImageChops.subtract(mask, occupied)
        occupied = ImageChops.lighter(occupied, mask)
        if owned.getbbox() is None:
            raise ValueError("Face {} has no owned texture pixels".format(face_id))
        atlas.paste(colours[face_id] + (255,), (0, 0, resolution, resolution), owned)
        masks.append(owned)
        face_vertices = [vertices[entry[0]] for entry in entries]
        normal = face_normal(face_vertices)
        if sum(a * b for a, b in zip(normal, normals[entries[0][2]])) < 0.99:
            raise ValueError("Face normal does not match the source winding")
        material_key = "{}:{}:{}:face:{}".format(version, seed, mesh_hash, face_id)
        face_records.append({
            "face_id": face_id, "vertex_indices0": [entry[0] for entry in entries],
            "uv_indices0": [entry[1] for entry in entries],
            "vertices_mesh": face_vertices, "uv_polygon": polygon,
            "normal_object": normal,
            "face_seed": int.from_bytes(hashlib.sha256(material_key.encode()).digest()[:4], "big"),
            "adjacent_face_ids": adjacent[face_id],
            "pattern": {"style": "single_solid_colour", "colour_rgb": list(colours[face_id]),
                        "palette_rgb": [list(colours[face_id])],
                        "atlas_bbox_pixels": list(owned.getbbox()),
                        "owned_pixels": owned.histogram()[255], "common_edge_band_pixels": 0}})
    padding_occupied = occupied.copy()
    for colour, mask in zip(colours, masks):
        padding = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), padding_occupied)
        padding_occupied = ImageChops.lighter(padding_occupied, padding)
        atlas.paste(colour + (255,), (0, 0, resolution, resolution), padding)
    for face, mask in zip(face_records, masks):
        expected = tuple(face["pattern"]["colour_rgb"])
        bounds = mask.getbbox()
        values = {atlas.getpixel((x, y))[:3]
                  for y in range(bounds[1], bounds[3]) for x in range(bounds[0], bounds[2])
                  if mask.getpixel((x, y))}
        if values != {expected}:
            raise ValueError("Face {} is not a single colour".format(face["face_id"]))
    stream = io.BytesIO()
    atlas.save(stream, format="PNG", compress_level=9, optimize=False)
    png = stream.getvalue()
    texture_hash = sha256(png)
    manifest = {
        "schema_version": 1, "generator_version": version,
        "object_name": specification["object_name"], "seed": seed,
        "resolution": [resolution, resolution], "mesh_file": mesh_file,
        "mesh_sha256": mesh_hash, "texture_file": "atlas.png",
        "texture_sha256": texture_hash,
        "texture_id": "{}_s{}_r{}_{}".format(
            specification["texture_prefix"], seed, resolution, texture_hash[:12]),
        "source_mesh_scale": [0.03, 0.03, 0.03],
        "palette_contract": {"colours_per_face": 1, "colours_total": len(colours),
                             "face_palettes_disjoint": True, "common_edge_colour": False,
                             "adjacency": "source OBJ faces sharing a mesh edge",
                             "adjacent_face_pairs": [list(edge) for edge in edges],
                             "contrast_metric": "CIE76 Euclidean distance in CIELAB (sRGB D65)",
                             "minimum_adjacent_delta_e76": minimum_contrast,
                             "sampling_note": "Texture filtering and lighting can interpolate edge/display colours."},
        "uv_convention": {"mesh_uv_origin": "bottom_left", "png_origin": "top_left",
                          "rasterization": "x=round(u*(width-1)); y=round((1-v)*(height-1))",
                          "shared_edge_pixel_ownership": "lowest face_id first; no common border",
                          "outside_face_padding_pixels": 2},
        "coordinate_frame": {"texture_attachment": "object_local_mesh_UV",
                             "pose_storage": "position_m_and_quaternion_xyzw",
                             "mapping": "One fixed object-local surface appearance, independent of pose, environment, episode and time.",
                             "image_note": "Matching object pose does not fix camera pose, lighting or hand occlusion."},
        "faces": face_records}
    return png, (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def build(seed=DEFAULT_SEED, resolution=1024, profile="solid_v3"):
    if profile in SOLID_PROFILES:
        return build_solid(seed, resolution, profile)
    if profile == "simple_v2":
        return build_simple(seed, resolution)
    if profile == "fixed_v1":
        return build_v1(seed, resolution)
    raise ValueError("Unknown texture profile: {}".format(profile))


def write_outputs(output, png, manifest):
    output = Path(output)
    generated = {"atlas.png": png, "manifest.json": manifest}
    for name, data in generated.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError("{} differs; select a new --output-dir to preserve the fixed texture".format(path))
    output.mkdir(parents=True, exist_ok=True)
    for name, data in generated.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--profile", choices=("solid_v3", "simple_v2", "fixed_v1", "cube_solid_v1"),
                        default="solid_v3")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    directory = "cube_solid_v1" if args.profile == "cube_solid_v1" else "chamfered_block_" + args.profile
    output_dir = args.output_dir or ROOT / "assets/textures" / directory
    png, manifest_bytes = build(args.seed, args.resolution, args.profile)
    write_outputs(output_dir, png, manifest_bytes)
    manifest = json.loads(manifest_bytes)
    print(json.dumps({"output_dir": str(output_dir), "texture_id": manifest["texture_id"],
                      "faces": len(manifest["faces"]), "texture_sha256": manifest["texture_sha256"]}))


if __name__ == "__main__":
    main()
