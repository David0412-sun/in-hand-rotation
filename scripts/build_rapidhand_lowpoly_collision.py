#!/usr/bin/env python3
"""Build an inscribed, collision-only RapidHand asset using NumPy and SciPy.

Run with the robosyn Python interpreter. ``--check`` recomputes all geometry and
checks the committed/generated outputs without changing any files. Coordinates
stay in the source STL frame, in meters. Only ten collision mesh paths change;
visuals, links, joints, inertials, tactile pads, cameras and existing boxes stay
as in vision_hand_simple_collision.urdf.

This approximates each source mesh's *convex hull*, matching the existing
single-convex collision import, rather than its detailed/nonconvex visual mesh.
It does not reduce rigid-body count or guarantee a GPU-memory improvement.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets/rapidhand"
SOURCE_URDF = ASSET_DIR / "urdf/vision_hand_simple_collision.urdf"
OUTPUT_URDF = ASSET_DIR / "urdf/vision_hand_lowpoly_collision.urdf"
MESH_DIR = ASSET_DIR / "meshes/collision_lowpoly"
REPORT_PATH = ASSET_DIR / "collision_lowpoly_report.json"
TARGET_LINKS = ("BASE", "TPIP", "IPIP", "IDIP", "MPIP", "MDIP",
                "RPIP", "RDIP", "LPIP", "LDIP")
PREFERRED_ERROR_M = 0.0005
MAX_ERROR_M = 0.001
HARD_VERTEX_LIMIT = 64
STL_DTYPE = np.dtype([
    ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")
])


def read_stl(path):
    """Read the source asset's binary STL without an extra mesh dependency."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("Invalid binary STL: {}".format(path))
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError("Expected binary STL: {}".format(path))
    triangles = np.frombuffer(data, dtype=STL_DTYPE, offset=84,
                              count=count)["vertices"].astype(np.float64)
    if not np.isfinite(triangles).all():
        raise ValueError("Non-finite STL coordinates: {}".format(path))
    return triangles, hashlib.sha256(data).hexdigest()


def distance_to_convex_hull(points, vertices, hull):
    """Euclidean point-to-polyhedron distance, including edges and vertices.

    A plane violation alone underestimates distance at corners. Here every
    triangle's interior projection and its three edge projections are checked;
    points inside the convex hull have distance zero.
    """
    triangles = vertices[hull.simplices]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normals = hull.equations[:, :3]
    p = points[:, None, :]
    pa = p - a
    signed = np.einsum("nfi,fi->nf", pa, normals)
    projected = pa - signed[..., None] * normals
    d00 = np.einsum("fi,fi->f", ab, ab)
    d01 = np.einsum("fi,fi->f", ab, ac)
    d11 = np.einsum("fi,fi->f", ac, ac)
    d20 = np.einsum("nfi,fi->nf", projected, ab)
    d21 = np.einsum("nfi,fi->nf", projected, ac)
    denominator = d00 * d11 - d01 * d01
    if np.any(denominator <= 0):
        raise ValueError("Degenerate hull face")
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    inside_triangle = (v >= -1e-10) & (w >= -1e-10) & (v + w <= 1 + 1e-10)
    distances_squared = np.where(inside_triangle, signed * signed, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        delta = p - start
        alpha = np.clip(np.einsum("nfi,fi->nf", delta, edge)
                        / np.einsum("fi,fi->f", edge, edge), 0, 1)
        residual = delta - alpha[..., None] * edge
        distances_squared = np.minimum(
            distances_squared, np.einsum("nfi,nfi->nf", residual, residual))
    distances = np.sqrt(distances_squared.min(axis=1))
    inside = np.all(points @ normals.T + hull.equations[:, 3] <= 1e-12, axis=1)
    distances[inside] = 0
    return distances


def simplify(source_vertices):
    """Deterministic farthest-point refinement of an inscribed convex hull."""
    directions = np.asarray([direction for direction in
                             itertools.product((-1, 0, 1), repeat=3)
                             if any(direction)], dtype=float)
    # Include every axis extremum plus diagonal support points. Bounding boxes
    # and the existing tactile probe's parent-center normal convention survive.
    selected = sorted(set(np.argmax(source_vertices @ directions.T, axis=0)))
    while True:
        candidate = source_vertices[selected]
        hull = ConvexHull(candidate)
        distances = distance_to_convex_hull(source_vertices, candidate, hull)
        worst = int(np.argmax(distances))
        if distances[worst] <= PREFERRED_ERROR_M:
            return candidate, hull, float(distances[worst])
        if len(selected) >= HARD_VERTEX_LIMIT:
            if distances[worst] <= MAX_ERROR_M:
                return candidate, hull, float(distances[worst])
            raise ValueError("1.0 mm error bound requires more than 64 vertices")
        if worst in selected:
            raise ValueError("Hull refinement stopped making progress")
        selected.append(worst)


def encode_stl(vertices, hull):
    """Emit closed convex triangles with explicit outward winding/normals."""
    faces = []
    for face, equation in zip(hull.simplices, hull.equations):
        face = face.tolist()
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), equation[:3]) < 0:
            face[1], face[2] = face[2], face[1]
        first = face.index(min(face))
        faces.append(tuple(face[first:] + face[:first]))
    faces = np.asarray(sorted(faces), dtype=int)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-14):
        raise ValueError("Degenerate output triangle")
    edge_counts = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(a), int(b))))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    if not all(count == 2 for count in edge_counts.values()):
        raise ValueError("Output is not a closed two-manifold")
    signed_volume = np.einsum("fi,fi->f", triangles[:, 0],
                             np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    if not np.isclose(signed_volume, hull.volume, rtol=1e-10, atol=1e-15):
        raise ValueError("Output winding/volume check failed")
    records = np.zeros(len(faces), dtype=STL_DTYPE)
    records["normal"] = normals / lengths[:, None]
    records["vertices"] = triangles
    header = b"RapidHand inscribed convex collision; source coordinates in meters"
    data = header.ljust(80, b"\0") + struct.pack("<I", len(faces)) + records.tobytes()
    return data, len(faces)


def max_coplanar_polygon_vertices(hull):
    """Conservative polygon size after merging coplanar triangular faces.

    Count all triangle vertices in each near-coplanar face group (including any
    collinear edge vertices); a cooked polygon cannot need more than this count.
    """
    groups = []
    for face, equation in zip(hull.simplices, hull.equations):
        for reference, indices in groups:
            if (np.linalg.norm(equation[:3] - reference[:3]) <= 1e-6
                    and abs(equation[3] - reference[3]) <= 1e-8):
                indices.update(face.tolist())
                break
        else:
            groups.append((equation, set(face.tolist())))
    return max(len(indices) for _, indices in groups)


def replace_collision_filename(text, name, source_filename, target_filename):
    pattern = re.compile(r'(<link name="' + re.escape(name) + r'">)(.*?)(</link>)',
                         re.DOTALL)
    match = pattern.search(text)
    if match is None:
        raise ValueError("Missing link: " + name)
    body = match.group(2)
    collision = re.search(r"<collision>.*?</collision>", body, re.DOTALL)
    if collision is None or collision.group().count(source_filename) != 1:
        raise ValueError("Unexpected collision structure: " + name)
    new_collision = collision.group().replace(source_filename, target_filename)
    body = body[:collision.start()] + new_collision + body[collision.end():]
    return text[:match.start(2)] + body + text[match.end(2):]


def build():
    source_text = SOURCE_URDF.read_text()
    source = ET.fromstring(source_text)
    output_text = source_text
    outputs = {}
    report = {
        "source_urdf": str(SOURCE_URDF.relative_to(ROOT)),
        "source_urdf_sha256": hashlib.sha256(SOURCE_URDF.read_bytes()).hexdigest(),
        "output_urdf": str(OUTPUT_URDF.relative_to(ROOT)),
        "algorithm": "Source-convex-hull vertex subset, 26-direction seed, greedy Euclidean farthest point",
        "max_inward_error_mm": MAX_ERROR_M * 1000,
        "preferred_inward_error_mm": PREFERRED_ERROR_M * 1000,
        "hard_vertex_limit": HARD_VERTEX_LIMIT,
        "hard_coplanar_polygon_vertex_limit": 32,
        "gpu_note": "Geometry satisfies the <=64 hull vertices and <=32 polygon vertices limits; GPU cooking/runtime and memory savings require separate measurement",
        "error_definition": "Maximum Euclidean distance from source hull vertices to candidate solid; bounds the whole source convex hull by convexity of distance",
        "containment_definition": "Candidate vertices are exact float32 source hull vertices; no outward expansion beyond source convex hull",
        "preserved": {
            "links": 41, "joints": 40, "box_collisions": 11,
            "tactile_mesh_collisions": 15, "camera_mesh_collisions": 5,
            "all_visuals_inertials_joint_origins_and_mesh_scales": True,
        },
        "links": [],
    }
    # Four fingers share identical STL geometry. Avoid repeating expensive hull
    # refinement while keeping a separate, readily inspectable asset per link.
    geometry_cache = {}
    for name in TARGET_LINKS:
        mesh = source.find("link[@name='{}']/collision/geometry/mesh".format(name))
        if mesh is None or mesh.attrib.get("scale", "1 1 1") != "1 1 1":
            raise ValueError("Expected unscaled source mesh for " + name)
        source_filename = mesh.attrib["filename"]
        source_path = (SOURCE_URDF.parent / source_filename).resolve()
        triangles, source_hash = read_stl(source_path)
        if source_hash not in geometry_cache:
            unique = np.unique(triangles.reshape(-1, 3), axis=0)
            source_hull = ConvexHull(unique)
            hull_vertices = unique[source_hull.vertices]
            candidate, hull, error = simplify(hull_vertices)
            data, face_count = encode_stl(candidate, hull)
            polygon_vertices = max_coplanar_polygon_vertices(hull)
            if polygon_vertices > 32:
                raise ValueError("Convex polygon exceeds 32 vertices")
            # STL float32 conversion must preserve the exact source subset.
            if not np.array_equal(candidate.astype(np.float32).astype(float), candidate):
                raise ValueError("Export would round source coordinates")
            violation = max(0.0, float(np.max(
                candidate @ source_hull.equations[:, :3].T + source_hull.equations[:, 3])))
            if violation > 1e-10:
                raise ValueError("Candidate extends outside the source hull")
            if not np.array_equal(np.min(candidate, axis=0), np.min(unique, axis=0)) or not np.array_equal(
                    np.max(candidate, axis=0), np.max(unique, axis=0)):
                raise ValueError("Axis extrema changed")
            stats = {
                "source_triangles": len(triangles),
                "source_unique_vertices": len(unique),
                "source_convex_hull_vertices": len(source_hull.vertices),
                "candidate_vertices": len(hull.vertices),
                "candidate_triangles": face_count,
                "max_coplanar_polygon_vertices": polygon_vertices,
                "max_euclidean_inward_error_mm": error * 1000,
                "max_outward_plane_violation_mm": violation * 1000,
                "source_hull_volume_m3": float(source_hull.volume),
                "candidate_volume_m3": float(hull.volume),
                "volume_ratio": float(hull.volume / source_hull.volume),
                "bounds_min_m": np.min(candidate, axis=0).tolist(),
                "bounds_max_m": np.max(candidate, axis=0).tolist(),
                "axis_extrema_preserved": True,
                "preferred_0_5_mm_error_met": error <= PREFERRED_ERROR_M,
                "error_budget_note": ("64-vertex cap reached; retain the measured error below the hard 1.0 mm bound"
                                      if error > PREFERRED_ERROR_M else None),
                "closed_outward_convex_mesh": True,
            }
            geometry_cache[source_hash] = data, stats
        data, stats = geometry_cache[source_hash]
        mesh_path = MESH_DIR / (name + ".stl")
        outputs[mesh_path] = data
        target_filename = "../meshes/collision_lowpoly/" + mesh_path.name
        output_text = replace_collision_filename(output_text, name, source_filename, target_filename)
        report["links"].append(dict(stats, name=name,
                                    source_mesh=str(source_path.relative_to(ROOT)),
                                    source_mesh_sha256=source_hash,
                                    candidate_mesh=str(mesh_path.relative_to(ROOT)),
                                    candidate_mesh_sha256=hashlib.sha256(data).hexdigest()))
    output_text = re.sub(
        r"<!-- Collision-only variant\..*?-->",
        "<!-- Low-vertex collision-only variant, generated by scripts/build_rapidhand_lowpoly_collision.py.\n"
        "       BASE, TPIP and four finger PIP/DIP collisions use inscribed convex meshes (<=64 vertices, <=1 mm inward).\n"
        "       All visuals, inertials, joints, 11 boxes, 15 tactile meshes and 5 camera meshes are preserved. -->",
        output_text, count=1, flags=re.DOTALL)
    # Undo the ten intended path substitutions in the parsed output. Every XML
    # element/attribute (including collision transforms) must then be identical.
    verification = ET.fromstring(output_text)
    for name in TARGET_LINKS:
        location = "link[@name='{}']/collision/geometry/mesh".format(name)
        verification.find(location).attrib = dict(source.find(location).attrib)
    if ET.tostring(verification) != ET.tostring(source):
        raise ValueError("Unexpected change outside target collision mesh filenames")
    outputs[OUTPUT_URDF] = output_text.encode()
    outputs[REPORT_PATH] = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    return outputs, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Recompute and verify output bytes; do not write files")
    args = parser.parse_args()
    outputs, report = build()
    for path, data in outputs.items():
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit("Generated file differs/missing: {}".format(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    for link in report["links"]:
        print("{name}: {source_convex_hull_vertices} -> {candidate_vertices} vertices, "
              "max inward {max_euclidean_inward_error_mm:.4f} mm, "
              "volume ratio {volume_ratio:.4f}".format(**link))
    print("{} {} generated files; original assets unchanged".format(
        "Verified" if args.check else "Wrote", len(outputs)))


if __name__ == "__main__":
    main()
