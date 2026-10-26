#!/usr/bin/env python3
"""Physically probe every RapidHand tactile link in an isolated Isaac Gym sim.

Run with the project's Isaac Gym environment. No project task is imported. The
default uses GPU PhysX and two environments; only environment zero is touched.
This checks link-level net contact forces, not pressure images or hardware
calibration. A nonzero exit status means at least one check failed.
"""

import argparse
import json
import math
from pathlib import Path
import struct
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np


def origin_transform(origin):
    transform = np.eye(4)
    if origin is None:
        return transform
    roll, pitch, yaw = map(float, origin.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
        math.cos(yaw), math.sin(yaw),
    )
    transform[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    transform[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
    return transform


def stl_vertices(path):
    data = path.read_bytes()
    if len(data) >= 84:
        triangles = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * triangles:
            dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)),
                              ("attribute", "<u2")])
            return np.frombuffer(data[84:], dtype=dtype)["vertices"].reshape(-1, 3).astype(float)
    vertices = []
    for line in data.decode("ascii").splitlines():
        fields = line.split()
        if fields and fields[0] == "vertex":
            vertices.append(list(map(float, fields[1:4])))
    if not vertices:
        raise ValueError("No STL vertices in {}".format(path))
    return np.asarray(vertices)


def collision_vertices(link, urdf_path):
    """Collect mesh vertices and box corners in the owning link's frame."""
    collisions = link.findall("collision")
    if not collisions:
        raise ValueError("{} has no collision".format(link.get("name")))
    vertices_in_link = []
    for collision in collisions:
        mesh = collision.find("geometry/mesh")
        box = collision.find("geometry/box")
        if mesh is not None:
            vertices = stl_vertices((urdf_path.parent / mesh.get("filename")).resolve())
            vertices *= np.asarray(list(map(float, mesh.get("scale", "1 1 1").split())))
        elif box is not None:
            half_size = np.asarray(list(map(float, box.get("size").split()))) / 2
            vertices = np.asarray([[x, y, z] for x in (-1, 1)
                                   for y in (-1, 1) for z in (-1, 1)]) * half_size
        else:
            raise ValueError("Expected a collision mesh or box on {}".format(link.get("name")))
        transform = origin_transform(collision.find("origin"))
        vertices_in_link.append(vertices @ transform[:3, :3].T + transform[:3, 3])
    return np.concatenate(vertices_in_link)


def inspect_pads(urdf_path):
    """Find a center on each thin pad's outward face, in its own link frame."""
    robot = ET.parse(str(urdf_path)).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
    roots = set(links) - set(joints)
    if len(roots) != 1:
        raise ValueError("Expected one URDF root")
    fk = {next(iter(roots)): np.eye(4)}
    pending = dict(joints)
    while pending:
        progressed = False
        for child, joint in list(pending.items()):
            parent = joint.find("parent").get("link")
            if parent in fk:
                # All moving joints are tested at zero, so origin is their FK.
                fk[child] = fk[parent] @ origin_transform(joint.find("origin"))
                del pending[child]
                progressed = True
        if not progressed:
            raise ValueError("URDF joint tree is disconnected or cyclic")
    pads = []
    for name, link in links.items():
        if "tactile" not in name.lower():
            continue
        joint = joints[name]
        if joint.get("type") != "fixed":
            raise ValueError("Tactile link {} is not attached by a fixed joint".format(name))
        parent_name = joint.find("parent").get("link")
        vertices = collision_vertices(link, urdf_path)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        thin_axis = int(np.argmin(high - low))
        normal = np.eye(3)[thin_axis]
        center = (low + high) / 2
        parent_vertices = collision_vertices(links[parent_name], urdf_path)
        parent_center = (parent_vertices.min(axis=0) + parent_vertices.max(axis=0)) / 2
        joint_transform = origin_transform(joint.find("origin"))
        center_in_parent = joint_transform[:3, :3] @ center + joint_transform[:3, 3]
        normal_in_parent = joint_transform[:3, :3] @ normal
        sign = 1 if np.dot(normal_in_parent, center_in_parent - parent_center) >= 0 else -1
        normal *= sign
        surface = center.copy()
        surface[thin_axis] = high[thin_axis] if sign > 0 else low[thin_axis]
        pads.append({
            "name": name,
            "parent": parent_name,
            "joint": joint.get("name"),
            "surface_local": surface.tolist(),
            "normal_local": normal.tolist(),
            "zero_pose_xyz": fk[name][:3, 3].tolist(),
            "pad_extents": (high - low).tolist(),
        })
    if not pads:
        raise ValueError("No links containing 'tactile' in {}".format(urdf_path))
    return pads, len(links), [j.get("name") for j in joints.values() if j.get("type") != "fixed"]


def run_check(args):
    # Isaac Gym must be imported before torch, including when used as a module.
    from isaacgym import gymapi, gymtorch
    import torch

    urdf_path = args.asset_root / args.asset_file
    pads, expected_bodies, expected_dofs = inspect_pads(urdf_path)
    if args.pads and set(args.pads) - {pad["name"] for pad in pads}:
        raise ValueError("Unknown pads: {}".format(sorted(set(args.pads) - {pad["name"] for pad in pads})))
    requested = set(args.pads or [pad["name"] for pad in pads])
    gym = gymapi.acquire_gym()
    sim = viewer = None
    report = {
        "asset": str(urdf_path.resolve()),
        "measurement": "per-link net contact force; no contact-pair or pressure-image claim",
        "settings": vars(args).copy(),
        "expected_tactile_links": [pad["name"] for pad in pads],
        "tested_tactile_links": [pad["name"] for pad in pads if pad["name"] in requested],
        "pads": [],
    }
    report["settings"] = {key: str(value) if isinstance(value, Path) else value
                          for key, value in report["settings"].items()}
    try:
        params = gymapi.SimParams()
        params.dt = args.dt
        params.substeps = 2
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, 0)
        params.use_gpu_pipeline = not args.cpu
        params.physx.use_gpu = not args.cpu
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 12
        params.physx.num_velocity_iterations = 2
        params.physx.contact_offset = args.contact_offset
        params.physx.rest_offset = 0.0
        params.physx.max_depenetration_velocity = 0.5
        params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
        sim = gym.create_sim(args.device, -1 if args.headless else args.device,
                             gymapi.SIM_PHYSX, params)
        if sim is None:
            raise RuntimeError("Failed to create PhysX simulation")
        hand_options = gymapi.AssetOptions()
        hand_options.fix_base_link = True
        hand_options.disable_gravity = True
        hand_options.collapse_fixed_joints = False
        hand_options.flip_visual_attachments = False
        hand_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        hand_options.armature = 0.001
        hand_options.thickness = min(args.contact_offset, 0.0001)
        hand_asset = gym.load_asset(sim, str(args.asset_root), args.asset_file, hand_options)
        if hand_asset is None:
            raise RuntimeError("Failed to load hand")
        body_names = list(gym.get_asset_rigid_body_names(hand_asset))
        imported_pads = {name for name in body_names if "tactile" in name.lower()}
        if imported_pads != set(report["expected_tactile_links"]):
            raise RuntimeError("Imported tactile links differ from URDF: {}".format(body_names))
        if len(body_names) != expected_bodies:
            raise RuntimeError("Fixed links were lost: expected {}, got {}".format(expected_bodies, len(body_names)))
        dof_names = list(gym.get_asset_dof_names(hand_asset))
        if set(dof_names) != set(expected_dofs):
            raise RuntimeError("Imported DOFs differ from URDF")
        dof_properties = gym.get_asset_dof_properties(hand_asset)
        if np.any(dof_properties["lower"] > 0) or np.any(dof_properties["upper"] < 0):
            raise ValueError("This zero-pose probe requires zero to be inside every joint limit")
        dof_properties["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_properties["stiffness"].fill(40.0)
        dof_properties["damping"].fill(2.0)
        sphere_options = gymapi.AssetOptions()
        sphere_options.disable_gravity = True
        sphere_options.thickness = 0.0
        sphere_asset = gym.create_sphere(sim, args.radius, sphere_options)
        envs, hands, probes, probe_indices, pad_indices = [], [], [], [], []
        probe_body_indices, parent_indices, hand_body_indices = [], [], []
        for env_index in range(args.num_envs):
            env = gym.create_env(sim, gymapi.Vec3(-0.5, -0.5, 0),
                                 gymapi.Vec3(0.5, 0.5, 0.8), int(math.ceil(math.sqrt(args.num_envs))))
            hand_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0.3))
            hand = gym.create_actor(env, hand_asset, hand_pose, "rapidhand", env_index, 1)
            probe_pose = gymapi.Transform(gymapi.Vec3(0.4, 0.4, 0.7))
            probe = gym.create_actor(env, sphere_asset, probe_pose, "probe", env_index, 0)
            if hand < 0 or probe < 0:
                raise RuntimeError("Failed to create actors")
            gym.set_actor_dof_properties(env, hand, dof_properties)
            properties = gym.get_actor_rigid_body_properties(env, probe)
            properties[0].mass = args.probe_mass
            gym.set_actor_rigid_body_properties(env, probe, properties, True)
            probe_indices.append(gym.get_actor_index(env, probe, gymapi.DOMAIN_SIM))
            probe_body_indices.append(gym.get_actor_rigid_body_index(env, probe, 0, gymapi.DOMAIN_SIM))
            indices = [gym.find_actor_rigid_body_index(env, hand, pad["name"], gymapi.DOMAIN_SIM)
                       for pad in pads]
            if min(indices) < 0:
                raise RuntimeError("A tactile rigid body index could not be resolved")
            pad_indices.append(indices)
            parent_indices.append([
                gym.find_actor_rigid_body_index(env, hand, pad["parent"], gymapi.DOMAIN_SIM)
                for pad in pads
            ])
            hand_body_indices.append([
                gym.find_actor_rigid_body_index(env, hand, name, gymapi.DOMAIN_SIM)
                for name in body_names
            ])
            envs.append(env)
            hands.append(hand)
            probes.append(probe)
        if len(set(index for row in pad_indices for index in row)) != args.num_envs * len(pads):
            raise RuntimeError("Tactile global indices overlap between environments")
        gym.prepare_sim(sim)
        roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1, 13)
        bodies = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(-1, 13)
        contact = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(-1, 3)
        dofs = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(args.num_envs, len(dof_names), 2)
        tensor_device = contact.device
        pad_ids = torch.tensor(pad_indices, device=tensor_device, dtype=torch.long)
        probe_ids = torch.tensor(probe_indices, device=tensor_device, dtype=torch.long)
        probe_ids_i32 = probe_ids.to(torch.int32)
        probe_body_ids = torch.tensor(probe_body_indices, device=tensor_device, dtype=torch.long)
        parent_ids = torch.tensor(parent_indices, device=tensor_device, dtype=torch.long)
        hand_body_ids = torch.tensor(hand_body_indices, device=tensor_device, dtype=torch.long)
        applied_forces = torch.zeros_like(contact)
        targets = torch.zeros((args.num_envs, len(dof_names)), device=tensor_device)
        gym.refresh_actor_root_state_tensor(sim)
        parked = roots[probe_ids, :3].clone()
        if not args.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError("Failed to create viewer")
            gym.viewer_camera_look_at(viewer, envs[0], gymapi.Vec3(0.5, -0.5, 0.6), gymapi.Vec3(0, 0, 0.45))

        def refresh():
            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_net_contact_force_tensor(sim)
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)

        def park_probes(active_position=None):
            roots[probe_ids, :3] = parked
            roots[probe_ids, 3:7] = roots.new_tensor([0, 0, 0, 1])
            roots[probe_ids, 7:13] = 0
            if active_position is not None:
                roots[probe_ids[0], :3] = active_position
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(roots),
                                                    gymtorch.unwrap_tensor(probe_ids_i32), len(probe_indices))

        def step(probe_target=None):
            # A real dynamic probe is pushed by forces. Teleporting and resetting
            # its velocity every step produces impact-only signals, not a hold.
            applied_forces.zero_()
            if probe_target is not None:
                position = roots[probe_ids[0], :3]
                velocity = roots[probe_ids[0], 7:10]
                force = args.probe_stiffness * (probe_target - position) - args.probe_damping * velocity
                force *= torch.clamp(args.max_probe_force / torch.linalg.vector_norm(force).clamp_min(1e-12), max=1)
                applied_forces[probe_body_ids[0]] = force
            gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(applied_forces), None,
                                              gymapi.ENV_SPACE)
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(targets))
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            refresh()
            if viewer is not None:
                if gym.query_viewer_has_closed(viewer):
                    raise RuntimeError("Viewer closed before completion")
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
            values = contact[pad_ids].clone()
            if not torch.isfinite(values).all().item():
                raise RuntimeError("Non-finite tactile contact force")
            return values

        def reset():
            dofs.zero_()
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dofs))
            park_probes()
            for _ in range(args.settle_steps):
                step()

        def rotate(quaternion, vector):
            cross = torch.cross(quaternion[:3], vector, dim=0)
            return vector + 2 * (quaternion[3] * cross + torch.cross(quaternion[:3], cross, dim=0))

        reset()
        report["actual_body_names"] = body_names
        report["actual_dof_names"] = dof_names
        report["tactile_global_indices"] = pad_indices
        report["tensor_device"] = str(tensor_device)
        fk_errors = []
        for pad_index, pad in enumerate(pads):
            actual = bodies[pad_ids[0, pad_index], :3].cpu().numpy()
            expected = np.asarray(pad["zero_pose_xyz"]) + np.array([0, 0, 0.3])
            fk_errors.append(float(np.linalg.norm(actual - expected)))
        report["zero_pose_fk_position_errors"] = fk_errors
        if max(fk_errors) > 0.002:
            raise RuntimeError("Imported body frames differ from zero-pose URDF FK by over 2 mm")
        for pad_index, pad in enumerate(pads):
            if pad["name"] not in requested:
                continue
            result = dict(pad, attempts=[])
            surface = torch.tensor(pad["surface_local"], dtype=torch.float32, device=tensor_device)
            normal = torch.tensor(pad["normal_local"], dtype=torch.float32, device=tensor_device)
            for depth in args.depths:
                reset()
                records = []

                def pad_surface():
                    body = bodies[pad_ids[0, pad_index]]
                    return body[:3] + rotate(body[3:7], surface), rotate(body[3:7], normal)

                def record(phase, gap=None):
                    position = None
                    if gap is not None:
                        point, direction = pad_surface()
                        position = point + direction * (args.radius + gap)
                    forces = step(position)
                    norms = torch.linalg.vector_norm(forces, dim=-1)
                    own = norms[0].clone()
                    own[pad_index] = 0
                    point, direction = pad_surface()
                    actual_probe = roots[probe_ids[0], :3]
                    parent_force = contact[parent_ids[0, pad_index]]
                    hand_norms = torch.linalg.vector_norm(contact[hand_body_ids[0]], dim=-1)
                    hand_max_index = int(hand_norms.argmax().item())
                    records.append({
                        "phase": phase,
                        "gap_m": gap,
                        "target_force": forces[0, pad_index].cpu().tolist(),
                        "target_norm": float(norms[0, pad_index].item()),
                        "other_pad_max": float(own.max().item()),
                        "other_env_max": float(norms[1:].max().item()),
                        "probe_position": actual_probe.cpu().tolist(),
                        "probe_velocity": roots[probe_ids[0], 7:10].cpu().tolist(),
                        "probe_contact_force": contact[probe_body_ids[0]].cpu().tolist(),
                        "applied_probe_force": applied_forces[probe_body_ids[0]].cpu().tolist(),
                        "pad_surface_position": point.cpu().tolist(),
                        "pad_surface_normal": direction.cpu().tolist(),
                        "actual_gap_m": float((torch.dot(actual_probe - point, direction) - args.radius).item()),
                        "parent_force": parent_force.cpu().tolist(),
                        "parent_norm": float(torch.linalg.vector_norm(parent_force).item()),
                        "all_hand_max_norm": float(hand_norms[hand_max_index].item()),
                        "all_hand_max_body": body_names[hand_max_index],
                    })

                for _ in range(args.baseline_steps):
                    record("baseline")
                point, direction = pad_surface()
                park_probes(point + direction * (args.radius + args.clearance))
                for gap in np.linspace(args.clearance, -depth, args.ramp_steps):
                    record("approach", float(gap))
                for _ in range(args.hold_steps):
                    record("hold", -depth)
                for gap in np.linspace(-depth, args.clearance, args.ramp_steps):
                    record("retreat", float(gap))
                park_probes()
                for _ in range(args.settle_steps):
                    record("released")
                baseline = max(row["target_norm"] for row in records if row["phase"] == "baseline")
                peak = max(row["target_norm"] for row in records if row["phase"] == "hold")
                released = max(row["target_norm"] for row in records[-args.off_samples:])
                control = max(row["other_env_max"] for row in records)
                force_floor = max(args.min_force, baseline * 5)
                release_limit = max(args.off_absolute, peak * args.off_ratio)
                passed = peak > force_floor and released < release_limit and control <= args.crosstalk_tolerance
                hold_norms = [row["target_norm"] for row in records if row["phase"] == "hold"]
                result["attempts"].append({
                    "depth_m": depth, "baseline_max": baseline, "hold_peak": peak,
                    "released_tail_max": released, "other_environment_max": control,
                    "required_force": force_floor, "release_limit": release_limit,
                    "all_phases_target_peak": max(row["target_norm"] for row in records),
                    "all_phases_parent_peak": max(row["parent_norm"] for row in records),
                    "hold_above_threshold_fraction": sum(value > force_floor for value in hold_norms) / len(hold_norms),
                    "hold_tail_mean": sum(hold_norms[-args.off_samples:]) / len(hold_norms[-args.off_samples:]),
                    "passed": passed, "samples": records,
                })
                if passed:
                    break
            result["passed"] = result["attempts"][-1]["passed"]
            report["pads"].append(result)
            attempt = result["attempts"][-1]
            print("{}: {} peak={:.6g}, released={:.6g}, other_env={:.6g}".format(
                pad["name"], "PASS" if result["passed"] else "FAIL", attempt["hold_peak"],
                attempt["released_tail_max"], attempt["other_environment_max"]), flush=True)
        actual_tested = [pad["name"] for pad in report["pads"]]
        if actual_tested != report["tested_tactile_links"]:
            raise RuntimeError("Executed pad checks do not match the requested selection")
        report["passed"] = all(pad["passed"] for pad in report["pads"])
    except Exception as error:
        report["passed"] = False
        report["error"] = "{}: {}".format(type(error).__name__, error)
        report["traceback"] = traceback.format_exc()
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        if sim is not None:
            gym.destroy_sim(sim)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path(__file__).resolve().parents[1] / "assets/rapidhand")
    parser.add_argument("--asset-file", default="urdf/vision_hand.urdf")
    parser.add_argument("--pads", nargs="+", help="Test only these exact link names; default tests all tactile links")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU diagnostic; default is GPU pipeline")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", action="store_true", default=True)
    display.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--output", type=Path, help="Optional JSON report, e.g. /tmp/rapidhand_tactile.json")
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--probe-mass", type=float, default=0.02)
    parser.add_argument("--probe-stiffness", type=float, default=500.0, help="Dynamic probe position gain, N/m")
    parser.add_argument("--probe-damping", type=float, default=6.0, help="Dynamic probe damping, N*s/m")
    parser.add_argument("--max-probe-force", type=float, default=1.0)
    parser.add_argument("--contact-offset", type=float, default=0.0002)
    parser.add_argument("--clearance", type=float, default=0.008)
    parser.add_argument("--depths", default="0.00005,0.00015,0.0003", help="Shallow penetration retry depths in meters")
    parser.add_argument("--dt", type=float, default=1 / 240)
    parser.add_argument("--baseline-steps", type=int, default=12)
    parser.add_argument("--ramp-steps", type=int, default=60)
    parser.add_argument("--hold-steps", type=int, default=24)
    parser.add_argument("--settle-steps", type=int, default=24)
    parser.add_argument("--off-samples", type=int, default=8)
    parser.add_argument("--min-force", type=float, default=0.05,
                        help="Required hold peak, N; default matches the task contact threshold")
    parser.add_argument("--off-absolute", type=float, default=0.00001)
    parser.add_argument("--off-ratio", type=float, default=0.1)
    parser.add_argument("--crosstalk-tolerance", type=float, default=0.00001)
    args = parser.parse_args(argv)
    try:
        args.depths = [float(value) for value in args.depths.split(",")]
        if args.num_envs < 2:
            raise ValueError("At least two environments are required for the crosstalk check")
        if not 0 < args.contact_offset <= 0.0005:
            raise ValueError("contact-offset must be in (0, 0.0005] meters")
        if not args.depths or any(not 0 < depth <= 0.0005 for depth in args.depths):
            raise ValueError("depths must all be in (0, 0.0005] meters")
        if any(getattr(args, key) <= 0 for key in (
                "radius", "probe_mass", "probe_stiffness", "probe_damping", "max_probe_force",
                "clearance", "dt", "baseline_steps", "ramp_steps",
                "hold_steps", "settle_steps", "off_samples", "min_force", "off_absolute",
                "off_ratio", "crosstalk_tolerance")):
            raise ValueError("Probe settings and sample counts must be positive")
        if args.off_samples > args.settle_steps:
            raise ValueError("off-samples cannot exceed settle-steps")
        report = run_check(args)
    except Exception as error:
        report = {"passed": False, "error": "{}: {}".format(type(error).__name__, error),
                  "traceback": traceback.format_exc()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    print("RapidHand tactile check: {}".format("PASS" if report["passed"] else "FAIL"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
