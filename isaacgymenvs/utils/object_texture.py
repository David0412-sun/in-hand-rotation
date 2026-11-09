"""Load one verified, object-fixed texture atlas and reuse it across actors.

The atlas is bound to the mesh's local UV coordinates. Pose, environment ID,
episode ID and the policy random seed never select or regenerate its pixels.
The simulation owns the resulting GPU texture and releases it on destruction.
"""

import copy
import hashlib
import json
from pathlib import Path


class FixedObjectTexture:
    """A deterministic material shared by all instances of one object asset."""

    def __init__(self, gym, sim, asset_root, config):
        self.gym, self.sim = gym, sim
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.object_name = str(config.get("object_name", "set_obj6_block_corner"))
        self.atlas_path = self._asset_path(config.get(
            "atlas_file", "textures/chamfered_block_solid_v3/atlas.png"))
        self.manifest_path = self._asset_path(config.get(
            "manifest_file", "textures/chamfered_block_solid_v3/manifest.json"))
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Cannot read fixed texture manifest: {}".format(
                self.manifest_path)) from error
        if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int \
                or manifest["schema_version"] != 1:
            raise ValueError("Fixed texture manifest must use schema_version=1")
        if manifest.get("object_name") != self.object_name:
            raise ValueError("Fixed texture object_name does not match configured object: {}".format(
                self.object_name))
        if not isinstance(manifest.get("texture_id"), str) or not manifest["texture_id"]:
            raise ValueError("Fixed texture manifest requires a nonempty texture_id")
        if type(manifest.get("seed")) is not int:
            raise ValueError("Fixed texture manifest requires an integer seed")

        self._verify_file(self.atlas_path, manifest.get("texture_sha256"), "texture")
        mesh_file = manifest.get("mesh_file")
        if not isinstance(mesh_file, str) or not mesh_file or Path(mesh_file).is_absolute():
            raise ValueError("Fixed texture mesh_file must be relative to asset_root")
        mesh_path = self._asset_path(mesh_file)
        mesh_bytes = self._verify_file(mesh_path, manifest.get("mesh_sha256"), "mesh")
        faces = manifest.get("faces")
        if not isinstance(faces, list) or not faces or any(not isinstance(face, dict) for face in faces):
            raise ValueError("Fixed texture manifest requires a nonempty faces list")
        face_ids = [face.get("face_id") for face in faces]
        if any(type(face_id) is not int for face_id in face_ids) \
                or sorted(face_ids) != list(range(len(faces))):
            raise ValueError("Fixed texture face_id values must cover every face exactly once")
        mesh_face_count = sum(line.lstrip().startswith("f ")
                              for line in mesh_bytes.decode("utf-8").splitlines())
        if mesh_face_count != len(faces):
            raise ValueError("Fixed texture manifest face count differs from the source OBJ")
        try:
            json.dumps(manifest, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Fixed texture metadata must contain finite JSON values") from error
        self._metadata = copy.deepcopy(manifest)
        self._metadata.update(atlas_file=str(self.atlas_path),
                              manifest_file=str(self.manifest_path))

        self.texture_handle = self.gym.create_texture_from_file(self.sim, str(self.atlas_path))
        if self.texture_handle is None or self.texture_handle < 0:
            raise RuntimeError("Isaac Gym could not load fixed texture: {}".format(self.atlas_path))

    def _asset_path(self, filename):
        path = Path(filename).expanduser()
        return (path if path.is_absolute() else self.asset_root / path).resolve()

    @staticmethod
    def _verify_file(path, expected_hash, label):
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 \
                or any(character not in "0123456789abcdef" for character in expected_hash):
            raise ValueError("Fixed texture manifest requires a lowercase {} SHA256".format(label))
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ValueError("Cannot read fixed texture {} file: {}".format(label, path)) from error
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("Fixed texture {} SHA256 mismatch: {}".format(label, path))
        return data

    @property
    def metadata(self):
        """Return JSON-safe provenance without exposing mutable internal state."""
        return copy.deepcopy(self._metadata)

    def apply(self, env, actor, name):
        """Texture matching actors only; return whether an actor was modified."""
        if name != self.object_name:
            return False
        # Keep metadata verification/import usable without loading the Gym binary.
        from isaacgym import gymapi

        body_count = self.gym.get_actor_rigid_body_count(env, actor)
        if body_count <= 0:
            raise RuntimeError("Cannot texture an object actor with no rigid bodies")
        for body in range(body_count):
            color_result = self.gym.set_rigid_body_color(
                env, actor, body, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0))
            texture_result = self.gym.set_rigid_body_texture(
                env, actor, body, gymapi.MESH_VISUAL, self.texture_handle)
            if color_result is False or texture_result is False:
                raise RuntimeError("Isaac Gym could not apply fixed texture to body {}".format(body))
        return True
