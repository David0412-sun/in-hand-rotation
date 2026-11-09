"""Run an existing RapidHand teacher and collect a bounded sensor dataset."""

import sys
from pathlib import Path

import isaacgym  # Must precede torch.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import torch

import isaacgymenvs  # Registers the project's Hydra resolvers.
from isaacgymenvs.tasks.rapid_hand_rotation import RapidHandRotation
from isaacgymenvs.utils.dataset_collector import file_sha256
from rl_games.algos_torch.model_builder import ModelBuilder
from utils.utils import set_seed


@hydra.main(config_name="config", config_path="./cfg", version_base="1.1")
def collect(cfg):
    if cfg.task.name != "RapidHandRotation":
        raise ValueError("Sensor collection currently supports task=RapidHandRotation")
    if not cfg.checkpoint:
        raise ValueError("Specify checkpoint=<trained teacher .pth>")
    checkpoint_path = Path(to_absolute_path(cfg.checkpoint))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cfg.test = True
    cfg.task.dataset.enabled = True
    cfg.task.dataset.mode = "replay"
    cfg.task.dataset.checkpoint = str(checkpoint_path)
    cfg.task.dataset.output_dir = to_absolute_path(cfg.task.dataset.output_dir)
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.task.dataset.seed = cfg.seed
    task_config = OmegaConf.to_container(cfg.task, resolve=True)
    env = None
    try:
        env = RapidHandRotation(task_config, cfg.rl_device, cfg.sim_device,
                                cfg.graphics_device_id, cfg.headless, force_render=cfg.force_render)
        params = OmegaConf.to_container(cfg.train.params, resolve=True)
        model = ModelBuilder().load(params).build({
            "actions_num": env.num_actions, "input_shape": (env.num_obs,),
            "num_seqs": 1, "value_size": 1,
            "normalize_input": params["config"]["normalize_input"],
            "normalize_value": params["config"]["normalize_value"],
        }).to(cfg.rl_device).eval()
        checkpoint = torch.load(checkpoint_path, map_location=cfg.rl_device)
        model.load_state_dict(checkpoint["model"], strict=True)
        if model.is_rnn():
            raise ValueError("This collection entry currently supports the project's feed-forward teacher")
        # Metadata includes a checkpoint hash; loading never updates its weights.
        print("Collecting teacher epoch {} | checkpoint SHA256 {}".format(
            checkpoint.get("epoch", "unknown"), file_sha256(checkpoint_path)))
        deterministic = bool(params["config"].get("player", {}).get("deterministic", True))
        observations = env.reset()
        collector = env.dataset_collector
        while collector.active:
            with torch.no_grad():
                result = model({"is_train": False, "prev_actions": None,
                                "obs": observations["obs"], "rnn_states": None})
                actions = result["mus"] if deterministic else result["actions"]
                actions = torch.clamp(actions, -1, 1)
            observations, _, _, _ = env.step(actions)
            if not cfg.headless and not cfg.force_render and not cfg.task.camera.display:
                env.render()
        print("Collected {} synchronized rows: {}".format(collector.samples, collector.writer.directory))
    except KeyboardInterrupt:
        print("Collection interrupted; flushing completed samples.")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    collect()
