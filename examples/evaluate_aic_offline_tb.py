"""Offline AIC MIP checkpoint evaluation with TensorBoard logging.

This evaluates a saved MIP checkpoint against recorded LeRobot sequences. It is
not a Gazebo rollout: it compares predicted action chunks with demonstration
actions and records a few observation clips so runs can be inspected together in
TensorBoard.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from mip.agent import TrainingAgent
from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig
from mip.datasets.lerobot_dataset import make_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--action-frame", choices=["tcp", "base"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--num-videos", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tb-root", type=Path, default=None)
    parser.add_argument("--network-config", default="chitransformer")
    parser.add_argument("--rgb-model-name", default="siglip2:google/siglip2-base-patch16-224")
    parser.add_argument("--emb-dim", type=int, default=384)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.pop("defaults", None)
    return data


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(args: argparse.Namespace) -> Config:
    config_root = Path(__file__).resolve().parent / "configs"
    task_cfg = load_yaml(config_root / "task" / "aic_lerobot_image_state.yaml")
    network_cfg = merge_dicts(
        load_yaml(config_root / "network" / "_base.yaml"),
        load_yaml(config_root / "network" / f"{args.network_config}.yaml"),
    )
    optimization_cfg = load_yaml(config_root / "optimization" / "default.yaml")
    log_cfg = load_yaml(config_root / "log" / "default.yaml")

    task_cfg["dataset_path"] = str(args.dataset_path)
    task_cfg["enable_env_eval"] = False
    task_cfg["use_group_norm"] = False
    network_cfg["rgb_model_name"] = args.rgb_model_name
    network_cfg["emb_dim"] = args.emb_dim
    optimization_cfg["device"] = args.device
    optimization_cfg["num_steps"] = args.num_steps
    optimization_cfg["use_compile"] = False
    optimization_cfg["use_cudagraphs"] = False
    optimization_cfg["auto_resume"] = False
    log_cfg["log_dir"] = str(args.run_dir)
    log_cfg["wandb_mode"] = "disabled"
    log_cfg["group"] = args.tag
    log_cfg["exp_name"] = f"offline_eval_{args.action_frame}"

    for float_key in ("lr", "weight_decay"):
        if float_key in optimization_cfg:
            optimization_cfg[float_key] = float(optimization_cfg[float_key])

    return Config(
        optimization=OptimizationConfig(**optimization_cfg),
        network=NetworkConfig(**network_cfg),
        task=TaskConfig(**task_cfg),
        log=LogConfig(**log_cfg),
        mode="eval",
    )


def step_from_checkpoint(path: Path) -> int:
    match = re.search(r"_step(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 0


def to_device_obs(obs_batch: dict[str, torch.Tensor], device: str, obs_steps: int) -> TensorDict:
    obs_dict = {k: v[:, :obs_steps].to(device) for k, v in obs_batch.items()}
    batch_size = next(iter(obs_dict.values())).shape[0]
    return TensorDict(obs_dict, batch_size=batch_size)


def unnormalize_action(dataset, action: np.ndarray) -> np.ndarray:
    return dataset.normalizer["action"].unnormalize(action.astype(np.float32))


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    tb_root = args.tb_root or (args.run_dir / "tensorboard")
    tb_dir = tb_root / "offline_eval" / args.tag
    writer = SummaryWriter(log_dir=str(tb_dir))

    config = build_config(args)
    if config.optimization.device.startswith("cuda") and not torch.cuda.is_available():
        config.optimization.device = "cpu"

    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)
    agent.load(str(args.checkpoint), load_optimizer=False)
    agent.eval()

    step = step_from_checkpoint(args.checkpoint)
    max_items = min(len(dataset), args.num_sequences)
    indices = np.linspace(0, len(dataset) - 1, max_items, dtype=int)

    all_pred = []
    all_target = []
    video_count = 0
    start = config.task.obs_steps - 1
    end = start + config.task.act_steps

    with torch.no_grad():
        for batch_start in range(0, max_items, args.batch_size):
            batch_indices = indices[batch_start : batch_start + args.batch_size]
            samples = [dataset[int(i)] for i in batch_indices]
            obs_batch = {
                key: torch.stack([sample["obs"][key] for sample in samples], dim=0)
                for key in samples[0]["obs"]
            }
            action = torch.stack([sample["action"] for sample in samples], dim=0)
            obs = to_device_obs(obs_batch, config.optimization.device, config.task.obs_steps)
            act_0 = torch.randn(
                (len(samples), config.task.horizon, config.task.act_dim),
                device=config.optimization.device,
            )
            pred_norm = agent.sample(
                act_0=act_0,
                obs=obs,
                num_steps=args.num_steps,
                use_ema=True,
            )
            pred = unnormalize_action(dataset, pred_norm[:, start:end].cpu().numpy())
            target = unnormalize_action(dataset, action[:, start:end].numpy())
            all_pred.append(pred)
            all_target.append(target)

            if video_count < args.num_videos and "center_camera" in obs_batch:
                clips = obs_batch["center_camera"][: args.num_videos - video_count]
                clips = (clips + 1.0) / 2.0
                clips = clips.clamp(0.0, 1.0)
                writer.add_video(
                    f"offline_eval/{args.tag}/center_camera",
                    clips,
                    global_step=step + video_count,
                    fps=20,
                )
                video_count += clips.shape[0]

    pred_arr = np.concatenate(all_pred, axis=0)
    target_arr = np.concatenate(all_target, axis=0)
    err = pred_arr - target_arr
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))

    writer.add_scalar("offline_eval/action_mse", mse, step)
    writer.add_scalar("offline_eval/action_mae", mae, step)
    writer.add_scalar("offline_eval/action_max_abs_error", max_abs, step)
    writer.add_scalar("offline_eval/num_sequences", max_items, step)
    writer.add_scalar("offline_eval/num_sampling_steps", args.num_steps, step)
    for dim in range(err.shape[-1]):
        writer.add_scalar(f"offline_eval/action_dim_{dim}_mse", float(np.mean(err[..., dim] ** 2)), step)
        writer.add_scalar(f"offline_eval/action_dim_{dim}_mae", float(np.mean(np.abs(err[..., dim]))), step)

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_path": str(args.dataset_path),
        "tag": args.tag,
        "action_frame": args.action_frame,
        "step": step,
        "num_sequences": max_items,
        "num_sampling_steps": args.num_steps,
        "action_mse": mse,
        "action_mae": mae,
        "action_max_abs_error": max_abs,
        "tensorboard_dir": str(tb_dir),
    }
    summary_path = args.run_dir / f"offline_eval_{args.tag}_step{step}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    writer.add_text("offline_eval/summary_json", json.dumps(summary, indent=2), step)
    writer.flush()
    writer.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
