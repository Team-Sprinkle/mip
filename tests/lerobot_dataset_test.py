"""Tests for LeRobot multimodal dataset adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pytest
import torch

from mip.agent import TrainingAgent
from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig
from mip.datasets.lerobot_dataset import make_dataset


@dataclass
class _TaskLikeConfig:
    dataset_path: str
    obs_type: str
    shape_meta: dict
    obs_steps: int
    act_steps: int
    horizon: int
    val_dataset_percentage: float = 0.0


def _write_episode(
    root: Path,
    *,
    file_idx: int,
    n_frames: int | list[int],
    state_dim: int,
    act_dim: int,
    h: int,
    w: int,
) -> None:
    chunk_dir = root / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(n_frames, int):
        episode_lengths = [n_frames]
    else:
        episode_lengths = list(n_frames)

    actions = []
    states = []
    episode_index = []
    frame_index = []
    global_step = 0
    for ep_idx, episode_len in enumerate(episode_lengths):
        for local_frame_idx in range(episode_len):
            actions.append(np.full((act_dim,), global_step, dtype=np.float32))
            states.append(
                np.linspace(global_step, global_step + 1, state_dim, dtype=np.float32)
            )
            episode_index.append(ep_idx)
            frame_index.append(local_frame_idx)
            global_step += 1

    rows = {
        "action": actions,
        "observation.state": states,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "index": list(range(global_step)),
    }
    df = pd.DataFrame(rows)
    file_stem = f"file-{file_idx:03d}"
    df.to_parquet(chunk_dir / f"{file_stem}.parquet", index=False)

    camera_keys = [
        "observation.images.left_camera",
        "observation.images.center_camera",
        "observation.images.right_camera",
    ]
    for cam_idx, cam_key in enumerate(camera_keys):
        video_dir = root / "videos" / cam_key / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            video_dir / f"{file_stem}.mp4", fps=10, macro_block_size=1
        )
        try:
            for episode_len in episode_lengths:
                for frame_idx in range(episode_len):
                    # Deterministic per-frame value helps padding assertions.
                    val = np.uint8(frame_idx * 10 + cam_idx)
                    frame = np.full((h, w, 3), val, dtype=np.uint8)
                    writer.append_data(frame)
        finally:
            writer.close()


def _default_shape_meta(h: int = 12, w: int = 16) -> dict:
    return {
        "action": {"shape": [6]},
        "obs": {
            "state": {
                "shape": [26],
                "type": "low_dim",
                "lerobot_key": "observation.state",
            },
            "left_camera": {
                "shape": [3, h, w],
                "type": "rgb",
                "lerobot_key": "observation.images.left_camera",
            },
            "center_camera": {
                "shape": [3, h, w],
                "type": "rgb",
                "lerobot_key": "observation.images.center_camera",
            },
            "right_camera": {
                "shape": [3, h, w],
                "type": "rgb",
                "lerobot_key": "observation.images.right_camera",
            },
        },
    }


@pytest.fixture()
def fake_lerobot_root(tmp_path: Path) -> Path:
    root = tmp_path / "lerobot_fake"
    _write_episode(root, file_idx=0, n_frames=6, state_dim=26, act_dim=6, h=12, w=16)
    _write_episode(root, file_idx=1, n_frames=6, state_dim=26, act_dim=6, h=12, w=16)
    return root


def _make_task_cfg(root: Path) -> _TaskLikeConfig:
    return _TaskLikeConfig(
        dataset_path=str(root),
        obs_type="image",
        shape_meta=_default_shape_meta(),
        obs_steps=2,
        act_steps=2,
        horizon=4,
        val_dataset_percentage=0.0,
    )


def test_lerobot_dataset_sample_schema_and_shapes(fake_lerobot_root: Path):
    dataset = make_dataset(_make_task_cfg(fake_lerobot_root))
    assert len(dataset) > 0

    sample = dataset[0]
    assert set(sample.keys()) == {"obs", "action"}
    assert set(sample["obs"].keys()) == {
        "state",
        "left_camera",
        "center_camera",
        "right_camera",
    }
    assert tuple(sample["obs"]["state"].shape) == (2, 26)
    assert tuple(sample["obs"]["left_camera"].shape) == (2, 3, 12, 16)
    assert tuple(sample["action"].shape) == (4, 6)
    assert sample["obs"]["state"].dtype == torch.float32
    assert sample["action"].dtype == torch.float32


def test_lerobot_dataset_normalizer_roundtrip(fake_lerobot_root: Path):
    dataset = make_dataset(_make_task_cfg(fake_lerobot_root))
    sample = dataset[0]

    state = sample["obs"]["state"].numpy()
    act = sample["action"].numpy()
    state_back = dataset.normalizer["obs"]["state"].unnormalize(state)
    act_back = dataset.normalizer["action"].unnormalize(act)

    assert np.isfinite(state_back).all()
    assert np.isfinite(act_back).all()
    # Image normalization should remain in [-1, 1].
    left = sample["obs"]["left_camera"].numpy()
    assert np.max(left) <= 1.0 + 1e-6
    assert np.min(left) >= -1.0 - 1e-6


def test_lerobot_sequence_padding_repeats_first_frame(fake_lerobot_root: Path):
    dataset = make_dataset(_make_task_cfg(fake_lerobot_root))
    frame_ids = dataset._frame_ids_for_sequence(0)
    assert frame_ids[0] == frame_ids[1] == 0

    sample = dataset[0]
    left = sample["obs"]["left_camera"].numpy()
    assert np.allclose(left[0], left[1])


def test_lerobot_missing_video_raises(tmp_path: Path):
    root = tmp_path / "lerobot_missing_video"
    _write_episode(root, file_idx=0, n_frames=4, state_dim=26, act_dim=6, h=12, w=16)
    missing = (
        root
        / "videos"
        / "observation.images.left_camera"
        / "chunk-000"
        / "file-000.mp4"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError):
        _ = make_dataset(_make_task_cfg(root))


def test_lerobot_imageio_fallback_path(fake_lerobot_root: Path):
    dataset = make_dataset(_make_task_cfg(fake_lerobot_root))
    dataset._disable_cv2_decoder = False
    dataset._logged_decoder_fallback = False

    def _always_fail_cv2(_video_path: str, _frame_idx: int):
        return None

    dataset._read_frame_cv2 = _always_fail_cv2  # type: ignore[method-assign]
    frame = dataset._read_frame(dataset._episode_video_paths[0]["left_camera"], 0)
    assert frame.shape == (12, 16, 3)
    assert dataset._disable_cv2_decoder is True


def test_lerobot_splits_multi_episode_parquet_and_uses_frame_index(tmp_path: Path):
    root = tmp_path / "lerobot_multi_episode"
    _write_episode(
        root,
        file_idx=0,
        n_frames=[3, 4],
        state_dim=26,
        act_dim=6,
        h=12,
        w=16,
    )
    dataset = make_dataset(_make_task_cfg(root))

    assert dataset.replay_buffer.n_episodes == 2
    assert dataset._frame_index_by_step.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert dataset._video_frame_index_by_step.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert dataset._global_to_episode_local(3) == (1, 0)
    assert dataset._global_to_episode_video_frame(3) == (1, 3)

    frame = dataset._read_frame(dataset._episode_video_paths[1]["left_camera"], 3)
    assert frame.shape == (12, 16, 3)


def test_lerobot_one_step_agent_update_cpu(fake_lerobot_root: Path):
    dataset = make_dataset(_make_task_cfg(fake_lerobot_root))
    sample = dataset[0]

    cfg = Config(
        optimization=OptimizationConfig(
            device="cpu",
            use_compile=False,
            use_cudagraphs=False,
            batch_size=1,
            gradient_steps=1,
            lr=1e-4,
        ),
        network=NetworkConfig(
            network_type="mlp",
            emb_dim=32,
            num_layers=2,
            num_encoder_layers=1,
            rgb_model_name="resnet18",
        ),
        task=TaskConfig(
            obs_type="image",
            obs_steps=2,
            act_steps=2,
            horizon=4,
            act_dim=6,
            obs_dim=32,
            shape_meta=_default_shape_meta(),
        ),
        log=LogConfig(
            log_dir="./logs",
            wandb_mode="disabled",
            project="test",
            group="test",
            exp_name="test",
        ),
    )
    agent = TrainingAgent(cfg)

    obs = {k: v.unsqueeze(0) for k, v in sample["obs"].items()}
    act = sample["action"].unsqueeze(0)
    delta_t = torch.tensor([1.0], dtype=torch.float32)
    info = agent.update(act, obs, delta_t)

    assert "loss" in info and "grad_norm" in info
    assert torch.isfinite(info["loss"])
    assert torch.isfinite(info["grad_norm"])
