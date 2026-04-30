"""LeRobot dataset loader for multimodal state + image training.

Date: 2026-04-02
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from loguru import logger

from mip.dataset_utils import (
    ImageNormalizer,
    MinMaxNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency in runtime
    cv2 = None


def _resolve_dataset_path(task_config) -> str:
    if getattr(task_config, "dataset_path", None):
        return os.path.expanduser(task_config.dataset_path)
    raise ValueError("LeRobot dataset requires task.dataset_path to be set")


def make_dataset(task_config, mode="train"):
    dataset_path = _resolve_dataset_path(task_config)

    if task_config.obs_type != "image":
        raise ValueError(
            "LeRobot dataset loader currently supports obs_type='image' "
            "(mixed rgb + low_dim via shape_meta)."
        )

    return LeRobotImageDataset(
        dataset_dir=dataset_path,
        shape_meta=task_config.shape_meta,
        n_obs_steps=task_config.obs_steps,
        horizon=task_config.horizon,
        pad_before=task_config.obs_steps - 1,
        pad_after=task_config.act_steps - 1,
        val_dataset_percentage=task_config.val_dataset_percentage,
        mode=mode,
    )


class LeRobotImageDataset(BaseDataset):
    """LeRobot dataset with mixed RGB and low-dimensional observations."""

    def __init__(
        self,
        dataset_dir: str,
        shape_meta: dict,
        n_obs_steps: int | None = None,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        val_dataset_percentage: float = 0.0,
        mode: str = "train",
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode

        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = []
        self.lowdim_keys = []
        self.source_obs_key_map: dict[str, str] = {}
        for key, attr in obs_shape_meta.items():
            self.source_obs_key_map[key] = attr.get("lerobot_key", key)
            if attr.get("type", "low_dim") == "rgb":
                self.rgb_keys.append(key)
            else:
                self.lowdim_keys.append(key)

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        self._episode_video_paths: list[dict[str, str]] = []
        self._frame_index_by_step: np.ndarray | None = None
        self._video_frame_index_by_step: np.ndarray | None = None
        self._load_episodes()

        key_first_k = {}
        if n_obs_steps is not None:
            for key in self.lowdim_keys:
                key_first_k[key] = n_obs_steps

        sampler_keys = self.lowdim_keys + ["action"]
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=sampler_keys,
            key_first_k=key_first_k,
        )

        self.normalizer = self.get_normalizer()
        self._episode_ends = self.replay_buffer.episode_ends.copy()
        self._episode_index_of_step = self.replay_buffer.get_episode_idxs()
        self._video_capture_cache: dict[str, object] = {}
        self._video_reader_cache: dict[str, imageio.Reader] = {}
        self._logged_decoder_fallback = False
        # AV1-encoded LeRobot videos are reliably handled by imageio in this setup.
        # Keep OpenCV decode path available as a fallback path, but disabled by default
        # to avoid repeated noisy decoder failures.
        self._disable_cv2_decoder = True

    def _list_data_files(self) -> list[Path]:
        data_files = sorted(self.dataset_dir.glob("data/chunk-*/file-*.parquet"))
        data_files = [p for p in data_files if not p.name.endswith(".phased.parquet")]
        if not data_files:
            raise FileNotFoundError(
                f"No LeRobot parquet files found under {self.dataset_dir / 'data'}"
            )
        return data_files

    def _split_files(self, all_files: list[Path]) -> list[Path]:
        total = len(all_files)
        if self.val_dataset_percentage <= 0.0:
            return all_files

        val_count = int(total * self.val_dataset_percentage)
        train_count = total - val_count
        if self.mode == "train":
            return all_files[:train_count]
        if self.mode == "val":
            return all_files[train_count:]
        raise ValueError(f"Invalid mode: {self.mode}. Must be 'train' or 'val'.")

    def _build_episode_video_paths(self, data_file: Path) -> dict[str, str]:
        chunk_dir = data_file.parent.name  # chunk-000
        file_stem = data_file.stem  # file-000
        mapping = {}
        for key in self.rgb_keys:
            source_key = self.source_obs_key_map[key]
            video_path = (
                self.dataset_dir / "videos" / source_key / chunk_dir / f"{file_stem}.mp4"
            )
            if not video_path.exists():
                raise FileNotFoundError(
                    f"Missing video file for {key} (source={source_key}): {video_path}"
                )
            mapping[key] = str(video_path)
        return mapping

    def _load_episodes(self):
        data_files = self._split_files(self._list_data_files())
        logger.info(f"Loading {len(data_files)} LeRobot data files for mode={self.mode}")
        frame_index_chunks: list[np.ndarray] = []
        video_frame_index_chunks: list[np.ndarray] = []

        for data_file in data_files:
            # Only the columns needed for training are loaded.
            lowdim_source_keys = [self.source_obs_key_map[key] for key in self.lowdim_keys]
            parquet_columns = set(pq.read_schema(data_file).names)
            has_global_index = "index" in parquet_columns
            columns = [
                "action",
                "episode_index",
                "frame_index",
                *lowdim_source_keys,
            ]
            if has_global_index:
                columns.append("index")
            frame_df = pd.read_parquet(
                data_file,
                columns=columns,
            )
            video_paths = self._build_episode_video_paths(data_file)
            episode_ids = frame_df["episode_index"].to_numpy()
            split_points = np.flatnonzero(np.diff(episode_ids)) + 1
            file_video_indices = (
                frame_df["index"].to_numpy(dtype=np.int64, copy=True)
                if has_global_index
                else np.arange(len(frame_df), dtype=np.int64)
            )

            start = 0
            for episode_df in np.split(frame_df, split_points):
                end = start + len(episode_df)
                action = np.stack(episode_df["action"].to_numpy()).astype(np.float32)
                episode = {"action": action}
                for key in self.lowdim_keys:
                    source_key = self.source_obs_key_map[key]
                    episode[key] = np.stack(episode_df[source_key].to_numpy()).astype(
                        np.float32
                    )
                self.replay_buffer.add_episode(episode)
                self._episode_video_paths.append(video_paths)
                frame_index_chunks.append(
                    episode_df["frame_index"].to_numpy(dtype=np.int64, copy=True)
                )
                video_frame_index_chunks.append(file_video_indices[start:end].copy())
                start = end

        self._frame_index_by_step = np.concatenate(frame_index_chunks, axis=0)
        self._video_frame_index_by_step = np.concatenate(
            video_frame_index_chunks, axis=0
        )

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __del__(self):
        video_capture_cache = getattr(self, "_video_capture_cache", {})
        video_reader_cache = getattr(self, "_video_reader_cache", {})

        for cap in video_capture_cache.values():
            if hasattr(cap, "release"):
                cap.release()
        video_capture_cache.clear()
        for reader in video_reader_cache.values():
            reader.close()
        video_reader_cache.clear()

    def _get_capture(self, video_path: str):
        if cv2 is None:
            return None
        cap = self._video_capture_cache.get(video_path)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            self._video_capture_cache[video_path] = cap
        return cap

    def _get_reader(self, video_path: str):
        reader = self._video_reader_cache.get(video_path)
        if reader is None:
            reader = imageio.get_reader(video_path)
            self._video_reader_cache[video_path] = reader
        return reader

    def _drop_reader(self, video_path: str):
        reader = self._video_reader_cache.pop(video_path, None)
        if reader is not None:
            reader.close()

    def _read_frame_cv2(self, video_path: str, frame_idx: int) -> np.ndarray | None:
        cap = self._get_capture(video_path)
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _read_frame_imageio(self, video_path: str, frame_idx: int) -> np.ndarray:
        reader = self._get_reader(video_path)
        return reader.get_data(int(frame_idx))

    def _read_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        if not self._disable_cv2_decoder:
            frame = self._read_frame_cv2(video_path, frame_idx)
            if frame is not None:
                return frame
            self._disable_cv2_decoder = True
            if not self._logged_decoder_fallback:
                logger.warning(
                    "OpenCV failed to decode at least one video frame. "
                    "Falling back to imageio decoder for LeRobot videos."
                )
                self._logged_decoder_fallback = True

        try:
            return self._read_frame_imageio(video_path, frame_idx)
        except Exception as exc:
            self._drop_reader(video_path)
            try:
                return self._read_frame_imageio(video_path, frame_idx)
            except Exception:
                frame = self._read_frame_cv2(video_path, frame_idx)
                if frame is not None:
                    return frame
                raise RuntimeError(
                    f"Failed to read frame {frame_idx} from video file: {video_path}"
                ) from exc

    def _frame_ids_for_sequence(self, sample_idx: int) -> np.ndarray:
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.sampler.indices[sample_idx]
        seq_len = self.sampler.sequence_length

        real_indices = np.arange(buffer_start_idx, buffer_end_idx, dtype=np.int64)
        frame_ids = np.empty((seq_len,), dtype=np.int64)

        if sample_start_idx > 0:
            frame_ids[:sample_start_idx] = real_indices[0]
        frame_ids[sample_start_idx:sample_end_idx] = real_indices
        if sample_end_idx < seq_len:
            frame_ids[sample_end_idx:] = real_indices[-1]

        return frame_ids

    def _global_to_episode_local(self, global_idx: int) -> tuple[int, int]:
        ep_idx = int(self._episode_index_of_step[global_idx])
        frame_idx = int(self._frame_index_by_step[global_idx])
        return ep_idx, frame_idx

    def _global_to_episode_video_frame(self, global_idx: int) -> tuple[int, int]:
        ep_idx = int(self._episode_index_of_step[global_idx])
        frame_idx = int(self._video_frame_index_by_step[global_idx])
        return ep_idx, frame_idx

    def _build_rgb_obs(self, sample_idx: int) -> dict[str, np.ndarray]:
        T_slice = slice(self.n_obs_steps)
        frame_ids = self._frame_ids_for_sequence(sample_idx)[T_slice]

        obs_dict = {}
        for key in self.rgb_keys:
            c, h_expected, w_expected = self.shape_meta["obs"][key]["shape"]
            frames = []
            for global_idx in frame_ids:
                ep_idx, video_idx = self._global_to_episode_video_frame(int(global_idx))
                frame = self._read_frame(
                    self._episode_video_paths[ep_idx][key],
                    video_idx,
                )
                if frame.shape[0] != h_expected or frame.shape[1] != w_expected:
                    frame = cv2.resize(
                        frame,
                        dsize=(w_expected, h_expected),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(frame)

            rgb = np.stack(frames, axis=0)  # (T, H, W, C)
            if rgb.shape[-1] != c:
                raise ValueError(
                    f"Unexpected channel count for {key}: got {rgb.shape[-1]}, expected {c}"
                )
            rgb = np.moveaxis(rgb, -1, 1).astype(np.float32) / 255.0  # (T, C, H, W)
            obs_dict[key] = self.normalizer["obs"][key].normalize(rgb)

        return obs_dict

    def __getitem__(self, idx: int):
        sample = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = self._build_rgb_obs(idx)
        for key in self.lowdim_keys:
            lowdim = sample[key][T_slice].astype(np.float32)
            obs_dict[key] = self.normalizer["obs"][key].normalize(lowdim)

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
        }
        return torch_data
