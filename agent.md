# Agent Guide: AIC MIP Training Pipeline and Encoder Fusion

This guide is for work on the AIC LeRobot image/state training path and proposed multi-camera encoder changes.

## Objective

Understand and modify the MIP training pipeline so left, center, and right camera images can be compared with cross-attention before appending joint-space and force/torque sensor observations.

Keep the downstream policy interface unchanged unless there is a strong reason to expand it. The preferred integration point is the observation encoder.

## High-Level Block Diagram

```text
Hydra config
  |
  v
examples/train_aic.py
  |
  +-- make_dataset(task_config)
  |     |
  |     v
  |   LeRobotImageDataset
  |     |
  |     +-- loads parquet actions and low-dimensional observations
  |     +-- maps RGB observation keys to LeRobot video files
  |     +-- samples fixed-length action/state windows
  |     +-- decodes camera frames on demand
  |
  v
TrainingAgent(config)
  |
  +-- get_encoder(network_config, task_config)
  |     |
  |     v
  |   MultiImageObsEncoder
  |
  +-- get_network(network_config, task_config)
        |
        v
      ChiTransformer / UNet / MLP / RNN action network
```

## Key Files

- `examples/train_aic.py`: AIC training entry point and training loop.
- `examples/configs/task/aic_lerobot_image_state.yaml`: AIC dataset/task declaration.
- `examples/configs/network/chitransformer.yaml`: ChiTransformer network config.
- `mip/datasets/lerobot_dataset.py`: LeRobot dataset adapter for RGB + low-dimensional observations.
- `mip/network_utils.py`: Constructs the encoder and action network from config.
- `mip/encoders.py`: Observation encoders, including `MultiImageObsEncoder`.
- `mip/networks/chitfm.py`: `ChiTransformer` action backbone.
- `mip/losses.py`: Flow-matching losses; calls `encoder(obs)` before the flow map.
- `mip/agent.py`: Owns encoder, flow map, optimizer, EMA, compile/cudagraph setup.

## Dataset Config Contract

The AIC task config uses `shape_meta` as the central schema:

```yaml
shape_meta:
  action:
    shape: [6]
  obs:
    state:
      shape: [32]
      type: low_dim
      lerobot_key: "observation.state"
    left_camera:
      shape: [3, 256, 288]
      type: rgb
      lerobot_key: "observation.images.left_camera"
    center_camera:
      shape: [3, 256, 288]
      type: rgb
      lerobot_key: "observation.images.center_camera"
    right_camera:
      shape: [3, 256, 288]
      type: rgb
      lerobot_key: "observation.images.right_camera"
```

`type: rgb` keys become camera streams. `type: low_dim` keys become state-like vectors. `lerobot_key` maps local model-facing names to LeRobot dataset column/video names.

If joint-space and force/torque are split in the dataset, prefer separate low-dimensional entries:

```yaml
joint_state:
  shape: [N]
  type: low_dim
  lerobot_key: "observation.joint_state"
force_torque:
  shape: [6]
  type: low_dim
  lerobot_key: "observation.force_torque"
```

The existing dataset loader already supports multiple low-dimensional keys.

## Dataset Loading Path

`make_dataset(task_config)` in `mip/datasets/lerobot_dataset.py` constructs `LeRobotImageDataset`.

Inside `LeRobotImageDataset`:

```text
shape_meta["obs"]
  |
  +-- rgb_keys: keys with type == "rgb"
  +-- lowdim_keys: all other observation keys
```

Low-dimensional arrays and actions are read from parquet into a replay buffer. RGB frames are not loaded eagerly. The loader stores video paths and per-key frame indices, then decodes frames in `__getitem__`.

Each sample has this structure:

```text
{
  "obs": {
    "left_camera":   (obs_steps, 3, H, W),
    "center_camera": (obs_steps, 3, H, W),
    "right_camera":  (obs_steps, 3, H, W),
    "state":         (obs_steps, state_dim),
  },
  "action":          (horizon, act_dim),
}
```

The training loop batches this into:

```text
left_camera:   (B, obs_steps, 3, H, W)
center_camera: (B, obs_steps, 3, H, W)
right_camera:  (B, obs_steps, 3, H, W)
state:         (B, obs_steps, state_dim)
action:        (B, horizon, act_dim)
```

## Training Start

Training begins in `examples/train_aic.py`.

The loop:

```text
batch = next(dataloader)
  |
  +-- obs tensors moved to optimization.device
  +-- obs truncated to task.obs_steps
  +-- action truncated to task.horizon
  |
  v
agent.update(action, obs, delta_t)
```

For image observations, `obs` is a `TensorDict` containing all camera and low-dimensional keys. The training loop does not fuse modalities itself.

## Model Construction

`TrainingAgent` calls:

```python
net = get_network(config.network, config.task)
encoder = get_encoder(config.network, config.task)
```

For `task.obs_type == "image"`, `get_encoder()` forces `MultiImageObsEncoder`.

For `network.network_type == "chitransformer"`, `get_network()` builds `ChiTransformer`.

The action network receives encoded observations only:

```text
raw obs dict
  |
  v
encoder(obs) -> obs_emb
  |
  v
flow_map.get_velocity(t, act_t, obs_emb)
```

`ChiTransformer` expects condition shape:

```text
(B, obs_steps, network.emb_dim)
```

So encoder changes should preserve that output shape for compatibility.

## Current Encoder Behavior

`MultiImageObsEncoder` currently performs late fusion:

```text
left_camera
  -> transform
  -> RGB backbone

center_camera
  -> transform
  -> RGB backbone

right_camera
  -> transform
  -> RGB backbone

state / low_dim
  -> optional low-dimensional MLP

all features
  -> concatenate
  -> final MLP
  -> (B, obs_steps, emb_dim)
```

With `use_seq=True`, images and low-dimensional vectors are flattened from `(B, T, ...)` to `(B*T, ...)`, encoded per timestep, then reshaped back to `(B, T, emb_dim)` when `keep_horizon_dims=True`.

## Recommended Encoder Modification

Add camera fusion inside `MultiImageObsEncoder`, after camera backbone features are produced and before low-dimensional state is appended.

Preferred abstract block:

```text
left image   -> vision backbone -> left token
center image -> vision backbone -> center token
right image  -> vision backbone -> right token
                                      |
                                      v
                         camera cross-attention fusion
                                      |
                                      v
                              fused visual token

joint state / force-torque
  -> low-dimensional projection

fused visual token + low-dimensional projections
  -> final projection MLP
  -> (B, obs_steps, emb_dim)
```

This preserves the existing training, loss, and action-network APIs.

## Camera Fusion Options

Symmetric self-attention over camera tokens:

```python
tokens = torch.stack([left_token, center_token, right_token], dim=1)
tokens = tokens + self.camera_pos_emb[:, :3]
tokens = self.camera_fusion(tokens)
visual_fused = tokens.mean(dim=1)
```

Center-camera query against all views:

```python
q = center_token.unsqueeze(1)
kv = torch.stack([left_token, center_token, right_token], dim=1)
fused, _ = self.camera_cross_attn(q, kv, kv)
visual_fused = fused.squeeze(1)
```

Use symmetric self-attention for the most neutral comparison against concatenation. Use center-query attention if the center view should be treated as the primary view.

## Suggested Config Additions

Add fields to `NetworkConfig` in `mip/config.py`:

```python
camera_fusion_type: str = "concat"  # "concat", "self_attention", "center_cross_attention"
camera_fusion_layers: int = 1
camera_fusion_heads: int = 4
camera_fusion_dropout: float = 0.0
```

Then pass them through `get_encoder()` in `mip/network_utils.py`.

Keep `"concat"` as the default to preserve current behavior and checkpoints.

## Testing Targets

Add or update encoder tests for:

- Current concat behavior remains unchanged.
- Three RGB streams plus one low-dimensional state key produce `(B, obs_steps, emb_dim)`.
- Three RGB streams plus separate `joint_state` and `force_torque` keys produce `(B, obs_steps, emb_dim)`.
- Attention fusion works with `share_rgb_model=True`.
- Attention fusion works with `use_seq=True` and `keep_horizon_dims=True`.

## Implementation Rule of Thumb

Do not modify `ChiTransformer` for camera-to-camera comparison unless the goal changes to action-token attention over individual camera tokens. For this task, the clean integration point is `MultiImageObsEncoder`; the downstream policy should continue receiving a normal observation embedding sequence.
