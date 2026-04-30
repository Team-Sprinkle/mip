uv run python examples/train_aic.py\
 task=aic_lerobot_image_state\
 network=chitransformer\
 optimization.gradient_steps=100000\
 optimization.batch_size=4\
 optimization.device=cuda\
 optimization.auto_resume=false\
 log.log_freq=10\
 log.save_freq=5000\
 log.eval_freq=1000\
 log.wandb_mode=online \
 log.project=aic \
 log.group=chitransformer
