uv run python examples/train_aic.py\
 task=aic_lerobot_image_state\
 optimization.gradient_steps=1000\
 optimization.batch_size=4\
 optimization.device=cpu\
 optimization.auto_resume=false\
 log.log_freq=10\
 log.save_freq=100\
 log.eval_freq=100
