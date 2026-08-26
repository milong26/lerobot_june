指定模型，用metaworld测试

因为出现提示
tesupport.py:24 No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate
所以安装了
conda install -c conda-forge pyopengl pyopengl-accelerate没啥用，不影响

1. 测试random_42能否有效果，wandb显示8kstep的时候loss就收敛了

CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
    --policy.path=personal/work2/duibi/random_42/random_112_seed42/checkpoints/last/pretrained_model \
    --env.type=metaworld \
    --env.task=metaworld-pick-place-v3 \
    --env.camera_name=corner2,gripperPOV \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --rename_map='{"observation.pixels/top": "observation.images.camera1", "observation.pixels/wrist": "observation.images.camera2"}'


1. 测试corner2
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
    --policy.path=personal/work2/duibi/uniform_42_corner2/uniform_112_seed42/checkpoints/008000/pretrained_model \
    --env.type=metaworld \
    --env.task=metaworld-pick-place-v3 \
    --env.camera_name=corner2,gripperPOV \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --rename_map='{"observation.pixels/top": "observation.images.camera1", "observation.pixels/wrist": "observation.images.camera2"}'


2. 测试corner1
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
    --policy.path=personal/work2/duibi/uniform_42/uniform_112_seed42/checkpoints/002000/pretrained_model \
    --env.type=metaworld \
    --env.task=metaworld-pick-place-v3 \
    --env.camera_name=corner,gripperPOV \
    --eval.batch_size=20 \
    --eval.n_episodes=20 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --rename_map='{"observation.pixels/top": "observation.images.camera1", "observation.pixels/wrist": "observation.images.camera2"}'




3. 测试corner3

CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
    --policy.path=personal/work2/duibi/uniform_42_corner3/uniform_112_seed42/checkpoints/007000/pretrained_model \
    --env.type=metaworld \
    --env.task=metaworld-pick-place-v3 \
    --env.camera_name=corner3,gripperPOV \
    --eval.batch_size=20 \
    --eval.n_episodes=20 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --rename_map='{"observation.pixels/top": "observation.images.camera1", "observation.pixels/wrist": "observation.images.camera2"}'




