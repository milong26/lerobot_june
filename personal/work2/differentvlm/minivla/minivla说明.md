
# 训练

bash personal/work2/differentvlm/minivla/scripts/launch_random.sh \
    --gpu-id 0 \
    --num-episodes 200 \
    --dataset-name disassemble-v3_corner \
    --seed 42

运行了一个
bash personal/work2/differentvlm/minivla/scripts/launch_random.sh \
    --gpu-id 0 \
    --num-episodes 200 \
    --dataset-name disassemble-v3_corner \
    --seed 42
Launched MiniVLA experiment: random_200_seed42_disassemble-v3_corner_minivla_random
tmux session: minivla_random_ep200_s42_disassemble-v3_corner
Output dir: /data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/random_200_seed42_disassemble-v3_corner_minivla_random
Dataset: disassemble-v3_corner
Dataset root: /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner
GPU: 0
Episodes: 200
Selection mode: random

Monitor with: tmux attach -t minivla_random_ep200_s42_disassemble-v3_corner
Check logs: tail -f /data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/random_200_seed42_disassemble-v3_corner_minivla_random/logs/random_200_seed42_disassemble-v3_corner_minivla_random.log

# 测试
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0

lerobot-eval \
    --policy.path=personal/work2/differentvlm/minivla/experiments/random_200_seed42_disassemblev3corner_minivla_random/checkpoints/checkpoints/050000/pretrained_model \
    --env.type=metaworld \
    --env.task=disassemble-v3 \
    --env.camera_name=corner,gripperPOV \
    --env.use_self_mw=true \
    --eval.batch_size=8 \
    --eval.n_episodes=200 \
    --policy.device=cuda \
    --rename_map='{"observation.images.camera1": "observation.images.top", "observation.images.camera2": "observation.images.wrist"}' \
    2>&1 | tee personal/work2/eval_model/eval_mw_minivla_disassemble-v3_corner_200ep_50kcp.log




