
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


# 第二次修改代码


cd /data/zhonglinye/jun/lerobot && python personal/work2/duibi/train_and_eval_scripts/merge_selected_episodes.py \
    --dataset-root personal/work2/dataset_view \
    --dataset-names disassemble-v3_corner pick_place-v3_corner coffee-button-v3_corner \
    --episodes-per-dataset 112 \
    --selection-mode random \
    --seed 42

首先合并数据集
然后启动训练
bash personal/work2/differentvlm/minivla/scripts/launch_random.sh \
    --gpu-id 4 \
    --num-episodes 336 \
    --dataset-name merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42 \
    --seed 42
bash personal/work2/differentvlm/minivla/scripts/launch_random.sh \ 
    --gpu-id 4 \
    --num-episodes 336 \
    --dataset-name merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42 \
    --seed 42
bash: /data/zhonglinye/application/miniconda3/envs/lb_server/lib/libtinfo.so.6: no version information available (required by bash)
bash: /data/zhonglinye/application/miniconda3/envs/lb_server/lib/libtinfo.so.6: no version information available (required by bash)
tmux: /data/zhonglinye/application/miniconda3/envs/lb_server/lib/libtinfo.so.6: no version information available (required by tmux)
Launched MiniVLA experiment: random_336_seed42_merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42_minivla_random
tmux session: minivla_random_ep336_s42_merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42
Output dir: /data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/random_336_seed42_merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42_minivla_random
Dataset: merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42
Dataset root: /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42
GPU: 4
Episodes: 336
Selection mode: random

Monitor with: tmux attach -t minivla_random_ep336_s42_merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42
Check logs: tail -f /data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/random_336_seed42_merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42_minivla_random/logs/random_336_seed42_merged_disassemble-v3_corner+pick_place-v3_corner+coffee-button-v3_corner_112x112x112_random42_minivla_random.log


eval:



export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0

lerobot-eval \
    --policy.path=personal/work2/differentvlm/minivla/experiments/random_336_seed42_mergeddisassemblev3corner+pickplacev3corner+coffeebuttonv3corner112x112x112random42_minivla_random/checkpoints/checkpoints/038000/pretrained_model \
    --env.type=metaworld \
    --env.task=disassemble-v3 \
    --env.camera_name=corner,gripperPOV \
    --env.use_self_mw=true \
    --eval.batch_size=8 \
    --eval.n_episodes=200 \
    --policy.device=cuda \
    --rename_map='{"observation.images.camera1": "observation.images.top", "observation.images.camera2": "observation.images.wrist"}' \
    2>&1 | tee personal/work2/eval_model/eval_mw_miniall_disassemble-v3_corner_200ep_38kcp.log





our方法的话

python personal/work2/duibi/train_and_eval_scripts/merge_selected_episodes.py \
    --dataset-root personal/work2/dataset_view \
    --dataset-names disassemble-v3_corner pick_place-v3_corner coffee-button-v3_corner \
    --episodes-per-dataset 112 \
    --selection-mode ours_v5 \
    --seed 42 \
    --v5-output-dir personal/work2/duibi/our_v5_multi_336_seed42/subsets \
    --output-dir personal/work2/dataset_view/merged_3tasks_v5_336_seed42


