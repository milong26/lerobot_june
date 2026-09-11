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