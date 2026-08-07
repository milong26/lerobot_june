目标：研究单任务微调什么数据集能实现最好的效果

# 初步测试：metaworld验证
metaworld+ppick-place-v3任务
## 准备数据集
目标：生成大量数据集，存储在work2/dataset目录
数据集是500episode
每次初始位置都随机，最终得到的数据集，它里面的初始位置应该是均匀采样的

运行collect_metaworld_dataset.py，指令是
python personal/work2/collect_metaworld_dataset.py \
    --task pick-place-v3 \
    --num-episodes 500 \
    --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \
    --repo-id your-username/metaworld_pick_place \
    --randomize-obj \
    --seed-start 0 \
    --max-steps 500

然后检查这个数据集，运行代码可以看到它的初始位置分布
python personal/work2/visualize_initial_positions.py \
    --dataset-dir personal/work2/dataset \
    --output-dir personal/work2/dataset_lookin/

采集完以后做什么？
1. 用完整的微调smolvla，tinyvla和pi0fast，evo-1
2. 