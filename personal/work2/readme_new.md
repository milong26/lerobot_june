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

新增代码personal/work2/dataset_lookin/inspect_dataset.py
可以检查数据集，输出
============================================================
Dataset Overview
============================================================
Repo ID: lerobot/metaworld_pick_place
Num episodes: 500
Num frames: 31095
FPS: 80
Features: ['observation.images.top', 'observation.images.wrist', 'observation.state', 'observation.environment_state', 'action', 'next.reward', 'next.success', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']

============================================================
Episode Metadata (first 3 episodes)
============================================================
Type of dataset.meta.episodes: <class 'datasets.arrow_dataset.Dataset'>

============================================================
First Frame Structure
============================================================
Frame keys: ['observation.images.top', 'observation.images.wrist', 'observation.state', 'observation.environment_state', 'action', 'next.reward', 'next.success', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'task']
  observation.images.top: shape=torch.Size([3, 480, 480]), dtype=torch.float32
  observation.images.wrist: shape=torch.Size([3, 480, 480]), dtype=torch.float32
  observation.state: shape=torch.Size([4]), dtype=torch.float32
  observation.environment_state: shape=torch.Size([39]), dtype=torch.float32
  action: shape=torch.Size([4]), dtype=torch.float32
  next.reward: shape=torch.Size([]), dtype=torch.float32
  next.success: shape=torch.Size([]), dtype=torch.bool
  timestamp: shape=torch.Size([]), dtype=torch.float32
  frame_index: shape=torch.Size([]), dtype=torch.int64
  episode_index: shape=torch.Size([]), dtype=torch.int64
  index: shape=torch.Size([]), dtype=torch.int64
  task_index: shape=torch.Size([]), dtype=torch.int64
  task: len=31, type=<class 'str'>

============================================================
Episode 0 Frame (episode_index check)
============================================================
  episode_index: 0
  frame_index: 0
  index: 0

============================================================
Episode Meta Details
============================================================



## 寻找ours
就确定是metaworld的pickplacev3任务，唯一的目标是数据量少+成功率高
1. 每一集是什么、在哪里、有哪些任务变化
2. 从数据集中extract_embedding，首先检验smolvlm2

python extract_embeddings.py --num-keyframes 5 --force 运行这个代码以后，personal/work2/ours/pca/k5文件内得到了相应的embedding
这个过程是这样的：k=5的时候，按照0%, 25%, 50%, 75%, 100%的时间比例从每一集里面抽取5个关键帧，每个关键帧通过 SmolVLM2 的 vision_model，提取视觉特征 → connector 投影到语言空间，得到 (1, hidden_dim) 的 embedding，形状: (N, K, D) = (500, 5, 1024)
然后降维 PCA: (500, 5120) → (500, 32)
也顺便运行了k7的：python extract_embeddings.py --num-keyframes 7 --force


3. 运行selction的同时train和eval
现在不知道我的方法怎么样，所以运行selction以后，会将选择出来的子集保存到
work2/ours/subsets/ours_sic_b0_fps_size_20_subset.json
等等这种文件里面。

python run_experiment.py --stage select --strategy sic --target-size 200
结果会保存到subsets/select_sic_ts200_subset.json这个文件夹