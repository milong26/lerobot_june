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
python extract_embeddings.py --force
之前相机的忘记提取了python extract_wrist_embeddings.py


3. 确定用什么方式得到每集的embedding
evaluate_episode_embeddings.py
这个代码运行以后应该可以得到比较好的以episode为单位的embedding，用于后续的微调
执行以后得到的结果显示
============================================================
TOP 20 TEMPORAL STRATEGIES
============================================================
Rank  Method                       Start%  End%    Len%    Weight       PosCorr  d_bar   Cluster  Overall 
--------------------------------------------------------------------------------------------------------------
1     temporal_multi_window        15      30      15      uniform      0.1118   0.0000  0.0000   0.0671  
2     temporal_multi_window        25      40      15      uniform      0.1118   0.0000  0.0000   0.0671  
3     temporal_multi_window        60      75      15      uniform      0.1118   0.0000  0.0000   0.0671  
4     temporal_multi_window        70      85      15      uniform      0.1118   0.0000  0.0000   0.0671  
5     temporal_multi_window        15      70      55      uniform      0.0965   0.1023  0.0000   0.0784  
6     temporal_multi_window        30      85      55      uniform      0.0914   0.1020  0.0000   0.0752  
7     temporal_multi_window        30      65      35      uniform      0.0902   0.1005  0.0000   0.0742  
8     temporal_multi_window        35      70      35      uniform      0.0901   0.1004  0.0000   0.0741  
9     temporal_window              50      95      45      uniform      0.0896   0.1017  288.4054 57.7552 
10    temporal_multi_window        45      60      15      uniform      0.0879   0.1003  0.0000   0.0728  
11    temporal_multi_window        10      80      70      uniform      0.0861   0.1000  0.0000   0.0716  
12    temporal_multi_window        5       30      25      uniform      0.0852   0.0823  0.0000   0.0676  
13    temporal_multi_window        70      95      25      uniform      0.0852   0.0823  0.0000   0.0676  
14    temporal_multi_window        40      95      55      uniform      0.0829   0.1021  0.0000   0.0702  
15    temporal_multi_window        20      75      55      uniform      0.0812   0.1021  0.0000   0.0691  
16    temporal_window              0       60      60      uniform      0.0796   0.1035  50.1713  10.1027 
17    temporal_multi_window        55      90      35      uniform      0.0789   0.0000  0.0000   0.0474  
18    temporal_multi_window        45      100     55      uniform      0.0775   0.0370  0.0000   0.0539  
19    temporal_window              30      90      60      uniform      0.0760   0.1049  30.2329  6.1132  
20    temporal_multi_window        30      90      60      uniform      0.0732   0.1018  0.0000   0.0643  

============================================================
GENERATING VISUALIZATIONS
============================================================
Saved embedding comparison to /data/zhonglinye/jun/lerobot/personal/work2/ours/extract_embedding_by_episode/embedding_comparison.png
Saved temporal importance to /data/zhonglinye/jun/lerobot/personal/work2/ours/extract_embedding_by_episode/temporal_importance.png

============================================================
SAVING RESULTS
============================================================
Saved evaluation results to /data/zhonglinye/jun/lerobot/personal/work2/ours/extract_embedding_by_episode/evaluation.csv
Saved temporal search results to /data/zhonglinye/jun/lerobot/personal/work2/ours/extract_embedding_by_episode/temporal_search_results.csv

Saving final episode embeddings...

============================================================
ADAPTIVE TEMPORAL EMBEDDING SEARCH
============================================================

Dataset:
MetaWorld PickPlace-v3

Episodes:
500

Frame embedding dimension:
1920

Final episode embedding dimension:
32

------------------------------------------------------------
BEST STRATEGY
------------------------------------------------------------

Method:
temporal_multi_window

Temporal window:
15% - 30%

Window length:
15%

Weighting:
uniform

------------------------------------------------------------
METRICS
------------------------------------------------------------

Position correlation:
0.1118

d_bar ratio:
0.0000

Cluster separation:
0.0000

Overall score:
0.0671

------------------------------------------------------------
BASELINE COMPARISON
------------------------------------------------------------

Full episode:
0.0220

Best temporal strategy:
0.1118

Improvement:
+409.09%

------------------------------------------------------------
RECOMMENDATION
------------------------------------------------------------

Use:
temporal_multi_window

Effective temporal region:
15% - 30%

Reason:
Highest position correlation on held-out episodes.






4. 找到ours

5. 对比方法
5.1 首先用random的方法，计算使用100数据集、200数据集、300数据集的成功率
2026年8月11日19:50:24：开始执行100和200seed=42的train，需要最后输出成功率结果
执行了
./jobs/run_random_200_seed42.sh
和
./jobs/run_random_100_seed42.sh