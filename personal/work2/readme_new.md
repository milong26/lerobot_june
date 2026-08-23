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
    --num-episodes 1 \
    --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset2 \
    --repo-id work2/metaworld_pick_place \
    --randomize-obj \
    --seed-start 0 \
    --max-steps 500

生成的时候是怎么指定工作空间的？用seed
换一下相机再重新生成1episode的数据看看
新相机更加符合我的需求，所以修改了代码以后重新采集
python personal/work2/collect_metaworld_dataset.py \
    --task pick-place-v3 \
    --num-episodes 500 \
    --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3 \
    --repo-id work2/metaworld_pick_place \
    --randomize-obj \
    --seed-start 0 \
    --max-steps 500



然后检查这个数据集，运行代码可以看到它的初始位置分布
python personal/work2/visualize_initial_positions.py \
    --dataset-dir personal/work2/dataset_view/pickplacev3 \
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

## see_embedding
因为之前extract embedding的时候不知道哪个方式更好，所以需要写代码验证
personal/work2/see_embedding 在这个文件夹里面
personal/work2/see_embedding/see_diff.py 这个代码大概比较了一下

personal/work2/see_embedding/see_embedding_deep.py 更加深入。
代码运行以后的结果是
SmolVLM 的视觉编码器对 pick-and-place 任务中的物体位置变化不敏感。绝对不能mean pooling
然后再选，但好像也选不出来什么？

TODO: 等能测试了再来继续刚这部分的代码



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



## 对比实验
### 代码修改
直接用lerobot_train的eval模式会报错，因为生成数据的时候可以提供top,wrist和state
默认代码评估的时候，只能提供单张图image和state
需要修改代码，最终得到image.top，image.wrist
（可能要看情况进行rename）
当lerobot_train 新增一个参数"use-self-mw"的时候，才用新的metaworld环境评估
代码修改完成。
lerobot-train \
  --env.use_self_mw=true
就能触发
输出 observation.images.camera1 和 observation.images.camera2
rename_map的时候要写camera1=top，caemra2=wrist

### 训练脚本
训练uniform，seed=42，100，200，挑选出100episode,200episode
训练random，seed分别是42和100，挑选出100episode和200episode

修改personal/work2/duibi/train_and_eval_scripts/train_and_eval.sh中的dataset等
在lerobot目录下运行

### uniform
cd /data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts
bash launch_uniform.sh 0
出现
Launched experiment: uniform_100_seed42
tmux session: uniform_100_s42
Output dir: /data/zhonglinye/jun/lerobot/personal/work2/duibi/uniform_42

Monitor with: tmux attach -t uniform_100_s42
Check logs: tail -f /data/zhonglinye/jun/lerobot/personal/work2/duibi/uniform_42/logs/uniform_100_seed42.log

检查是不是在正常运行，比如bacth_size要不要改一下，看起来可以正常运行
会首先生成subset，检查一下
 python personal/work2/dataset_lookin/see_uniform_dataset.py

## 测试eval能不能用
但是测试应该不需要多少资源啊？
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
    --policy.path=personal/work2/duibi/random_42/random_112_seed42/checkpoints/000200/pretrained_model \
    --env.type=metaworld \
    --env.task=metaworld-pick-place-v3 \
    --env.camera_name=corner2,gripperPOV \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cpu \
    --rename_map='{"observation.pixels/top": "observation.images.camera1", "observation.pixels/wrist": "observation.images.camera2"}'

测试的时候，batch_size=10是开启10个测试环境，max_episode_steps 默认self._max_episode_steps = 500 是从metaworld的环境来的


如果能用cuda
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
    --policy.path=personal/work2/duibi/random_42/random_112_seed42/checkpoints/000200/pretrained_model \
    --env.type=metaworld \
    --env.task=metaworld-pick-place-v3 \
    --env.camera_name=corner2,gripperPOV \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --rename_map='{"observation.pixels/top": "observation.images.camera1", "observation.pixels/wrist": "observation.images.camera2"}'