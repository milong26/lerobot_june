配置环境
重新配置
conda create -n robomme python=3.12 -y
conda activate robomme
python -m pip install -U pip wheel
conda install -c conda-forge ffmpeg -y
conda install -c conda-forge ffmpeg=7.1.1 -y



尝试执行采集程序
nohup python personal/work2/collect_dataset/robomme/collect_robomme_bytask.py \
  --task MoveCube \
  --difficulty easy \
  --num-random-episodes 300 \
  --num-uniform-episodes 100 \
  --output-dir personal/work2/dataset_view_robomme/MoveCube_easy \
  --repo-id work2/robomme_MoveCube_easy \
  > logs/movecube_easy.log 2>&1 &

再采集两个数据集
nohup python personal/work2/collect_dataset/robomme/collect_robomme_bytask.py \
  --task PatternLock \
  --difficulty medium \
  --num-random-episodes 300 \
  --num-uniform-episodes 100 \
  --output-dir personal/work2/dataset_view_robomme/PatternLock_medium \
  --repo-id work2/robomme_PatternLock_medium \
  > logs_patternlock_medium.log 2>&1 &

nohup python personal/work2/collect_dataset/robomme/collect_robomme_bytask.py \
  --task RouteStick \
  --difficulty hard \
  --num-random-episodes 300 \
  --num-uniform-episodes 100 \
  --output-dir personal/work2/dataset_view_robomme/RouteStick_hard \
  --repo-id work2/robomme_RouteStick_hard \
  > logs_routestick_hard.log 2>&1 &
