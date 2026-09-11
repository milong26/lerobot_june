配置环境
重新配置
conda create -n robomme python=3.12 -y
conda activate robomme
python -m pip install -U pip wheel
conda install -c conda-forge ffmpeg -y
conda install -c conda-forge ffmpeg=7.1.1 -y



尝试执行采集程序
nohup python personal/work2/collect_dataset/robomme/collect_robomme_bytask.py \
    --task PickXtimes \
    --num-random-episodes 300 \
    --num-uniform-episodes 100 \
    --output-dir personal/work2/dataset_view_robomme/PickXtimes/ \
    --repo-id work2/robomme_PickXtimes \
    > personal/work2/collect_dataset/robomme/collect_robomme.log 2>&1 &