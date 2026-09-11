配置环境
pip install gymnasium==0.29.1 numpy==1.26.4
pip install -e ".[smolvla,av-dep]"
pip install "robomme @ git+https://github.com/RoboMME/robomme_benchmark.git@main"
pip install numpy==2.2.6 gymnasium==1.3.0 因为tinvyla那些已经配置好了

尝试执行采集程序
nohup python personal/work2/collect_dataset/robomme/collect_robomme_bytask.py \
    --task PickXtimes \
    --num-random-episodes 300 \
    --num-uniform-episodes 100 \
    --output-dir personal/work2/dataset_view_robomme/PickXtimes/ \
    --repo-id work2/robomme_PickXtimes \
    > personal/work2/collect_dataset/robomme/collect_robomme.log 2>&1 &