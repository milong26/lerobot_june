cd /data/zhonglinye/jun/lerobot
代码中已经设置好了egl了

这个代码会采集数据，多等待10second，好像没有指定不同相机的地方，默认使用fps=20
python personal/work2/collect_dataset/libero/collect_libero_dataset.py \
    --suite libero_spatial \
    --task-id 0 \
    --num-episodes 300 \
    --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset/libero_spatial_task0 \
    --repo-id work2/libero_spatial_task0 \
    --fps 20 --image-size 360 \
    --seed-start 0 \
    --candidate-multiplier 10