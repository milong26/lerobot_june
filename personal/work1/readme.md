要求：
训练和测试ep10所有数据，记录测试结果。并且使用分辨率减半的功能。


数据集准备
merge_dataset.py 合并所有子项
.gitignore中添加personal/work1/ep1_all/

训练指令
nohup env CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
lerobot-train \
  --dataset.repo_id=ep10/all \
  --dataset.root=personal/work1/ep10_all \
  --policy.path=lerobot/smolvla_base \
  --output_dir=outputs/test_smolvla_ep10 \
  --job_name=smolvla_ep10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --steps=1000 \
  --batch_size=32 \
  --rename_map='{"observation.images.side":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  > train_log/test_smolvla_ep10.log 2>&1 &
