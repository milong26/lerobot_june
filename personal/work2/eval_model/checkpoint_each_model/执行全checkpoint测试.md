2026年9月2日 23:45:49 开始

chmod +x personal/work2/eval_model/checkpoint_each_model/*.sh
bash personal/work2/eval_model/checkpoint_each_model/batch_eval_gpu1_partA.sh
[INFO] Launching eval worker in tmux session 'eval_gpu1_partA'...
[INFO] tmux session 'eval_gpu1_partA' started.
[INFO] Attach with: tmux attach -t eval_gpu1_partA
[INFO] View log: tail -f /data/zhonglinye/jun/lerobot/personal/work2/eval_model/checkpoint_each_model/results_gpu1_partA/run.log


bash personal/work2/eval_model/checkpoint_each_model/batch_eval_gpu1_partB.sh
bash personal/work2/eval_model/checkpoint_each_model/batch_eval_gpu1_partB.sh
[INFO] Launching eval worker in tmux session 'eval_gpu1_partB'...
[INFO] tmux session 'eval_gpu1_partB' started.
[INFO] Attach with: tmux attach -t eval_gpu1_partB
[INFO] View log: tail -f /data/zhonglinye/jun/lerobot/personal/work2/eval_model/checkpoint_each_model/results_gpu1_partB/run.log

2026年9月3日 09:53:37 好像还在运行，那查看一下日志运行到哪里了
有点慢，想优先测试一下our_v3和our_v4的，为什么our_corner的成功率都是0，不应该啊
