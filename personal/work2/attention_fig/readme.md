运行代码，检查注意力图

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=1

python personal/work2/attention_fig/plot_attention.py \
  --device cuda \
  --mode probe \
  --seed 10042 \
  --query-mode mean_suffix \
  --layers 0,3,7,11 \
  --average-heads \
  2>&1 | tee personal/work2/attention_fig/attention_probe_all.log

修改了代码以后执行新的
cd /data/zhonglinye/jun/lerobot
conda activate lb_server
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
nohup python personal/work2/attention_fig/plot_attention.py \
  --mode inference_trace \
  --device cuda \
  --seed 10042 \
  --noise-seed 10042 \
  --query-mode mean_suffix \
  --layers 3,7,11 \
  --trace-heatmap-steps first,mid,last \
  --sanity-check \
  --output-dir personal/work2/attention_fig/result_inference_trace \
  > personal/work2/attention_fig/inference_trace.log 2>&1 &

终于改完代码执行成功了，怎么看结果
默认代码只执行了
Models: ['random_corner_16k', 'uniform_corner_16k', 'ours_corner_16k']
这三个模型

之前已经执行过，在最后的sanity_check报错，所以
nohup python personal/work2/attention_fig/plot_attention.py \
  --mode inference_trace \
  --device cuda \
  --seed 10042 \
  --noise-seed 10042 \
  --query-mode mean_suffix \
  --layers 3,7,11 \
  --trace-heatmap-steps first,mid,last \
  --sanity-check \
  --load-from-cache \
  --output-dir personal/work2/attention_fig/result_inference_trace \
  > personal/work2/attention_fig/inference_trace_continue.log 2>&1 &


ok这边暂时不看了。