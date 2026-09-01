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