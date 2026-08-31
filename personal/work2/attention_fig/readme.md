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


# 结论是什么
不同数据集微调模型的成功率高低，能否用注意力图来解释？



# 结果怎么看
看图？



# 复用
我的ours应该有问题，所以下次看代码的时候肯定还要再看