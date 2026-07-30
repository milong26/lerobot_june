在10.10.16.18服务器执行
python -m lerobot.async_inference.policy_server      --host=10.10.16.18      --port=8080


本地生成测试脚本
work/test_smolvla_ep10_half$ python generate_smolvla_eval_scripts.py 

在ce0ce1ff99a8e421872207e8cf115ba69824c123 commitid执行测试


在work/test_smolvla_ep10_half/smolvla_eval_sh 目录
记录
script smolvla_terminal.log
conda activate lb_local

中途修改了一点键盘控制的代码，更新为commit id=
不影响测试
