# 本地精简版：只暴露轻量的 TinyVLAConfig，供本地脚本核对 action_dim/state_dim/相机数等
# 训练配置元信息使用。真正的模型代码（modeling_tinyvla.py / llava_pythia / policy_heads）
# 只存在于 server-dev 分支、只跑在 policy_server.py 所在的服务器进程里，
# local-dev 故意不引入，以保持本地依赖精简。
from .configuration_tinyvla import TinyVLAConfig as TinyVLAConfig

__all__ = ["TinyVLAConfig"]
