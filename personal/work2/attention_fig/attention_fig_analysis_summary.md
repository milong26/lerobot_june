# 基于 Attention 与 Action Trajectory 的不同数据集训练模型差异分析

## 1. 研究背景与目标

本实验研究不同数据采集策略对机器人操作模型性能的影响，重点分析不同初始摆放位置的数据分布差异如何影响模型学习到的
visuomotor 能力。

实验假设为：训练数据覆盖范围会影响模型对状态空间和动作空间的学习。如果采集的数据能够覆盖已有数据集
embedding
空间中的空洞区域，减少训练数据分布中的未覆盖区域，模型能够学习更加完整的状态到动作映射，从而提升未知初始状态下的任务成功率。

本实验重点分析三个问题：不同数据集训练后的模型是否产生不同的输入感知模式；不同数据集训练后的模型是否产生不同的动作生成轨迹；动作生成差异是否能够解释任务成功率差异。

## 2. Attention Figure 分析代码功能说明

代码路径：

    personal/work2/attention_fig

实验输出路径：

    personal/work2/attention_fig/result_inference_trace

该目录实现了一套针对 SmolVLA Flow Matching 推理过程的模型分析框架。

代码支持多个不同数据集训练模型的公平比较，例如：

    random_corner_16k
    uniform_corner_16k
    ours_corner_16k

模型比较过程中固定 MetaWorld 初始环境状态、camera observation、robot
state、language instruction 以及 Flow Matching initial noise。

每个 camera group 首先创建共享环境状态，生成 reference
observation。之后每个模型使用相同 observation 副本进行推理。

代码通过 SHA256 对输入进行一致性验证。当前实验日志显示：

camera1 SHA256: 8ce905612bb9def9

camera2 SHA256: b92683fbf52aa33a

state SHA256: 376ad397079b3f96

language SHA256: 8a7ef22f61b29941

noise SHA256: 86e681c6f819505c

三个模型使用完全一致的输入条件。

代码支持从：

    personal/work2/attention_fig/metaworld_config.json

读取 MetaWorld task 对应自然语言描述，并经过 SmolVLA tokenizer 生成
language tokens 和 language attention mask。

运行结果：

    Language tensor shape: torch.Size([1,48])
    Non-padding language tokens: 10

说明模型输入包含完整 language conditioning。

代码通过真实 Flow Matching 推理过程记录动作生成轨迹，保存：

    trace.pt

其中包含 denoising timestep、x_t action state、v_t velocity
prediction、suffix hidden representation 和 final action。

代码同时提取推理过程中的 attention 信息，包括视觉 token attention、语言
token attention 和 state 相关 attention。

代码生成：

    result_inference_trace/plots/pairwise_action_divergence.csv

用于分析不同模型之间的 action trajectory 差异。

主要指标包括 x_t_l2、v_t_l2、v_t_cosine 和 hidden_cosine。

## 3. 实验结果分析

Attention 分析结果位于：

    result_inference_trace/plots

结果显示，不同数据集训练模型在输入模态利用方面整体接近。

不同模型对于 camera observation、language instruction 和 robot state
保持相似的信息利用模式。

Action trajectory 分析结果位于：

    result_inference_trace/plots/pairwise_action_divergence.csv

当前结果来自 corner3, gripperPOV，比较模型：

    random_corner3_16k
    uniform_corner3_16k
    ours_corner3_16k

x_t trajectory 分析显示，三个模型从相同初始 action noise 开始生成动作。

初始 denoising 阶段：

x_t_l2 = 0

随着 denoising step 增加，不同模型动作状态逐渐分离。

Random 与 Ours 最终 x_t_l2 = 6.86。

Uniform 与 Ours 最终 x_t_l2 = 7.13。

Random 与 Uniform 最终 x_t_l2 = 5.96。

这说明不同数据集训练模型在动作生成过程中逐渐形成不同轨迹。

Velocity field 分析显示，不同模型学习到了不同的 Flow Matching velocity
field。

最终 denoising step：

Random 与 Ours 的 v_t_l2 = 10.54。

Random 与 Uniform 的 v_t_l2 = 8.68。

Uniform 与 Ours 的 v_t_l2 = 12.88。

Hidden representation 分析显示，action expert 内部表示随着 denoising
过程逐渐分离。

Random 与 Ours 的 hidden cosine 从 0.963 降低到 0.896。

Uniform 与 Ours 的 hidden cosine 从 0.957 降低到 0.901。

说明不同数据集训练后的模型在 action expert 内部任务表示上产生差异。

## 4. Velocity Trajectory 分析

结果文件：

    result_inference_trace/v_t_norm_analysis.csv

统计不同模型每一步 velocity magnitude。

实验结果：

Random 模型平均 v_t_norm 为 22.42，标准差为 2.93。

Uniform 模型平均 v_t_norm 为 21.81，标准差为 2.61。

Ours 模型平均 v_t_norm 为 22.45，标准差为 2.22。

Random 模型变化范围：20.25 到 29.29。

Uniform 模型变化范围：19.97 到 28.14。

Ours 模型变化范围：20.70 到 27.54。

三个模型具有不同的 velocity evolution。

Ours 模型 velocity trajectory 波动范围较小，动作 refinement
过程更加平滑。

## 5. 与 embedding 空洞数据采集假设的关联

当前实验结果支持 embedding 空洞数据采集假设。

实验显示，不同数据分布训练模型在输入 attention
层面的变化较小，而主要差异出现在 action expert 内部表示、Flow Matching
velocity field 和 action trajectory。

这说明数据采集策略影响模型学习到的状态空间与动作空间关系。

如果训练数据能够覆盖已有数据分布中的空洞区域，模型能够学习更加完整的
visuomotor mapping。

这种影响最终表现为相同视觉输入条件下，不同模型产生不同动作生成轨迹。

当前结果说明，数据采集分布影响的是模型内部 action manifold 结构。

## 6. 综合结论

当前实验建立了从数据分布到动作生成差异的分析链路。

实验结果表明：

不同数据集训练模型在相同 observation、language、state 和 noise
条件下，会产生不同 action trajectory。

Attention 分析显示，不同模型输入信息利用模式整体接近。

Action divergence 分析显示，不同数据集训练模型学习到了不同 Flow Matching
velocity dynamics。

Hidden representation 分析显示，数据集差异会影响 action expert
内部任务表示。

Velocity trajectory 分析显示，不同模型具有不同动作 refinement 特征。

因此，不同初始摆放位置的数据采集策略主要影响模型学习到的 visuomotor
action manifold。采集覆盖 embedding
空洞区域的数据，有可能通过改善状态空间覆盖和动作空间表示，提高模型对于未见状态的泛化能力。

后续需要结合训练数据 embedding 分布、测试状态距离、success rate 和
trajectory stability，进一步建立数据覆盖范围与任务成功率之间的定量关系。
