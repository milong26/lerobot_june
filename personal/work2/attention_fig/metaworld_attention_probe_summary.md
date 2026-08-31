# MetaWorld / SmolVLA Attention Probe 实验总结
包含 200-episode Eval、attention 工具修复、8 个模型 initial probe 结果与后续分析方法
相关提交：4e2db26 “注意力图” → 4b19e47 “注意力图结果”
| 核心结论 当前 initial-state attention probe 没有发现 Ours 存在“明显忽略某一路相机”或“完全不同的 modality 使用方式”。同一 camera 条件下 Random / Uniform / Ours 的 camera1、camera2、language、state attention mass 非常接近。性能差异更可能来自图像内部关注区域、task-state representation、数据覆盖或 post-grasp 控制阶段，而不是简单的“看不看某个 camera”。 |
| --- |

## 1. 研究背景与当前问题
当前任务为 MetaWorld pick-place-v3，使用 SmolVLA policy，在 corner / corner2 / corner3 三种全局相机视角加 gripperPOV wrist camera 的条件下比较 Random、Uniform 与 Ours 数据选择方法。此前 Eval 图像归一化问题修复后，已有 checkpoint 可以正常完成抓取与放置测试。
当前最重要的研究现象是：多数模型 grasp success 很高，但最终 task success 明显更低。这意味着模型通常能完成“找到物体并抓住”，但在 lift、transport、approach-goal、release/place 等 post-grasp 阶段仍存在明显失败。
Ours 的设计原理依赖 VLM embedding 的可区分性与覆盖性：如果不同 task state 在 embedding 空间中无法形成有意义的结构，那么基于 embedding distance / SIC 的选择也难以产生优势。因此需要同时从 Dataset/Embedding 层和 Policy Attention 层进行诊断。
### 1.1 已获得的 200-episode Eval 结果
| Camera | Method | Task Success | Grasp Success |
| --- | --- | --- | --- |
| corner | Random | 31.0% | 94.0% |
| corner | Uniform | 28.0% | 93.0% |
| corner | Ours | 24.5% | 91.5% |
| corner2 | Random | 42.5% | 97.0% |
| corner2 | Uniform | 26.0% | 67.5% |
| corner3 | Random | 34.5% | 96.0% |
| corner3 | Uniform | 37.0% | 97.0% |
| corner3 | Ours | 35.0% | 91.5% |

从最终性能看，Ours-v1 没有形成稳定优势：corner 上落后于 Random/Uniform，corner3 基本与两者打平；corner2 暂无 Ours 结果。与此同时，除 Uniform-corner2 外，多数组合 grasp success 均在 90% 以上，因此后半段控制/表示是当前非常值得关注的瓶颈。
## 2. Attention 分析工具的目的与修复过程
脚本：personal/work2/attention_fig/plot_attention.py。该工具用于分析训练后 policy-level attention / input utilization，不用于直接证明 VLM embedding separability 或 SIC coverage。它回答的是“训练后的 SmolVLA 在做 action-related computation 时，把注意力分配给哪些输入，以及图像内部看哪里”。
### 2.1 已修复的关键问题
- 图像范围：原先 uint8 [0,255] 直接转 float，已修复为 float32 [0,1]，与真实 SmolVLA prepare_images() 输入一致。
- 语言长度：原先错误使用 tokenizer.model_max_length，导致 8192 language tokens、prefix=8321；已改为 checkpoint config 的 tokenizer_max_length=48，当前 non-padding tokens=10。
- Image token grid：移除 224×224 dummy image probe；现在基于真实 prepared image（512×512）、真实 vision patch/connector 结构以及真实 image token count 推导 grid。当前 camera1/camera2 均验证为 64 tokens = 8×8。
- Attention topology：不再把 24 个 attention calls 当成 24 个 model layers。现在区分 joint_self、prefix_self、expert_cross，并映射回真实 model layer。
- Query 坐标：joint_self 的 mean_suffix 使用完整序列中的 suffix indices；expert_cross 的 query 本身就是 50 个 expert/action tokens，因此使用 local 0..49。
- 输出：可以生成每层 heatmap、attention mass、CSV/JSON，以及每模型 metadata。
- 环境清理：异常路径加入 try/finally，MetaWorld/EGL context 可以正常关闭。
### 2.2 当前 probe 的输入与 attention 结构
| Component | Length | Meaning |
| --- | --- | --- |
| Camera 1 image tokens | 64 | 8×8 |
| Camera 2 image tokens | 64 | 8×8 |
| Language tokens | 48 | 10 个 non-padding |
| State tokens | 1 | 机器人 state |
| Prefix length | 177 | 64+64+48+1 |
| Suffix length | 50 | action/expert queries |
| Total sequence | 227 | joint self-attention 时使用 |

当前捕获的 attention 不是单一形状。joint_self 通常为 227×227；cross-attention layer 中 prefix_self 为 177×177，而 expert_cross 为 50×177。当前用于 action-related 可视化时，优先选择 expert_cross；没有 expert_cross 的层才使用 joint_self。
## 3. 本次 Initial Probe 实验设置
- Seed：10042。
- Mode：probe。
- Query mode：mean_suffix。
- 分析层：0、3、7、11。
- Head：所有 heads 平均。
- 同一 seed 下，metadata 中 obj_init_pos 约为 [-0.0470, 0.6887, 0.02]，goal_pos 为 [0.1, 0.8, 0.2]。
- 共获得 8 个模型结果：corner 的 Random/Uniform/Ours；corner2 的 Random/Uniform；corner3 的 Random/Uniform/Ours。
| 重要限制 这是 initial observation + observation-conditioned probe。它可以分析模型在相同初始场景下的输入利用方式，但不能直接回答抓取后的 transport/place 为什么失败，也不能直接验证 dataset VLM embedding 的可区分性。 |
| --- |

## 4. 定量 Attention Mass 结果
最值得比较的是 layer 3 / 7 / 11 的 expert_cross，因为其 Q×K=50×177：50 个 action/expert queries 对 observation prefix 的 attention。layer 0 是 joint_self（227×227），机制不同，不应直接用绝对数值和 expert_cross 层做纵向比较。
### 4.1 Expert-cross 三层平均 Attention Mass
| Model | Camera1 | Camera2 | Language | State | Visual Total |
| --- | --- | --- | --- | --- | --- |
| Random corner | 38.1% | 34.7% | 18.2% | 9.1% | 72.8% |
| Uniform corner | 38.6% | 34.1% | 18.3% | 9.0% | 72.7% |
| Ours corner | 38.1% | 35.0% | 19.0% | 7.8% | 73.1% |
| Random corner2 | 39.3% | 32.4% | 18.7% | 9.7% | 71.7% |
| Uniform corner2 | 37.3% | 32.9% | 20.6% | 9.2% | 70.2% |
| Random corner3 | 37.3% | 33.2% | 20.1% | 9.3% | 70.5% |
| Uniform corner3 | 34.9% | 33.2% | 22.0% | 9.9% | 68.1% |
| Ours corner3 | 35.6% | 33.2% | 20.9% | 10.3% | 68.8% |

最直接的结论是：同一 camera 下 Random / Uniform / Ours 的 modality-level attention allocation 很接近。特别是在 corner 中，三种方法 expert-cross 三层平均的 visual total 都约为 73%。因此当前数据不支持“Ours 训练后明显忽略某一路 camera”这一解释。
### 4.2 随层深度的共同趋势
以 Random-corner 为例，layer 3 的 camera1/camera2/language/state 约为 36.8% / 30.2% / 17.2% / 15.8%；layer 7 约为 35.0% / 31.4% / 24.0% / 9.6%；layer 11 约为 42.4% / 42.5% / 13.2% / 1.9%。
- 较浅 expert layer：视觉、语言和 robot state 都有明显贡献。
- 中间层：state 比重下降，language 仍保持较高参与。
- 较深 expert layer：视觉占比显著提高，state attention 下降到很低。
这说明训练后的 action expert 在深层表示中主要依赖视觉输入。该趋势在不同方法和不同 camera 中都比较一致，因此工具输出具有一定结构合理性。
### 4.3 局部现象与谨慎解释
corner 的 layer 11 中，Random 的 visual total 约 84.9%，Uniform 约 84.1%，Ours 约 80.2%；其对应 task success 为 31.0%、28.0%、24.5%。corner2 中 Random layer11 visual total 约 83.1%，Uniform 约 79.1%，对应 task success 42.5% 与 26.0%。这一现象提示“深层视觉利用更强”可能与部分实验的性能相关。
但是 corner3 不支持这一简单关系：Random、Uniform、Ours 的 layer11 visual total 都在约 80% 左右，而 success 分别为 34.5%、37.0%、35.0%。因此不能得出“visual attention mass 越高，success 一定越高”的因果结论。
## 5. 当前结果可以支持什么、不能支持什么
| 编号 | 假设 | 当前判断 | 依据 |
| --- | --- | --- | --- |
| H-A | Ours 导致 policy 明显忽略某一路 camera | 当前不支持 | 同 camera 下 camera1/camera2 mass 与 Random/Uniform 非常接近 |
| H-B | 三种数据选择方法导致完全不同的 modality usage | 当前不支持 | 总体 camera/language/state 分配高度相似 |
| H-C | 不同方法看同一路图像时，关注的空间区域不同 | 尚未验证 | 需要分析 heatmap 内部 object/goal/gripper/background 分布 |
| H-D | post-grasp failure 来自抓取后 attention/representation 问题 | 无法判断 | 当前仅 initial probe，需要 rollout phase |
| H-E | Ours 的 VLM embedding 可分性/coverage 假设成立 | 无法由本实验判断 | 需要 dataset embedding separability/coverage 分析 |

## 6. 如果分析生成的 Attention 图片，应该怎么看
CSV/JSON 回答的是“attention 有多少落在 camera1/camera2/language/state”，而热力图回答的是“camera 内部到底看哪里”。如果不同方法的总 camera mass 相似，那么空间位置差异反而可能成为更关键的信息。
### 6.1 推荐比较方式：同 camera、同 seed、同 layer 横向比较
不要把不同 layer 的颜色强度直接做绝对比较。推荐固定一个 camera 与一个 layer，横向看 Random → Uniform → Ours。例如先比较 corner3 的 layer11，再比较 corner 的 layer11。
- Camera1/global：重点看 object、goal、robot/gripper、背景/桌面边缘。
- Camera2/wrist：重点看 gripper、object、gripper-object 相对关系。
- 如果总 camera mass 相同，但 Ours 的热点更多落在背景而 Random 落在 object/goal，则说明“总视觉使用量”相似但“task-relevant spatial attention”不同。
- 如果三者热点位置也高度一致，则 attention 层面很可能不是 Ours 性能差异的主要来源。
### 6.2 优先查看 Layer 11
Layer 11 是当前最值得人工检查的 expert_cross 层之一：其深层 expert representation 已经高度依赖视觉，camera1+camera2 通常约 80% 或更高。此时研究重点不再是“看不看图”，而是“图里看哪里”。
### 6.3 Layer 3 的意义
Layer 3 中 state/language 仍占较明显比例，更适合观察早期多模态融合是否因数据选择方法不同而产生偏移。
### 6.4 不应进行的比较
| 不要直接比较 Layer 0 与 Layer 3/7/11 的 attention mass Layer 0 当前是 joint_self，key space 包含 prefix+suffix（227 tokens）；layer 3/7/11 是 expert_cross，key space 只有 prefix（177 tokens）。它们属于不同 attention mechanism。正确方式是同层、同 source 横向比较 Random/Uniform/Ours。 |
| --- |

## 7. 关于“Grasp 高、最终 Success 低”的可能原因
- Post-grasp representation 不足：当前 Ours 的 episode representation 主要依赖 global 前 5 帧与 wrist 20%-70% 平均，可能更善于编码初始物体位置/抓取阶段，而没有充分表示 transport/place/release。
- Action chunk / replan 频率：SmolVLA 默认 chunk_size=50、n_action_steps=50。抓取后的精确运输与放置对闭环纠错要求更高，长时间开环执行可能放大误差。
- 数据 observation/action 时序错位：若现有训练数据仍包含 post-step observation 与 pre-step action 配对问题，粗粒度抓取可能仍能工作，但精确 transport/place 更容易受影响。
- 最终 success 条件严格：可能出现抓住后掉落、接近 goal 但不够近、没有正确 release、释放后偏出 success region 等。
当前 initial probe 不能在这些原因之间做区分。最有效的下一步是 rollout phase analysis 与 failure-stage diagnostics。
## 8. 下一步实验建议
### 8.1 Dataset / VLM Embedding 可区分性与覆盖性分析
这是验证 Ours 设计原理的核心实验，应优先于继续调 SIC 参数。建议验证三组假设：H1 embedding 是否可区分不同 task state；H2 embedding distance 是否与真实 workspace/state distance 一致；H3 Ours subset 是否显著提高整个 dataset embedding-space coverage。
- Embedding distance vs obj_init_pos XY distance：Spearman correlation + permutation test。
- 从 embedding 预测 obj_x/obj_y：Ridge / kNN regression + shuffled-label baseline。
- 真实 state kNN 与 embedding kNN 的 neighborhood overlap。
- Random / Uniform / Ours 的 fixed reference universe coverage：mean/p95/max nearest-selected distance。
- 1000 次 Random subset bootstrap，报告 Ours/Uniform 的 percentile。
- 检查 Ours 是否只是近似重新实现 workspace-position coverage。
### 8.2 Attention Rollout Phase 分析
在修正实时 object position 和 phase detection 后，使用相同模型做 initial → pre_grasp → post_grasp → pre_place 的 attention 分析。真正需要回答的是抓住之后模型是否开始关注 goal / object-goal relation，以及成功与失败 trajectory 的 attention 是否出现系统差异。
- Initial：object / goal / scene geometry。
- Pre-grasp：wrist 是否集中于 object 与 gripper-object relation。
- Post-grasp：global camera 是否增强对 goal / transport direction 的关注。
- Pre-place：是否同时利用 global goal region 与 wrist 精细位置关系。
### 8.3 Failure-stage 诊断
- first_grasp_step
- max_object_height
- post_grasp_drop
- min_object_goal_distance_after_grasp
- final_object_goal_distance
- release_detected
- success
最终将 200 episodes 自动分为：未抓到；抓到后掉落；抓到但未运输到目标；到目标附近但 release/place 失败；成功。这样能够将最终 success 的下降定位到具体阶段。
## 9. 综合结论
| 结论 1 当前 attention probe 管线已经从“粗略可视化”修复为结构上较可信的 policy-level attention 分析工具：输入范围、语言长度、image token grid、attention call topology 与 query 坐标都已与 SmolVLA 当前结构对齐。 |
| --- |

| 结论 2 Initial-state probe 中，Random / Uniform / Ours 的 modality attention distribution 高度相似，因此 Ours 当前较弱的最终性能不能简单归因于“模型忽略某一路相机”。 |
| --- |

| 结论 3 下一层问题已经从“看不看图”转为“图里看哪里”和“抓取后是否仍有正确的 task-relevant representation / closed-loop control”。Heatmap spatial pattern 与 rollout-phase analysis 因此比继续只看总 attention mass 更重要。 |
| --- |

| 结论 4 要验证 Ours 的设计原理，最关键的仍是独立完成 Dataset/VLM Embedding separability + coverage analysis。Attention probe 是 policy-level 证据链，不能替代 embedding-space 假设检验。 |
| --- |

## 附录：关键文件与输出
- 分析脚本：personal/work2/attention_fig/plot_attention.py
- 汇总表：personal/work2/attention_fig/result/attention_summary.csv
- JSON 指标：personal/work2/attention_fig/result/attention_metrics.json
- 每模型 metadata：personal/work2/attention_fig/result/<model>/seed_10042/metadata.json
- 当前 GitHub 提交中没有看到 initial_summary.png；如果需要进一步做 spatial heatmap 解释，应上传/提交 PNG 或直接提供图片。
