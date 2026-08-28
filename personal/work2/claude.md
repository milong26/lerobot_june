# MetaWorld 单任务微调 · 示范数据配方研究 — 技术方案

> **给谁看**：接下来在 `lerobot_june`(`server-dev` 分支)上继续写代码的人或 AI coding agent(如 Claude Code),以及用户本人复查计划用。
> **目标仓库**：`https://github.com/milong26/lerobot_june/tree/server-dev`
> **建议落盘位置**：把本文件放到 `personal/work2/SPEC.md`,本文档建议的新代码也都放在 `personal/work2/` 下(见第 5 节目录结构)。
> **审计方法**：克隆了 `server-dev` 分支现状 + Meta-World **v3.0.0**(`pyproject.toml` 里的 pin 版本)官方源码逐行核对。所有标注"已验证"的结论都能在你自己环境里用
> `python -c "import metaworld, os; print(os.path.dirname(metaworld.__file__))"` 找到源文件重新确认,不是我猜的。

---

## 0. 一句话结论(先看这个)

- `personal/work2/collect_metaworld_dataset.py` 整体框架是对的,但**只能靠换 seed 间接影响物体/目标位置**,做不到"按照我指定的分布/坐标精确摆放"——这是要补的核心能力。
- `personal/test/metaworld/` 的探索脚本对 39 维 `observation.environment_state` 的解读**有一处实质性错误**(把物体四元数当成"目标误差"、把帧堆叠当成"one-hot 任务向量"),而且这个错误已经被写进了 `collect_metaworld_dataset.py` 的 `ENV_STATE_DESCRIPTION`,进了正式采集脚本的输出元数据,需要修正(第 3.1 节有修正版对照表)。
- 现在的评测环境 `MetaworldEnv`(`src/lerobot/envs/metaworld.py`)每次 `reset()` 都在**整个合法初始状态空间**里做无种子随机采样。这既是好事(把你的研究问题变成一个良定义的问题),也是坏事(**不同数据配方之间对比成功率,目前跑两次会得到两个不可比的随机样本**)。补一个"固定评测集"是所有后续对比实验成立的前提,优先级最高。
- 好消息:数据集合并/切分工具(`lerobot.datasets.dataset_tools`)和单任务微调命令(ACT + `--dataset.episodes=[1500..1549]` + `--remove_features`)已经在 `服务器.md` 里跑通并验证过。新实验建立在这条已验证链路上,不需要重新发明。
- 还有一个之前没暴露的正确性问题:`collect_metaworld_dataset.py` 目前**不检查 expert policy 是否真的成功**就把 episode 存进数据集——如果以后要故意采样"困难区域"(比如工作空间边界附近),expert 有一定概率也会失败,失败的示范会污染训练集。需要加成功过滤。

下面按"审计 → 删除清单 → 新增代码规格 → 实验方法论 → 实施顺序"展开。

---

## 1. 研究问题的形式化

你的问题口语化的说法是"单任务微调时,什么样的示范数据分布能让成功率最高"。为了让这个问题可操作,先把它钉死成一个明确的优化问题:

- **评测分布是固定的、不受你控制的**:已验证 `MetaworldEnv` 用 `_freeze_rand_vec=False` 且没设 `seeded_rand_vec`,所以每次 `reset()` 都会用全局无种子随机数在该任务的 `_random_reset_space`(连续 box)里重新采样 `(obj_pos, goal_pos)`。也就是说:

  > **成功率 = E\_{(o,g) ~ Uniform(_random_reset_space)} [ policy 在初始状态 (o,g) 下成功 | policy 是在数据集 D 上训练出来的 ]**

- 你能控制的自变量,只有 **D 的构造方式**:给定同一个 expert(scripted policy)和同一个总预算(episode 数 / 帧数 / 训练 step 数),D 里的 `(obj_pos, goal_pos)` 应该长什么分布,才能让上面这个期望最大?
- 这本质上是一个**数据覆盖 vs 数据密度**的问题,跟主动学习(active learning)、课程学习、以及模仿学习里的 distributional shift / covariate shift 文献是同一类问题:如果训练分布对评测分布的支撑覆盖不够,策略在覆盖外的点上大概率会因为复合误差(compounding error)失败;如果只堆重复而不增加覆盖,大概率很快就饱和。

这个形式化决定了后面两件事必须优先做:(a) 需要能够精确控制生成数据的 `(obj_pos, goal_pos)`,而不是只能通过换 seed 间接影响;(b) 需要一个**固定的、可复现的评测集**,否则"哪种配方成功率更高"这个问题在统计上没法回答。

---

## 2. 现状代码审计(逐文件)

### 2.1 `personal/work2/collect_metaworld_dataset.py` —— 保留主干,需要扩展

**做对的地方**:
- 用 `env.obj_init_pos` / `env.goal` 直接从环境对象读取初始位置(`get_obj_pose_from_env` / `get_goal_pose_from_env`),而不是猜测 observation 的切片——**这是对的**,比 `personal/test/metaworld/` 里那批探索脚本的做法更可靠,应该保留这个模式。
- 把每个 episode 的 `obj_init_pos` / `goal_pose` / `success` 存进独立的 `episode_initial_states.json`,而不是硬塞进 dataset 的列——这个设计是合理的,轻量、不污染 LeRobotDataset schema,分析代码直接读 JSON 就行,建议保留。
- `LeRobotDataset.create(...)` 的 `features` 定义、`add_frame` / `save_episode` / `finalize` 的调用顺序都符合 LeRobot 的标准用法。

**需要修的问题**:

1. **随机化机制太弱**:`create_metaworld_env()` 只是用不同 `seed` 构造 `metaworld.MT1(task_name, seed=seed)`,取 `train_tasks[0]`,并设 `_freeze_rand_vec=True`。这样只能得到"metaworld 内部认为的第 0 个随机任务",没法指定"我要 obj 在 (0.05, 0.65, 0.02),goal 在 (-0.05, 0.85, 0.1)"这种精确坐标,更没法实现网格采样、边界采样等策略。→ 详见 4.2 节的通用状态注入方案。
2. **`ENV_STATE_DESCRIPTION` 字典是错的**(第 86–96 行),而且被原样写进每个数据集的 `episode_initial_states.json`。正确版本见 3.1 节。这不只是注释问题——它会误导后面任何读这份元数据、以为"`[7:10]` 是目标位置"的人(包括未来的 AI coder)。
3. **不检查 expert 是否成功就收录 episode**:`run_episode()` 记录了 `success`,`main()` 也统计了 `success_count`,但从头到尾没有任何地方因为 `ep_info["success"] == False` 而跳过 `dataset.add_frame` / `save_episode`。如果之后要刻意采样"难"的初始状态(比如工作空间边界),expert 会有一定概率失败,失败的示范目前会原样进训练集。
4. **没有断点续采**:`main()` 里 `if output_dir.exists(): shutil.rmtree(output_dir)` ——每次跑都会整个删掉重建。如果想做"先生成 50 个,觉得不够再追加 50 个"这种增量工作流,现在做不到。(注意这跟文件头注释"支持中断续传"实际上是矛盾的——docstring 里没有这句话,是我最早看错了,这里更正。)
5. **双相机通过 2 个独立 env + 手动同步 qpos/qvel/ctrl 实现**(`sync_env_state`,第 128–136 行),每个 episode 要建 2 个 env、做 2 次物理仿真。这能工作,但没有验证是否必要——metaworld 的渲染器有没有可能支持同一个 env 运行时切换 `camera_name`,从而只需要 1 个 env、渲染 2 次?**这个我没有验证**(需要看 `gymnasium-robotics` 包的 `MujocoRenderer` 源码,不在这次审计范围内),值得让 AI coder 花 10 分钟写个小实验确认,如果可行能省一半仿真时间。
6. **`ENV_STATE_DESCRIPTION` 之外还有一个隐藏假设**:CLI 里 `--task` 默认 `pick-place-v3`,专家策略靠字符串拼接 `f"Sawyer{args.task.replace('-', ' ').title().replace(' ', '')}Policy"` 反射查找。这个拼接规则对 `pick-place-v3` 能凑出 `SawyerPickPlaceV3Policy`(已验证 metaworld.policies 里确实叫这个名字),但换任务时要小心有没有例外命名(比如带 `-wall-` 这种复合词的任务),建议后续加一个显式的任务名→策略类映射表,而不是纯字符串拼接。

### 2.2 `personal/work2/view_obj_poses.py` —— 没问题,可以保留并小幅扩展

读 JSON 和读 dataset 两种模式都用了正确的索引(`env_state[4:7]` 对物体位置来说是对的),没有第 2.3 节说的那个 bug。可以直接保留,建议扩展:加上 `goal_pose` 的统计(现在只统计了 obj)、加一个 obj-goal 平面距离的统计、可选画个 2D 散点图。

### 2.3 `personal/test/metaworld/*.py` —— 探索任务已完成,但结论有错,建议整体归档/删除

这 5 个脚本 + `readme.md` 是"摸底 mt50 数据集长什么样"的一次性探索,`日志.md` 里 2026-07-21 的记录印证了这一点。核心问题:**它们对 39 维 `observation.environment_state` 的解读弄错了**。已验证的正确结构见 3.1 节。具体来说:

- `analyze_pick_place_episodes.py` 和 `extract_init_positions.py` 里假设 `env_state[7:10]` 是"目标位置相关信息(误差)"——实际上 `[7:11]` 是物体的四元数朝向(4 维,不是 3 维!),跟目标位置没有任何关系。这正好解释了你当时看到的"目标误差全是 1e-6 / 1e-7 量级"的现象——那其实是一个姿态四元数分量的小方差,长得像"接近 0 的误差",但语义完全不对。
- 同理假设 `[10:18]` 是"one-hot 任务编码向量(8维)"——实际上 `[18:36]` 才是"上一帧的 `[0:18]`"(帧堆叠 / frame-stack),环境的原始 observation 里根本没有 one-hot 任务向量这种东西(那是某些多任务 RL 代码库自己包一层 wrapper 才会加的东西,metaworld 环境本身不提供)。
- `analyze_pick_place_episodes.py` / `test_env_state.py` 里用 `task_index == 30` 作为 pick-place-v3 的过滤条件——`src/lerobot/envs/metaworld_config.json` 里 `pick-place-v3` 对应的是 **31**,不是 30(30 是 `pick-out-of-hole-v3`)。更根本的问题是:`metaworld_config.json` 这份映射是给 eval 环境用的,`lerobot/metaworld_mt50` 这个 HF 数据集自己的 `task_index` 编号完全可能是另一套(取决于该数据集构建时怎么枚举 task 字符串),**硬编码任何数字都不可靠**,应该按 task 描述文本匹配(`extract_init_positions.py` 其实已经这么做了,这点是对的,只是没有同时把 `test_env_state.py` 等文件里的数字过滤也改掉)。

这 5 个文件里,`check_workspace_limits.py` 值得单独说一下:它是**用大量随机 seed 经验性地估计** `_random_reset_space` 的范围,这个思路没问题,只是现在有了更直接的方法——直接读 metaworld 源码里每个任务 `__init__` 里定义的 `obj_low/obj_high/goal_low/goal_high`,是精确值,不需要靠采样估计。建议把这个脚本的"经验估计"逻辑保留下来,重构成一个**交叉验证工具**(见 4.2 节 `task_ranges.py`),而不是完全扔掉——对于你以后想扩展到的其他 49 个任务,经验采样估计是一个不用一个个去读源码的兜底手段。

`test_metaworld_first.py`(纯粹是"能不能渲染视频"的冒烟测试)和 `test_pick_place_dataset.py`(纯粹打印 mt50 数据集的 meta 信息)已经完成了它们的探索目的,建议直接删除,不需要保留。

### 2.4 `src/lerobot/envs/metaworld.py` + `configs.py` —— 官方已接好的 eval 环境,现状与关键限制

`lerobot-eval --env.type=metaworld --env.task=pick-place-v3 ...` 可以直接跑,这条链路是通的(`服务器.md` 也验证过)。但审计发现三个和你的研究强相关的限制:

1. **随机化不可控**(已在第 0/1 节说过):`_ensure_env()` 里 `env._freeze_rand_vec = False`,且从未设置 `seeded_rand_vec = True`。已验证 metaworld 源码 `_get_state_rand_vec()` 在这种组合下走的是 `else` 分支,用**全局无种子的 `np.random.uniform`**——`lerobot-eval` 传的 `--eval.seed` / `start_seed` 只影响 gym 层面的东西,**完全不影响物体/目标位置**。
2. **观测里没有 obj_pos/goal_pos,也没有暴露在 info 里**:`MetaworldEnv` 的 `observation_space` 只有 `pixels` + `agent_pos`(4 维),`step()` 返回的 `info` 只有 `{"task", "is_success", ...}` 这类字段。也就是说,**现在没有任何办法在跑完一次 eval rollout 之后,知道这一局到底是在哪个 (obj_pos, goal_pos) 下测的**——这是"分析成功率 vs 初始状态"这件事目前完全做不了的根本原因,必须补(4.4 节)。
3. **只支持单相机**(`corner2`,映射成 `"top"`)。如果按 `collect_metaworld_dataset.py` 现在的双相机(top+wrist)方式采数据、训练出来的策略需要 `observation.images.wrist`,而 `MetaworldEnv` 根本不提供这个观测键,`lerobot-eval` 会因为特征不匹配跑不起来。**建议**:先把数据生成简化成单相机,跟现有 eval 环境保持一致,把"数据配方"这个研究问题先跑通;双相机作为后续如果确实需要再补(需要同步扩展 `MetaworldEnv`)。

另外顺带确认:`src/lerobot/scripts/lerobot_eval.py` 里其实**已经有一半"把 eval rollout 存成 LeRobotDataset"的代码**(`_save_eval_dataset`,配合 `--eval.save_dataset` / `return_episode_data=True`),从文件顶部那行手动 import 测试的注释来看,这应该是当前分支上另一处"写了一半"的工作。但即使这条路径完全打通,由于第 2 点的限制,存下来的 rollout 里依然不会有 obj_pos/goal_pos——所以**不建议**把精力花在修 `_save_eval_dataset`,而是走 4.4/4.5 节更直接的方案(给 `MetaworldEnv` 加一个可查询的 `get_init_state()`,复用现成的 `env.call(...)` 模式,`rollout()` 里已经在用 `env.call("task_description")` 了)。

### 2.5 已验证、可以直接复用的既有基础设施

- `src/lerobot/datasets/dataset_tools.py` 里的 `merge_datasets` / `split_dataset` / `add_features` 是通用、成熟的工具(`personal/work1/merge_dataset.py` 已经在用 `merge_datasets`)。**合并多批生成数据、切出固定评测子集,都不需要自己写,直接调用这两个函数就行**。
- 单任务微调的训练命令已经在 `服务器.md` 里验证跑通(ACT policy,`--dataset.episodes=[1500,...,1549]` 限定只用 pick-place-v3 的 50 个 episode,`--remove_features='["observation.environment_state"]'` 防止策略直接读到特权状态作弊)。**新实验的训练命令应该照抄这条,只把 `--dataset.repo_id` / `--dataset.root` / `--dataset.episodes` 换成每次实验对应的新数据集**。

---

## 3. 已验证的 Meta-World 事实速查表

这一节是"地基",后面所有新代码都建立在这些经过源码核实的事实上。

### 3.1 `observation.environment_state`(39 维)修正版结构表

以单物体任务(如 pick-place-v3)为例,已验证来源:`metaworld/sawyer_xyz_env.py` 的 `_get_curr_obs_combined_no_goal()` / `_get_obs()`,以及 `metaworld/__init__.py` 里 `MT1`/`MT10`/`MT50` 都用 `_MT_OVERRIDE = dict(partially_observable=False)`(意味着 goal 字段是真实值,没有被置零)。

| 切片 | 含义 | 备注 |
|---|---|---|
| `[0:3]` | 末端执行器(手)位置 xyz | |
| `[3]` | 夹爪开合度(归一化) | |
| `[4:7]` | **物体 1 位置 xyz**(= `obj_pose`) | 你之前代码里 `[4:7]` 的解读是对的 |
| `[7:11]` | 物体 1 四元数朝向(4 维) | **不是"目标位置相关信息"**,这是之前脚本的错误 |
| `[11:14]` | 物体 2 位置(单物体任务里恒为 0) | |
| `[14:18]` | 物体 2 四元数(恒为 0) | |
| `[18:36]` | **上一帧的 `[0:18]` 原样重复**(frame-stack) | **不是"one-hot 任务向量"**,环境层没有任务 one-hot |
| `[36:39]` | **目标位置 xyz**(= `goal_pose`,真实值) | 这才是逐帧的 goal,不是 `[7:10]` |

> 结论:如果只是要拿 `obj_pose`/`goal_pose`,**最稳妥的方式是像 `collect_metaworld_dataset.py` 现在这样,在 reset 之后直接读 `env.obj_init_pos` / `env.goal`(或 `env._target_pos`)**,而不是去解析 39 维向量——这样完全不用管上面这张表,也不会被将来任务不同导致的维度差异坑到。这张表主要是用来**修正现有脚本里的错误注释**,以及在你确实需要"逐帧"goal(而不是 episode 级别的初始 goal)时使用。

### 3.2 `_freeze_rand_vec` / `seeded_rand_vec` 三种模式(决定"这次 reset 到底怎么随机")

已验证来源:`metaworld/sawyer_xyz_env.py` 的 `_get_state_rand_vec()`。

| `_freeze_rand_vec` | `seeded_rand_vec` | 行为 | 谁在用 |
|---|---|---|---|
| `True` | (不生效) | 返回 `_last_rand_vec`(construct 时 `set_task()` 锁定的固定值),每次 reset 完全一样 | `collect_metaworld_dataset.py` 现在的 `create_metaworld_env`(靠换 seed 换这个固定值) |
| `False` | `True` | 用 `self.np_random`(**有种子**、可复现)在 `_random_reset_space` 里连续均匀采样 | 目前代码库里没人用,但对"可复现的随机基线"很有用 |
| `False` | `False`(默认) | 用**全局无种子**的 `np.random.uniform` 采样 | `src/lerobot/envs/metaworld.py` 的 eval 环境(`MetaworldEnv._ensure_env`) |

### 3.3 pick-place-v3 的精确数值范围(已验证,来自 `metaworld/envs/sawyer_pick_place_v3.py` `__init__`)

```python
# metaworld/envs/sawyer_pick_place_v3.py
obj_low  = (-0.1, 0.6, 0.02)
obj_high = ( 0.1, 0.7, 0.02)      # 注意 z 是常数(桌面高度),只有 x,y 会变
goal_low  = (-0.1, 0.8, 0.05)
goal_high = ( 0.1, 0.9, 0.30)

self._random_reset_space = Box(
    np.hstack((obj_low, goal_low)),      # 6维: [obj_x,obj_y,obj_z, goal_x,goal_y,goal_z]
    np.hstack((obj_high, goal_high)),
)
self.goal_space = Box(goal_low, goal_high)   # 只含 goal 的 3 维 box,公开属性(无下划线)
```

`reset_model()`(已验证源码)在拿到 `_get_state_rand_vec()` 的结果后,还有一个**拒绝采样约束**:

```python
goal_pos = self._get_state_rand_vec()                      # [obj(3), goal(3)]
while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.15:  # obj 与 goal 的平面距离必须 >= 0.15
    goal_pos = self._get_state_rand_vec()
```

**这个约束很重要**:第 4.2 节要介绍的"直接指定任意 (obj, goal)"的通用注入方法,如果你指定的一对坐标平面距离 < 0.15,会撞上这个 `while` 循环——原生行为是死循环重新采样,4.2 节的方案改成了**抛出明确异常**而不是死循环或静默换一个点。

成功判定(已验证,来自 `evaluate_state`):

```python
obj_to_target = np.linalg.norm(obs[4:7] - self._target_pos)
success = float(obj_to_target <= 0.07)
```

> 顺带一提:`self.goal` 的默认值是 `np.array([0.1, 0.8, 0.2])`——这和你在 `日志.md` 里记录的"mt50 里 50 个 pick-place-v3 episode 目标位置固定为 `[0.1, 0.8, 0.2]`"**数值完全一致**。合理的猜测是官方生成 mt50 时对这个任务用了某种绕开 `reset_model()` 随机化、直接用默认 `goal` 的生成方式,但这只是猜测,**没有办法在这个沙盒环境里核实**(访问不了 huggingface.co)。不影响你自己新生成数据的正确性,只是提醒一下不要过度相信"mt50 的分布就是这个任务真实的分布"这个假设。

### 3.4 通用的"自定义初始状态注入"方法(已验证,可直接用)

已验证:全部 50 个任务的 env 文件都调用 `self._get_state_rand_vec()` 来获取初始化用的随机向量(`grep -rl _get_state_rand_vec metaworld/envs/*.py` 命中 50/50)。这意味着可以通过**猴子补丁(monkeypatch)这一个方法**,在不为每个任务单独重写 `reset_model()` 的前提下,强制指定任意 `(obj_pos, goal_pos)`:

```python
import numpy as np
import metaworld


class RandVecExhaustedError(RuntimeError):
    """任务内部的拒绝采样循环拒绝了我们指定的状态(比如 obj/goal 靠得太近),
    而不是死循环或静默换成别的随机状态——这样能明确知道"这个点对这个任务不可行"。"""


def make_env_with_fixed_state(
    task_name: str,
    rand_vec,
    seed: int = 42,
    camera_name: str = "corner2",
    render_mode: str = "rgb_array",
):
    """创建一个 reset() 被强制使用 `rand_vec` 的 Meta-World 环境。

    rand_vec: 长度/语义要匹配 env._random_reset_space 的形状。
        对 pick-place-v3 是 6 个数: [obj_x,obj_y,obj_z, goal_x,goal_y,goal_z]。
        其他任务用 `env._random_reset_space.low` / `.high` 自省(见 task_ranges.py)。

    注意:这个函数按"每个 episode 建一次性 env"设计,不要对同一个 env 反复
    调用 reset()——第二次 reset 会触发 RandVecExhaustedError(计数器是一次性的)。
    """
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode=render_mode, camera_name=camera_name)
    env.set_task(mt1.train_tasks[0])
    env._freeze_rand_vec = False  # 否则 reset_model() 会直接用旧的 _last_rand_vec,不会调用下面的补丁

    fixed_vec = np.asarray(rand_vec, dtype=np.float64)
    state = {"calls": 0}

    def _patched():
        state["calls"] += 1
        if state["calls"] > 1:
            raise RandVecExhaustedError(
                f"{task_name}: rand_vec={fixed_vec.tolist()} 被该任务 reset_model() 内部的"
                "拒绝采样条件拒绝了(它试图重新采样)。换一组离得更远/更合法的 (obj, goal)。"
            )
        return fixed_vec.copy()

    env._get_state_rand_vec = _patched
    obs, info = env.reset()
    return env, obs, info
```

这个方法的价值在于**通用**(50 个任务共享同一套机制,不需要为每个任务重写 `reset_model()`),而且**失败模式明确**(抛异常而不是死循环或静默篡改实验设计)。这是 4.2 节 `mw_common/state_injection.py` 的核心实现,建议第一批就做,做完之后"能不能精确摆放物体"这件事就有了确定性答案。

---

## 4. 需要新增/重构的代码

### 4.1 模块总览

```
generate (多种配方) ──▶ N 个 LeRobotDataset ──▶ lerobot-train (每个配方各训一个 policy)
                                                            │
make_eval_set (一次性生成,所有配方共用) ──▶ 固定评测集 ────┤
                                                            ▼
                                          run_eval_with_states (记录每局的 obj/goal + success)
                                                            │
                                                            ▼
                                              analyze_results (配方 × 成功率 的结论)
```

新代码按依赖顺序列出(后面几节各有详细说明):

1. `mw_common/obs_utils.py` —— 39 维观测解析(修正版)+ 任务描述匹配
2. `mw_common/state_injection.py` —— 3.4 节的通用状态注入器(已给出可用实现)
3. `mw_common/task_ranges.py` —— 每个任务 `_random_reset_space` 的数值范围(先做 pick-place-v3,留经验估计兜底)
4. `sampling_strategies.py` —— 候选数据配方的采样器实现
5. `generate_dataset.py`(重构 `collect_metaworld_dataset.py`)—— 支持可插拔采样策略 + 成功过滤 + 断点续采
6. `make_eval_set.py` —— 生成固定评测集(一次性,所有配方共用)
7. `src/lerobot/envs/metaworld.py` 的最小改动 —— 暴露 obj/goal + 支持注入固定评测状态
8. `run_eval_with_states.py` —— 用固定评测集跑 eval,记录逐局结果
9. `run_experiment.py` —— 编排:生成 → 训练 → 评测 → 汇总
10. `analyze_results.py` —— 统计 / 可视化

### 4.2 `mw_common/` 共享工具包

建议放在 `personal/work2/mw_common/`,`generate_dataset.py`、`run_eval_with_states.py`、`analyze_results.py` 都从这里 import,避免像现在这样同一份"观测怎么解析"的逻辑在 5 个脚本里各写一遍、还各写错一部分。

**`obs_utils.py`**:

```python
"""Meta-World 39维 observation.environment_state 的解析工具(修正版,见 SPEC.md 3.1节)。"""

ENV_STATE_LAYOUT = {
    "hand_pos": slice(0, 3),
    "gripper": slice(3, 4),
    "obj1_pos": slice(4, 7),
    "obj1_quat": slice(7, 11),
    "obj2_pos": slice(11, 14),
    "obj2_quat": slice(14, 18),
    "prev_frame_stack": slice(18, 36),   # 上一帧的 [0:18],不是任务 one-hot
    "goal_pos": slice(36, 39),           # 仅当 partially_observable=False 时是真实值(MT系列默认如此)
}

def obj_pos(env_state):
    return env_state[..., ENV_STATE_LAYOUT["obj1_pos"]]

def goal_pos(env_state):
    return env_state[..., ENV_STATE_LAYOUT["goal_pos"]]

def find_task_index(dataset_meta, task_description_substring: str) -> int:
    """按任务描述文本匹配 task_index,不要硬编码数字(不同数据集的编号可能不同)。"""
    for idx, desc in dataset_meta.tasks.items():   # 具体属性名以 LeRobotDatasetMetadata 实际实现为准,建立前先 print 确认结构
        if task_description_substring in str(desc):
            return idx
    raise ValueError(f"未找到匹配 '{task_description_substring}' 的任务")
```

> 注:`find_task_index` 里 `dataset_meta.tasks` 的具体访问方式,请先在交互式环境里 `print(dataset_meta.tasks)` 确认真实结构再定,不同 LeRobot 版本这里的字段名可能有出入,不要凭空猜。

**`state_injection.py`**:直接使用 3.4 节给出的完整实现,再加一个批量校验函数:

```python
def validate_pick_place_pair(obj_xy, goal_xy, min_planar_dist=0.15, margin=0.01):
    """在真正调用 make_env_with_fixed_state 之前,先检查是否会触发拒绝采样,
    避免批量生成时才发现某个网格点/配方点不可行。"""
    import numpy as np
    d = np.linalg.norm(np.asarray(obj_xy) - np.asarray(goal_xy))
    return d >= (min_planar_dist + margin)
```

**`task_ranges.py`**:

```python
# 已验证的精确范围(来自 metaworld 源码,逐任务补充)
KNOWN_RANGES = {
    "pick-place-v3": {
        "obj_low": (-0.1, 0.6, 0.02), "obj_high": (0.1, 0.7, 0.02),
        "goal_low": (-0.1, 0.8, 0.05), "goal_high": (0.1, 0.9, 0.30),
        "rand_vec_layout": "obj(3)+goal(3)",
        "rejection_constraint": "planar_dist(obj_xy, goal_xy) >= 0.15",
    },
    # 其他任务:先用 introspect_range() 跑一次,再人工核对是否要写死进这里
}

def introspect_range(task_name: str, seed: int = 42):
    """自省任意任务的 _random_reset_space,不用一个个去读源码。"""
    import metaworld
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array")
    env.set_task(mt1.train_tasks[0])
    space = env._random_reset_space
    return {"low": space.low.tolist(), "high": space.high.tolist(), "shape": space.shape}

def empirical_estimate_range(task_name: str, n_samples: int = 200):
    """经验性交叉验证(把 check_workspace_limits.py 的思路重构成这个通用函数):
    多次 reset 用默认随机机制,统计 obj_init_pos 实际落点范围,
    跟 introspect_range() 的解析值做交叉核对。"""
    ...  # 复用 check_workspace_limits.py 里已经写好的采样循环逻辑
```

### 4.3 `sampling_strategies.py` —— 候选数据配方

每个策略是一个函数:输入预算 N,输出 N 个 `(obj_pos, goal_pos)` 元组(必要时带上重试/校验逻辑)。具体配方列表见第 6.2 节,这里只定接口:

```python
from typing import Protocol
import numpy as np

class SamplingStrategy(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        """返回 n 个 (obj_pos_3d, goal_pos_3d) 对,且必须满足该任务的拒绝采样约束
        (调用方在拿到结果后应该用 validate_pick_place_pair 之类的函数再检查一遍)。"""
        ...
```

具体到 pick-place-v3,几个例子(完整版本留给实现阶段展开):

- `UniformRandomStrategy`:在 `obj_low/high` 和 `goal_low/high` 各自独立均匀采样,拒绝掉不满足 0.15 约束的组合,重采样直至满足——**这是跟评测分布形状一致的基线**。
- `GridStrategy(n_obj_per_axis, n_goal_per_axis)`:在 obj 的 (x,y) 和 goal 的 (x,y,z) 上各打规则网格,笛卡尔积后过滤掉不满足约束的组合。
- `BoundaryBiasedStrategy(边界宽度比例)`:采样时以一定概率从"距离 box 边界一定范围内"采样,其余仍均匀采样,构造"偏向边界"的分布。
- `DistanceStratifiedStrategy(bins)`:先算出所有合法组合的 obj-goal 平面距离范围,分层(比如"刚好过 0.15 门槛" vs "远远大于 0.15"),每层内均匀采样等量样本。

### 4.4 `MetaworldEnv` 的最小改动(`src/lerobot/envs/metaworld.py`)

目标:(a) 让固定评测集变得可能;(b) 让每局 eval 能记录用的是哪个 `(obj, goal)`。**改动应尽量小**,不影响现有 `lerobot-eval --env.type=metaworld` 的默认行为(即不传新参数时,行为跟现在完全一样,仍然是无种子全随机)。

建议的改法(伪代码,具体以现有类结构为准):

```python
class MetaworldEnv(gym.Env):
    def __init__(self, ..., fixed_states: list[np.ndarray] | None = None):
        ...
        self._fixed_states = fixed_states   # 新增,可选
        self._fixed_state_cursor = 0

    def reset(self, seed=None, options=None):
        if self._fixed_states is not None:
            # 用 3.4 节的注入机制,强制这次 reset 使用固定列表里的下一个状态,循环使用
            rand_vec = self._fixed_states[self._fixed_state_cursor % len(self._fixed_states)]
            self._fixed_state_cursor += 1
            self._env._freeze_rand_vec = False
            self._env._get_state_rand_vec = lambda: rand_vec.copy()
        # ...原有 reset 逻辑不变...
        obs, info = self._env.reset(seed=seed)
        info["obj_init_pos"] = self._env.obj_init_pos.copy() if self._env.obj_init_pos is not None else None
        info["goal_pos"] = self._env.goal.copy()
        return obs, info
```

要点:
- `info` 里新增 `obj_init_pos` / `goal_pos` 这两个键,是最小的侵入性改动,**不改变** `observation_space`,不会影响任何已训练策略的推理(策略只看 `observation`,不看 `info`)。
- `fixed_states` 传入的向量应对应第 3.4 节的 `rand_vec` 格式(此处每个任务维度不同,pick-place-v3 是 6 维)。
- 这里的 `info["obj_init_pos"] / info["goal_pos"]` 在向量化环境(`gym.vector.VectorEnv`)下要走 `final_info` 那套机制,具体参照 `lerobot_eval.py` 里 `rollout()` 现在处理 `info["final_info"]["is_success"]` 的写法保持一致。

### 4.5 `make_eval_set.py`

```python
"""生成一份固定的、所有数据配方共用的评测集,写盘成 json/npz,后续所有实验都读这一份,
保证"配方 A 的成功率 62% vs 配方 B 的成功率 58%"是在完全相同的 200 个初始状态上比出来的。"""

# 建议:直接复用 sampling_strategies.UniformRandomStrategy 生成(因为评测分布本身就是均匀分布),
# n = 200~500(第 6.3 节有具体建议),固定一个 seed,存成:
# {
#   "task": "pick-place-v3",
#   "states": [{"obj_pos": [...], "goal_pos": [...]}, ...]
# }
```

### 4.6 `run_eval_with_states.py`

跑 4.4 节改造后的 `MetaworldEnv`(传入 `fixed_states=<make_eval_set.py 生成的列表>`),对每一局记录 `(episode_index, obj_pos, goal_pos, success, sum_reward)`,输出一份 CSV/parquet。可以直接复用 `lerobot_eval.py` 里的 `rollout()` / `eval_policy()`(import 后调用,不要复制粘贴一份),额外做的事只是:reset 后从 `info` 里把 `obj_init_pos`/`goal_pos` 摘出来,和 `eval_policy()` 已经算好的 `per_episode`(含 `success`)按 `episode_ix` 对齐拼在一起——注意 `eval_policy()` 现有的 `per_episode` 已经有 `success` 字段了,不用重新计算,只是现在没人把它和 obj/goal 存在一起。

### 4.7 `generate_dataset.py`(重构 `collect_metaworld_dataset.py`)

在现有脚本基础上的改动清单:
- 用 `sampling_strategies.py` 里的策略替换"换 seed"这条路径;CLI 新增 `--strategy {uniform,grid,boundary,distance_stratified}` 及各策略专属参数。
- 用 `mw_common/state_injection.py` 替换 `create_metaworld_env` 里的随机化部分。
- 新增 `--require-success`(默认 `True`):expert 失败的 episode 不写入 `LeRobotDataset`,但仍然记录进 `episode_initial_states.json`(标注 `success: false`),用于后续分析"哪些区域连 expert 都难"。**不要**因为失败就静默换一个更容易的点重试——那样会悄悄改变配方本来设计的分布,应该如实记录"这个点失败了",让 4.9 节的分析去解释。
- 新增断点续采:`main()` 里改成"若 `output_dir` 已存在,尝试 `LeRobotDataset(root=output_dir)` 加载,读取已有 episode 数,追加生成剩余的",而不是无条件 `shutil.rmtree`。
- 把 `ENV_STATE_DESCRIPTION` 换成从 `mw_common.obs_utils.ENV_STATE_LAYOUT` 生成,不要留一份重复且错误的拷贝。
- (可选,视 2.4 节第 5 点小实验结果而定)如果确认能省一个 env,把双相机改成单 env 切换 `camera_name`。

### 4.8 `run_experiment.py`

编排脚本,大致逻辑:

```python
RECIPES = {
    "uniform_50": dict(strategy="uniform", n_episodes=50),
    "grid_50":    dict(strategy="grid", n_episodes=50),
    "boundary_50":dict(strategy="boundary", n_episodes=50),
    # ... 见 6.2 节完整配方表
}

for name, cfg in RECIPES.items():
    dataset_root = f"personal/work2/generated/{name}"
    if not Path(dataset_root).exists():
        subprocess.run(["python", "generate_dataset.py", "--strategy", cfg["strategy"],
                         "--num-episodes", str(cfg["n_episodes"]), "--output-dir", dataset_root, ...])
    output_dir = f"outputs/exp_{name}"
    if not Path(output_dir).exists():
        subprocess.run(["lerobot-train", f"--dataset.root={dataset_root}", ...,
                         f"--output_dir={output_dir}", "--steps=20000",
                         '--remove_features=["observation.environment_state"]'])
    subprocess.run(["python", "run_eval_with_states.py", "--policy-path", f"{output_dir}/checkpoints/last",
                     "--eval-set", "personal/work2/fixed_eval_set.json",
                     "--out-csv", f"results/{name}.csv"])
```

真正实现时,训练/评测这两步耗时都很长,建议做成可以按 recipe 单独重跑、每步都先检查产物是否已存在(避免算力浪费),并且把每一步的 stdout/stderr 落盘到日志文件(参照 `日志.md`、`服务器.md` 里已经在用的 `train_log/*.log` 惯例)。

### 4.9 `analyze_results.py`

输入是 `run_experiment.py` 产出的一堆 `results/{recipe}.csv`(每行:episode 的 obj_pos/goal_pos/success)以及各配方数据集的 `episode_initial_states.json`(训练集分布)。建议产出:

- 配方 × 固定评测集成功率的汇总表(核心结论)。
- 对每个配方:评测集里每个失败点,到"最近的训练集样本"的欧氏距离——验证"离训练集支撑越远,越容易失败"这个假设,画成"距离 vs 成功率"的散点/分箱曲线。
- obj_pos(或 goal_pos)在 2D 平面上的成功率热力图(因为 z 通常是常数,天然可以画 2D)。
- 如果做了 6.3 节的采样量扫描:N 对成功率的曲线,观察饱和点。

---

## 5. 建议的目录结构

```
personal/work2/
  SPEC.md                        (本文档)
  readme.md                      (更新,见 8.1 节的重写建议)
  mw_common/
    __init__.py
    obs_utils.py
    state_injection.py
    task_ranges.py
  sampling_strategies.py
  generate_dataset.py            (重构自 collect_metaworld_dataset.py)
  make_eval_set.py
  run_eval_with_states.py
  run_experiment.py
  analyze_results.py
  view_obj_poses.py              (保留,小幅扩展)
  fixed_eval_set.json            (make_eval_set.py 的产物,提交到仓库以保证评测可复现)
  generated/                     (各配方生成的数据集,建议 .gitignore)
  results/                       (每个配方的 eval CSV + 汇总图表)

src/lerobot/envs/metaworld.py    (4.4 节的最小改动)
```

`personal/test/metaworld/` 建议整体归档(比如移到 `personal/test/metaworld_archive/` 或直接删除,探索目的已达成,结论已经吸收进本文档),不建议留着继续被引用,因为里面的注释是错的。

---

## 6. 实验方法论

### 6.1 需要控制的混淆变量

在比较任何两个数据配方之前,先确认这些东西是"固定不变"的,否则结论没法解释是配方本身的效果还是别的因素造成的:

- **训练 step 数 / batch size 完全相同**(`--steps` 固定,不要按数据集大小自动换算,这样"配方效果"和"训练时长"这两个变量才不会纠缠在一起)。
- **总 episode 数(或至少数量级)相同**——除非你是在做 6.3 节专门的"数据量扫描"实验。
- **expert policy 相同、`max_steps` 相同**。
- **评测用同一份固定评测集**(4.5 节),不同配方训出来的策略都在这 200~500 个点上测,不要各测各的随机 200 局。
- **episode 长度的隐藏偏差**:obj 和 goal 距离越远,expert 走的步数天然越多,意味着"每 episode 的帧数"会系统性地随配方(尤其是边界采样、距离分层采样)变化。如果你关心"总训练帧数"是否公平,除了对齐 episode 数,还应该在报告里同时列出每个配方数据集的总帧数,不要只看 episode 数。
- **训练本身的随机性**:同一份数据、同一套超参数,不同的训练随机种子跑出来的最终策略成功率也会有方差。如果算力允许,每个配方至少训 2~3 个种子,报告均值和方差;如果算力紧张,先接受"单种子"的结果作为初筛,但在结论里明确注明这一点,不要把单次训练的差异当成配方本身的效应。

### 6.2 候选数据配方(第一轮实验矩阵)

| 配方 | 定义 | 想验证的假设 |
|---|---|---|
| `uniform`(基线) | obj、goal 各自独立均匀采样,拒绝不满足 0.15 约束的组合 | 跟评测分布形状一致,是最直接的参照系 |
| `grid` | obj (x,y) 和 goal (x,y,z) 打规则网格,过滤非法组合 | 去掉采样噪声后,规则覆盖是否比随机覆盖更有效 |
| `boundary` | 以一定概率偏向 box 边界采样,其余均匀 | 策略在评测分布的"边角"上更容易失败,训练时特意补边角是否能提升整体成功率 |
| `distance_stratified` | 按 obj-goal 平面距离分层,各层等量采样 | 任务难度是否与 obj-goal 距离相关,针对性覆盖"难层"是否有帮助 |
| `narrow_dense` | 只在 box 中心一小块区域采样,同样的 episode 数在小范围内重复 | 用来对照"覆盖"和"密度"——如果这个配方在固定评测集上明显更差,说明覆盖比单纯的数据量更重要 |

第一轮建议:每个配方固定 `n_episodes=50`(跟你现有的 mt50 单任务切片大小一致,便于横向比较),ACT policy,固定 steps(比如 20000,或参照 `服务器.md` 里已验证过的训练时长设置),都在同一份 `n=200` 的固定评测集上测。

### 6.3 第二轮:数据量扫描(可选,在第一轮选出的最优配方上做)

固定第一轮里表现最好的配方,扫 `n_episodes ∈ {10, 25, 50, 100, 200}`,画"成功率 vs 数据量"曲线,找饱和点——这个问题("到底需要多少条示范才够")本身也是你研究问题的一部分,值得单独报告。

### 6.4 指标与解读框架

- **主指标**:固定评测集上的成功率(与 `lerobot-eval` 的 `pc_success` 定义一致)。
- **诊断指标**:失败点到最近训练样本的距离 vs 是否成功——如果这条曲线很陡(离训练支撑一远就迅速失败),说明当前问题主要是"覆盖不够"导致的外推失败,应该优先选覆盖广的配方;如果这条曲线很平(离训练支撑远近对成功率影响不大),说明问题可能出在别的地方(比如任务本身对初始状态不敏感,或者是训练量/策略容量的问题),继续加覆盖可能收益有限。
- 数据量扫描如果很快饱和(比如 25 条就跟 200 条差不多),说明"多样性"比"数量"重要;如果一直没饱和,说明还是数据量本身是瓶颈。

### 6.5 从 pick-place-v3 推广到其他任务时要注意

- 不是所有任务的 `_random_reset_space` 都是 6 维(obj+goal);有些任务可能只随机化 obj(goal 固定),有些可能维度语义完全不同(比如带门、带阀门旋钮的任务,状态是角度而不是位置)。**每换一个新任务,先跑 `task_ranges.introspect_range()` 确认维度语义**,不要假设都跟 pick-place-v3 一样。
- 不是所有任务都有 pick-place-v3/push-v3 这种"拒绝采样 while 循环";有约束的任务在批量生成前要先做校验(参照 `validate_pick_place_pair`),没约束的任务可以跳过这一步。
- expert policy 的可靠性因任务而异,批量生成前建议先用 `--require-success` 跑一遍,看 expert 在该任务全域的基础成功率,如果 expert 自己成功率就不高,数据配方研究的结论会被 expert 质量混淆,需要在报告里说明。

---

## 7. 建议的实施顺序(分阶段)

1. **Phase 0 · 修 bug,不加新功能**(半天量级):修正 `ENV_STATE_DESCRIPTION`;把 `task_index` 硬编码换成文本匹配;归档 `personal/test/metaworld/` 里已完成探索目的的脚本。
2. **Phase 1 · 打通"精确指定初始状态"这条关键路径**:实现 `mw_common/`(3.4 节的注入器是现成的),写一个 smoke test——手选几组 `(obj, goal)`,验证 expert 确实能在指定状态下完成任务并达到预期成功率。**这是全部后续工作的技术前提,建议第一个做**,做完就知道"精确摆放物体"在这套代码里到底可不可行、稳不稳。
3. **Phase 2 · 固定评测集 + eval 环境扩展**:`make_eval_set.py` + `MetaworldEnv` 的 4.4 节改动 + `run_eval_with_states.py`。这是"任何配方对比"能成立的前提。
4. **Phase 3 · 数据生成重构**:`sampling_strategies.py` + `generate_dataset.py`。
5. **Phase 4 · 编排 + 分析,跑第一轮小规模实验**:`run_experiment.py` + `analyze_results.py`,先用 6.2 节的 5 个配方、每个 50 episode 跑一遍,得到第一版结论。
6. **Phase 5 · 按需扩展**:双相机对齐 eval 环境(如果第一轮证明数据配方研究确实有效、值得投入更多真实感);扩展到其他任务;换其他 policy 架构复验结论是否稳健;第一轮结果好的配方做 6.3 节的数据量扫描 + 多训练种子确认。

---

## 8. 开放问题 / 需要人工确认的假设

- mt50 里 pick-place-v3 的 goal 是否真的对全部 episode 固定为 `[0.1, 0.8, 0.2]`、obj 是否真的是"49 个不同 seed 各自的 `train_tasks[0]`"——只能通过实际访问/重新解析 `lerobot/metaworld_mt50` 数据集本身核实(这次审计的沙盒环境访问不了 huggingface.co)。建议第一步用修正后的 `extract_init_positions.py` 逻辑重新跑一遍确认。
- 双相机渲染是否可以用同一个 env 实例切换 `camera_name` 而不用建 2 个 env——未验证,需要看 `gymnasium-robotics` 的 `MujocoRenderer` 源码或直接写个小实验试。
- 训练脚本目前只有 ACT 被验证过完整跑通;数据配方的结论是否对策略架构稳健,建议第一轮先只用 ACT 跑出初步结论,如果结果显著,再挑 1-2 个其他策略(比如 diffusion policy)复验,不要一开始就多策略 × 多配方全排列,组合数会爆炸。
- 每个 (配方 × 训练) 组合都要真实跑一次训练,算力/时间成本不低,建议先用小规模(比如更少的 steps、更小的 batch)做"预实验"筛掉明显不 work 的配方,再对少数候选配方做更大规模的确认实验。

### 8.1 `personal/work2/readme.md` 建议改写内容

现在的 readme 只有两行("目的是:单任务中示范数据的分布的影响"+"验证 metaworld,测试 pick-place-v3 任务")。建议改写为指向本文档,并加一句现状说明,例如:

> 目的:研究单任务模仿学习中,示范数据的初始状态分布(obj_pose / goal_pose)如何影响微调后策略的成功率。
> 详细方案见 `SPEC.md`。现状:`collect_metaworld_dataset.py` 能采集数据但只能靠 seed 间接控制初始状态;`mw_common/` 提供了精确指定 (obj_pos, goal_pos) 的通用方法;固定评测集机制(`make_eval_set.py` + `run_eval_with_states.py`)是保证不同数据配方之间成功率可比的前提,需要优先实现。