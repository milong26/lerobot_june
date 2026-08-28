"""
模拟SmolVLA训练/推理场景：使用VLM处理数据集中的帧，逐帧比较两个episode

要求：
1. 从数据集 lerobot root=personal/work2/dataset_view/pickplacev3 中读取episode=7和episode=94
2. 引入VLM模型（SmolVLM2-500M-Video-Instruct），参考SmolVLA的使用方式
3. 所有图像（top+wrist相机）和状态都进入计算，模拟训练场景
4. 逐帧比较两个episode（以较短的episode为准，丢弃多余的帧）
5. 分析VLM特征随时间的变化趋势

结果：
episode230：总帧数61，  物体位置 (索引4:7):X: 0.0002，Y: 0.6072，Z: 0.0200
episode 459：总帧数64，  物体位置 (索引4:7):X: 0.024，Y: 0.6113，Z: 0.0200
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from lerobot.datasets import LeRobotDataset

# VLM模型配置
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TASK_PROMPT = "pick and place the object"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_vlm_model():
    """加载VLM模型"""
    print(f"\n正在加载VLM模型: {VLM_MODEL_ID}")
    print(f"设备: {DEVICE}")
    
    from transformers import AutoModel, AutoProcessor
    
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = AutoModel.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"VLM模型加载完成")
    print(f"  vision_model 类型: {type(model.vision_model).__name__}")
    print(f"  connector 类型: {type(model.connector).__name__}")
    
    return model, processor


def tensor_to_pil(tensor_img):
    """将tensor图像转换为PIL Image"""
    if hasattr(tensor_img, 'numpy'):
        np_img = tensor_img.numpy()
    else:
        np_img = tensor_img
    
    if np_img.ndim == 3 and np_img.shape[0] < np_img.shape[-1]:
        np_img = np.transpose(np_img, (1, 2, 0))
    
    if np_img.max() <= 1.0:
        pil_img = Image.fromarray((np_img * 255).astype(np.uint8))
    else:
        pil_img = Image.fromarray(np_img.astype(np.uint8))
    
    return pil_img


def extract_frame_features_fast(model, processor, images_list, state, task_text):
    """
    快速提取帧特征（只提取connector输出，不做完整VLM前向传播）
    用于逐帧比较时提高效率
    """
    vlm_model = model
    image_token = processor.tokenizer.image_token if hasattr(processor.tokenizer, 'image_token') else "<image>"
    text = f"{image_token}\n{task_text}"
    
    connector_features = []
    vision_features = []
    pixel_values_list = []  # 保存processor处理后的pixel_values
    
    with torch.no_grad():
        for pil_img in images_list:
            inputs = processor(
                text=text,
                images=[pil_img],
                return_tensors="pt",
            ).to(DEVICE)
            
            pixel_values = inputs["pixel_values"]
            if pixel_values.ndim == 5:
                pixel_values = pixel_values[:, 0]
            pixel_values = pixel_values.to(dtype=vlm_model.vision_model.dtype)
            
            # 保存pixel_values用于对比
            pixel_values_list.append(pixel_values.cpu())
            
            vision_output = vlm_model.vision_model(pixel_values=pixel_values)
            vision_hidden = vision_output.last_hidden_state
            
            connector_output = vlm_model.connector(vision_hidden)
            
            connector_features.append(connector_output.squeeze(0).cpu())
            vision_features.append(vision_hidden.squeeze(0).cpu())
        
        all_connector = torch.cat(connector_features, dim=0)
    
    state_tensor = None
    if state is not None:
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.from_numpy(state)
        else:
            state_tensor = state
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.float().cpu()
    
    return {
        "all_connector": all_connector,
        "vision_features": vision_features,
        "connector_features": connector_features,
        "pixel_values": pixel_values_list,  # 新增
        "state": state_tensor,
    }


def check_pixel_values(pixel_values_7, pixel_values_94, image_keys, frame_idx):
    """检查processor处理后的pixel_values是否相同"""
    print(f"Frame {frame_idx} - Processor输出检查 (pixel_values)")
    
    all_identical = True
    
    for cam_idx, (pv_7, pv_94) in enumerate(zip(pixel_values_7, pixel_values_94)):
        cam_name = image_keys[cam_idx] if cam_idx < len(image_keys) else f"cam{cam_idx}"
        
        # 检查是否完全相同
        is_identical = torch.equal(pv_7, pv_94)
        
        # 计算差异
        if not is_identical:
            all_identical = False
            diff = torch.abs(pv_7 - pv_94)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            non_zero_diff = (diff > 0).sum().item()
            total_elements = pv_7.numel()
            
            print(f"\n  相机 {cam_name}:")
            print(f"    完全相同: False")
            print(f"    维度: {pv_7.shape}")
            print(f"    最大差异: {max_diff:.6f}")
            print(f"    平均差异: {mean_diff:.6f}")
            print(f"    不同元素数: {non_zero_diff}/{total_elements} ({non_zero_diff/total_elements*100:.2f}%)")
            print(f"    Episode7范围: [{pv_7.min():.4f}, {pv_7.max():.4f}]")
            print(f"    Episode94范围: [{pv_94.min():.4f}, {pv_94.max():.4f}]")
        else:
            print(f"\n  相机 {cam_name}:")
            print(f"    完全相同: True")
            print(f"    维度: {pv_7.shape}")
            print(f"    范围: [{pv_7.min():.4f}, {pv_7.max():.4f}]")
    
    return all_identical


def check_raw_frames(dataset, from_idx_7, from_idx_94, image_keys, num_frames=5):
    """检查原始帧数据的差异"""
    print(f"原始帧数据检查（前{num_frames}帧）")
    
    for frame_idx in range(num_frames):
        idx_7 = int(from_idx_7) + frame_idx
        idx_94 = int(from_idx_94) + frame_idx
        
        print(f"Frame {frame_idx} - 原始数据对比")
        
        frame_7 = dataset[idx_7]
        frame_94 = dataset[idx_94]
        
        for cam_key in image_keys:
            raw_7 = frame_7[cam_key]
            raw_94 = frame_94[cam_key]
            
            if hasattr(raw_7, 'numpy'):
                raw_7 = raw_7.numpy()
            if hasattr(raw_94, 'numpy'):
                raw_94 = raw_94.numpy()
            
            # 转换为tensor
            t_7 = torch.from_numpy(raw_7)
            t_94 = torch.from_numpy(raw_94)
            
            # 检查是否相同
            is_identical = torch.equal(t_7, t_94)
            
            if is_identical:
                print(f"\n  {cam_key}:")
                print(f"    完全相同: True")
                print(f"    维度: {t_7.shape}")
                print(f"    数值范围: [{t_7.min():.4f}, {t_7.max():.4f}]")
                print(f"    数据类型: {t_7.dtype}")
            else:
                diff = torch.abs(t_7 - t_94)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                non_zero_diff = (diff > 0).sum().item()
                total_elements = t_7.numel()
                
                print(f"\n  {cam_key}:")
                print(f"    完全相同: False")
                print(f"    维度: {t_7.shape}")
                print(f"    最大差异: {max_diff:.4f}")
                print(f"    平均差异: {mean_diff:.4f}")
                print(f"    不同元素数: {non_zero_diff}/{total_elements} ({non_zero_diff/total_elements*100:.2f}%)")
                print(f"    Episode230范围: [{t_7.min():.4f}, {t_7.max():.4f}]")
                print(f"    Episode459范围: [{t_94.min():.4f}, {t_94.max():.4f}]")


def compare_frame_pair(features_7, features_94, frame_idx):
    """比较单帧对的特征差异"""
    concat_7 = features_7["all_connector"]
    concat_94 = features_94["all_connector"]
    
    concat_l2 = torch.norm(concat_7 - concat_94, p=2).item()
    concat_cos_sim = torch.nn.functional.cosine_similarity(
        concat_7.mean(dim=0), concat_94.mean(dim=0), dim=0
    ).item()
    concat_mae = torch.abs(concat_7 - concat_94).mean().item()
    
    # Vision encoder对比
    vision_l2s = []
    for vis_7, vis_94 in zip(features_7["vision_features"], features_94["vision_features"]):
        vision_l2s.append(torch.norm(vis_7 - vis_94, p=2).item())
    
    # Connector对比
    connector_l2s = []
    for conn_7, conn_94 in zip(features_7["connector_features"], features_94["connector_features"]):
        connector_l2s.append(torch.norm(conn_7 - conn_94, p=2).item())
    
    state_7 = features_7["state"]
    state_94 = features_94["state"]
    state_l2 = torch.norm(state_7 - state_94, p=2).item()
    state_mae = torch.abs(state_7 - state_94).mean().item()
    
    return {
        "frame_idx": frame_idx,
        "connector_l2": concat_l2,
        "connector_cos_sim": concat_cos_sim,
        "connector_mae": concat_mae,
        "vision_l2s": vision_l2s,
        "connector_l2s": connector_l2s,
        "state_l2": state_l2,
        "state_mae": state_mae,
    }


def print_detailed_comparison(features_7, features_94, frame_idx, result=None):
    """打印详细的单帧对比"""
    print(f"\n{'─'*80}")
    print(f"Frame {frame_idx} 详细对比")
    print(f"{'─'*80}")
    
    # 逐相机对比
    cam_names = ["top", "wrist"]
    for cam_idx, (conn_7, conn_94) in enumerate(zip(features_7["connector_features"], features_94["connector_features"])):
        cam_name = cam_names[cam_idx] if cam_idx < len(cam_names) else f"cam{cam_idx}"
        conn_l2 = torch.norm(conn_7 - conn_94, p=2).item()
        conn_cos_sim = torch.nn.functional.cosine_similarity(
            conn_7.mean(dim=0), conn_94.mean(dim=0), dim=0
        ).item()
        conn_mae = torch.abs(conn_7 - conn_94).mean().item()
        
        # Vision encoder对比
        vis_7 = features_7["vision_features"][cam_idx]
        vis_94 = features_94["vision_features"][cam_idx]
        vis_l2 = torch.norm(vis_7 - vis_94, p=2).item()
        
        # 计算放大率
        if vis_l2 > 0:
            amp_ratio = conn_l2 / vis_l2
        else:
            amp_ratio = float('inf') if conn_l2 > 0 else 0
        
        print(f"\n  相机 {cam_name}:")
        print(f"    Vision Encoder: L2={vis_l2:.4f}")
        print(f"    Connector:      L2={conn_l2:.4f}, Cosine={conn_cos_sim:.6f}, MAE={conn_mae:.6f}")
        print(f"    放大率 (Connector/Vision): {amp_ratio:.4f}x")
    
    # 整体对比
    concat_7 = features_7["all_connector"]
    concat_94 = features_94["all_connector"]
    concat_l2 = torch.norm(concat_7 - concat_94, p=2).item()
    concat_cos_sim = torch.nn.functional.cosine_similarity(
        concat_7.mean(dim=0), concat_94.mean(dim=0), dim=0
    ).item()
    concat_mae = torch.abs(concat_7 - concat_94).mean().item()
    
    # Vision整体
    total_vis_l2 = torch.norm(
        torch.cat(features_7["vision_features"], dim=0) - torch.cat(features_94["vision_features"], dim=0),
        p=2
    ).item()
    
    if total_vis_l2 > 0:
        total_amp_ratio = concat_l2 / total_vis_l2
    else:
        total_amp_ratio = float('inf') if concat_l2 > 0 else 0
    
    print(f"\n  整体Concat:")
    print(f"    Vision Encoder: L2={total_vis_l2:.4f}")
    print(f"    Connector:      L2={concat_l2:.4f}, Cosine={concat_cos_sim:.6f}, MAE={concat_mae:.6f}")
    print(f"    放大率 (Connector/Vision): {total_amp_ratio:.4f}x")
    
    # State对比
    state_7 = features_7["state"]
    state_94 = features_94["state"]
    state_l2 = torch.norm(state_7 - state_94, p=2).item()
    state_mae = torch.abs(state_7 - state_94).mean().item()
    
    print(f"\n  State: L2={state_l2:.6f}, MAE={state_mae:.6f}")


def print_environment_state(dataset, episode_indices):
    """打印指定episode的第一帧environment_state"""
    print(f"指定Episode的第一帧 environment_state")
    
    for ep_idx in episode_indices:
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        first_frame = dataset[int(from_idx)]
        
        print(f"Episode {ep_idx} (帧索引: {from_idx})")
        print(f"{'─'*80}")
        
        if "observation.environment_state" in first_frame:
            env_state = first_frame["observation.environment_state"]
            if hasattr(env_state, 'numpy'):
                env_state = env_state.numpy()
            
            # 根据已知的状态结构打印各部分含义
            print(f"\n  状态结构解析:")
            if len(env_state) >= 3:
                print(f"    [0:3]   末端执行器(手)位置 xyz: ({env_state[0]:.4f}, {env_state[1]:.4f}, {env_state[2]:.4f})")
            if len(env_state) >= 4:
                print(f"    [3:4]   夹爪开合度(归一化): {env_state[3]:.4f}")
            if len(env_state) >= 7:
                print(f"    [4:7]   物体1位置 xyz: ({env_state[4]:.4f}, {env_state[5]:.4f}, {env_state[6]:.4f})")
            if len(env_state) >= 11:
                print(f"    [7:11]  物体1四元数朝向: ({env_state[7]:.4f}, {env_state[8]:.4f}, {env_state[9]:.4f}, {env_state[10]:.4f})")
            if len(env_state) >= 14:
                print(f"    [11:14] 物体2位置(单物体任务中恒为0): ({env_state[11]:.4f}, {env_state[12]:.4f}, {env_state[13]:.4f})")
            if len(env_state) >= 18:
                print(f"    [14:18] 物体2四元数(恒为0): ({env_state[14]:.4f}, {env_state[15]:.4f}, {env_state[16]:.4f}, {env_state[17]:.4f})")
            if len(env_state) >= 36:
                print(f"    [18:36] 上一帧[0:18]原样重复(frame-stack)")
            if len(env_state) >= 39:
                print(f"    [36:39] 目标位置 xyz: ({env_state[36]:.4f}, {env_state[37]:.4f}, {env_state[38]:.4f})")
        else:
            print(f"  警告: 未找到 observation.environment_state")
            print(f"  可用的keys: {list(first_frame.keys())}")


def print_environment_state(dataset, episode_indices):
    """打印指定episode的第一帧environment_state"""
    print(f"\n{'='*80}")
    print(f"指定Episode的第一帧 environment_state")
    print(f"{'='*80}")
    
    for ep_idx in episode_indices:
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        first_frame = dataset[int(from_idx)]
        
        print(f"\n{'─'*80}")
        print(f"Episode {ep_idx} (帧索引: {from_idx})")
        print(f"{'─'*80}")
        
        if "observation.environment_state" in first_frame:
            env_state = first_frame["observation.environment_state"]
            if hasattr(env_state, 'numpy'):
                env_state = env_state.numpy()
            
            print(f"  维度: {env_state.shape}")
            print(f"  完整数值:")
            for i, val in enumerate(env_state):
                print(f"    [{i:2d}]: {val:.6f}")
            
            # 根据已知的状态结构打印各部分含义
            print(f"\n  状态结构解析:")
            if len(env_state) >= 3:
                print(f"    [0:3]   末端执行器(手)位置 xyz: ({env_state[0]:.4f}, {env_state[1]:.4f}, {env_state[2]:.4f})")
            if len(env_state) >= 4:
                print(f"    [3:4]   夹爪开合度(归一化): {env_state[3]:.4f}")
            if len(env_state) >= 7:
                print(f"    [4:7]   物体1位置 xyz: ({env_state[4]:.4f}, {env_state[5]:.4f}, {env_state[6]:.4f})")
            if len(env_state) >= 11:
                print(f"    [7:11]  物体1四元数朝向: ({env_state[7]:.4f}, {env_state[8]:.4f}, {env_state[9]:.4f}, {env_state[10]:.4f})")
            if len(env_state) >= 14:
                print(f"    [11:14] 物体2位置(单物体任务中恒为0): ({env_state[11]:.4f}, {env_state[12]:.4f}, {env_state[13]:.4f})")
            if len(env_state) >= 18:
                print(f"    [14:18] 物体2四元数(恒为0): ({env_state[14]:.4f}, {env_state[15]:.4f}, {env_state[16]:.4f}, {env_state[17]:.4f})")
            if len(env_state) >= 36:
                print(f"    [18:36] 上一帧[0:18]原样重复(frame-stack)")
            if len(env_state) >= 39:
                print(f"    [36:39] 目标位置 xyz: ({env_state[36]:.4f}, {env_state[37]:.4f}, {env_state[38]:.4f})")
        else:
            print(f"  警告: 未找到 observation.environment_state")
            print(f"  可用的keys: {list(first_frame.keys())}")


def main():
    dataset_root = "personal/work2/dataset_view/pick_place_corner3"
    print(f"逐帧比较 Episode 1 和 Episode 250，使用数据集{dataset_root}")
    
    dataset = LeRobotDataset(
        repo_id="work2/pick_place_corner3",
        root=dataset_root
    )
    
    print(f"数据集加载完成!")
    print(f"总Episode数: {dataset.num_episodes}")
    print(f"总帧数: {dataset.num_frames}")
    print(f"FPS: {dataset.fps}")
    
    # 获取两个episode的帧范围
    from_idx_7 = dataset.meta.episodes["dataset_from_index"][1]
    to_idx_7 = dataset.meta.episodes["dataset_to_index"][1]
    from_idx_94 = dataset.meta.episodes["dataset_from_index"][250]
    to_idx_94 = dataset.meta.episodes["dataset_to_index"][250]
    
    num_frames_7 = to_idx_7 - from_idx_7
    num_frames_94 = to_idx_94 - from_idx_94
    
    print(f"\nEpisode 230: 帧范围 [{from_idx_7}, {to_idx_7}), 共 {num_frames_7} 帧")
    print(f"Episode 459: 帧范围 [{from_idx_94}, {to_idx_94}), 共 {num_frames_94} 帧")
    
    # 以较短的episode为准
    num_compare_frames = min(num_frames_7, num_frames_94)
    num_discarded = max(num_frames_7, num_frames_94) - num_compare_frames
    
    if num_discarded > 0:
        longer_ep = 150 if num_frames_94 > num_frames_7 else 1
        print(f"\n将以较短的episode为准，比较前 {num_compare_frames} 帧")
        print(f"Episode {longer_ep} 的最后 {num_discarded} 帧将被丢弃")
    
    # 获取图像keys
    sample_frame = dataset[int(from_idx_7)]
    image_keys = [k for k in sample_frame.keys() if "image" in k]
    print(f"\n图像keys: {image_keys}")
    
    # 打印指定episode的environment_state
    print_environment_state(dataset, [1, 250])
    
    # 检查原始帧数据
    check_raw_frames(dataset, from_idx_7, from_idx_94, image_keys, num_frames=5)
    
    # 加载VLM模型
    model, processor = load_vlm_model()
    
    # 逐帧提取特征并比较
    print(f"# 开始逐帧比较（共 {num_compare_frames} 帧）")
    
    all_results = []
    pixel_values_check_done = False
    
    for frame_idx in range(num_compare_frames):
        idx_7 = int(from_idx_7) + frame_idx
        idx_94 = int(from_idx_94) + frame_idx
        
        # 获取两帧数据
        frame_7 = dataset[idx_7]
        frame_94 = dataset[idx_94]
        
        # 转换图像
        images_7 = [tensor_to_pil(frame_7[k]) for k in image_keys]
        images_94 = [tensor_to_pil(frame_94[k]) for k in image_keys]
        
        # 获取state
        state_7 = frame_7.get("observation.state", None)
        state_94 = frame_94.get("observation.state", None)
        if state_7 is not None and hasattr(state_7, 'numpy'):
            state_7 = state_7.numpy()
        if state_94 is not None and hasattr(state_94, 'numpy'):
            state_94 = state_94.numpy()
        
        # 提取特征
        if frame_idx % 10 == 0:
            print(f"\n处理帧 {frame_idx}/{num_compare_frames}...")
        
        features_7 = extract_frame_features_fast(model, processor, images_7, state_7, TASK_PROMPT)
        features_94 = extract_frame_features_fast(model, processor, images_94, state_94, TASK_PROMPT)
        
        # 检查前3帧的pixel_values
        if frame_idx < 3:
            check_pixel_values(features_7["pixel_values"], features_94["pixel_values"], image_keys, frame_idx)
        
        # 比较
        result = compare_frame_pair(features_7, features_94, frame_idx)
        all_results.append(result)
        
        # 打印关键帧的详细信息（首帧、中间帧、最后帧）
        if frame_idx == 0 or frame_idx == num_compare_frames // 2 or frame_idx == num_compare_frames - 1:
            print_detailed_comparison(features_7, features_94, frame_idx, result)
    
    # 打印所有帧的汇总
    print(f"所有帧对比汇总（共 {num_compare_frames} 帧）")
    
    print(f"\n{'Frame':>5} | {'Vis Top':>10} | {'Vis Wrist':>10} | {'Conn Top':>10} | {'Conn Wrist':>10} | {'Amp Top':>8} | {'Amp Wrist':>10} | {'State L2':>10}")
    print(f"{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    
    for r in all_results:
        vis_top = r["vision_l2s"][0] if len(r["vision_l2s"]) > 0 else 0
        vis_wrist = r["vision_l2s"][1] if len(r["vision_l2s"]) > 1 else 0
        conn_top = r["connector_l2s"][0] if len(r["connector_l2s"]) > 0 else 0
        conn_wrist = r["connector_l2s"][1] if len(r["connector_l2s"]) > 1 else 0
        amp_top = conn_top / vis_top if vis_top > 0 else 0
        amp_wrist = conn_wrist / vis_wrist if vis_wrist > 0 else 0
        
        print(f"{r['frame_idx']:5d} | {vis_top:10.4f} | {vis_wrist:10.4f} | {conn_top:10.4f} | {conn_wrist:10.4f} | {amp_top:8.4f}x | {amp_wrist:10.4f}x | {r['state_l2']:10.6f}")
    
    # 统计分析
    connector_l2s = [r["connector_l2"] for r in all_results]
    connector_cos_sims = [r["connector_cos_sim"] for r in all_results]
    connector_maes = [r["connector_mae"] for r in all_results]
    state_l2s = [r["state_l2"] for r in all_results]
    
    # 提取vision和connector的逐相机数据
    vision_top_l2s = [r["vision_l2s"][0] for r in all_results if len(r["vision_l2s"]) > 0]
    vision_wrist_l2s = [r["vision_l2s"][1] for r in all_results if len(r["vision_l2s"]) > 1]
    conn_top_l2s = [r["connector_l2s"][0] for r in all_results if len(r["connector_l2s"]) > 0]
    conn_wrist_l2s = [r["connector_l2s"][1] for r in all_results if len(r["connector_l2s"]) > 1]
    
    print(f"统计分析")
    
    print(f"\nVision Encoder L2距离:")
    print(f"  Top相机:  最小={min(vision_top_l2s):.4f}, 最大={max(vision_top_l2s):.4f}, 平均={np.mean(vision_top_l2s):.4f}")
    print(f"  Wrist相机: 最小={min(vision_wrist_l2s):.4f}, 最大={max(vision_wrist_l2s):.4f}, 平均={np.mean(vision_wrist_l2s):.4f}")
    
    print(f"\nConnector L2距离:")
    print(f"  Top相机:  最小={min(conn_top_l2s):.4f}, 最大={max(conn_top_l2s):.4f}, 平均={np.mean(conn_top_l2s):.4f}")
    print(f"  Wrist相机: 最小={min(conn_wrist_l2s):.4f}, 最大={max(conn_wrist_l2s):.4f}, 平均={np.mean(conn_wrist_l2s):.4f}")
    
    # 计算放大率
    amp_top = [c/v if v > 0 else 0 for c, v in zip(conn_top_l2s, vision_top_l2s)]
    amp_wrist = [c/v if v > 0 else 0 for c, v in zip(conn_wrist_l2s, vision_wrist_l2s)]
    
    print(f"\n放大率 (Connector/Vision):")
    print(f"  Top相机:  最小={min(amp_top):.4f}x, 最大={max(amp_top):.4f}x, 平均={np.mean(amp_top):.4f}x")
    print(f"  Wrist相机: 最小={min(amp_wrist):.4f}x, 最大={max(amp_wrist):.4f}x, 平均={np.mean(amp_wrist):.4f}x")
    
    print(f"\nConnector L2距离 (整体):")
    print(f"  最小值: {min(connector_l2s):.4f} (帧 {connector_l2s.index(min(connector_l2s))})")
    print(f"  最大值: {max(connector_l2s):.4f} (帧 {connector_l2s.index(max(connector_l2s))})")
    print(f"  平均值: {np.mean(connector_l2s):.4f}")
    print(f"  标准差: {np.std(connector_l2s):.4f}")
    
    print(f"\nConnector Cosine相似度:")
    print(f"  最小值: {min(connector_cos_sims):.6f} (帧 {connector_cos_sims.index(min(connector_cos_sims))})")
    print(f"  最大值: {max(connector_cos_sims):.6f} (帧 {connector_cos_sims.index(max(connector_cos_sims))})")
    print(f"  平均值: {np.mean(connector_cos_sims):.6f}")
    print(f"  标准差: {np.std(connector_cos_sims):.6f}")
    
    print(f"\nConnector MAE:")
    print(f"  最小值: {min(connector_maes):.6f}")
    print(f"  最大值: {max(connector_maes):.6f}")
    print(f"  平均值: {np.mean(connector_maes):.6f}")
    print(f"  标准差: {np.std(connector_maes):.6f}")
    
    print(f"\nState L2距离:")
    print(f"  最小值: {min(state_l2s):.6f}")
    print(f"  最大值: {max(state_l2s):.6f}")
    print(f"  平均值: {np.mean(state_l2s):.6f}")
    print(f"  标准差: {np.std(state_l2s):.6f}")
    
    # 趋势分析
    print(f"趋势分析")
    
    print(f"\nVision Encoder L2趋势 (Wrist):")
    first_10_vis = np.mean(vision_wrist_l2s[:10])
    middle_10_vis = np.mean(vision_wrist_l2s[num_compare_frames//2-5:num_compare_frames//2+5])
    last_10_vis = np.mean(vision_wrist_l2s[-10:])
    print(f"  前10帧平均: {first_10_vis:.4f}")
    print(f"  中间10帧平均: {middle_10_vis:.4f}")
    print(f"  最后10帧平均: {last_10_vis:.4f}")
    if last_10_vis > first_10_vis * 1.5:
        print(f"  -> Vision差异随时间增大")
    elif last_10_vis < first_10_vis * 0.5:
        print(f"  -> Vision差异随时间减小")
    else:
        print(f"  -> Vision差异保持相对稳定")
    
    print(f"\nConnector L2趋势 (Wrist):")
    first_10_conn = np.mean(conn_wrist_l2s[:10])
    middle_10_conn = np.mean(conn_wrist_l2s[num_compare_frames//2-5:num_compare_frames//2+5])
    last_10_conn = np.mean(conn_wrist_l2s[-10:])
    print(f"  前10帧平均: {first_10_conn:.4f}")
    print(f"  中间10帧平均: {middle_10_conn:.4f}")
    print(f"  最后10帧平均: {last_10_conn:.4f}")
    if last_10_conn > first_10_conn * 1.5:
        print(f"  -> Connector差异随时间增大")
    elif last_10_conn < first_10_conn * 0.5:
        print(f"  -> Connector差异随时间减小")
    else:
        print(f"  -> Connector差异保持相对稳定")
    
    print(f"\n放大率趋势 (Wrist):")
    first_10_amp = np.mean(amp_wrist[:10])
    middle_10_amp = np.mean(amp_wrist[num_compare_frames//2-5:num_compare_frames//2+5])
    last_10_amp = np.mean(amp_wrist[-10:])
    print(f"  前10帧平均: {first_10_amp:.4f}x")
    print(f"  中间10帧平均: {middle_10_amp:.4f}x")
    print(f"  最后10帧平均: {last_10_amp:.4f}x")
    if last_10_amp > first_10_amp * 1.1:
        print(f"  -> 放大率随时间增大（Connector对差异的放大效应增强）")
    elif last_10_amp < first_10_amp * 0.9:
        print(f"  -> 放大率随时间减小（Connector对差异的放大效应减弱）")
    else:
        print(f"  -> 放大率保持相对稳定")
    
    print(f"分析完成!")


if __name__ == "__main__":
    main()