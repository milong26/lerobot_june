"""
功能：代码中定义模型路径列表比如personal/work2/duibi/random_42/random_112_seed42/checkpoints/000200/pretrained_model
根据模型路径，采用lerobot的形式加载模型
利用metaworld生成pick-place-v3随机种子（定义10042）的场景的第一张top和wrist图以及state和task作为输入
绘制不同模型的注意力图。
要求标注英文
图片保存在personal/work2/attention_fig/result中，以模型文件名random_112_seed42命名
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 设置环境变量 - 必须在导入mujoco/gymnasium之前设置
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

# 添加lerobot到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.envs.metaworld import MetaworldEnv
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

# ============================================================
# 配置
# ============================================================

# 模型路径列表 - 在此添加你的模型路径
MODEL_PATHS = [
    "personal/work2/duibi/random_42/random_112_seed42/checkpoints/000200/pretrained_model",
    # 可以添加更多模型路径
    "personal/work2/duibi/uniform_42/uniform_112_seed42/checkpoints/000200/pretrained_model",
]

# MetaWorld 配置
TASK = "pick-place-v3"
SEED = 10042
CAMERA_NAMES = "corner2,gripperPOV"  # top相机和wrist相机

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "personal/work2/attention_fig/result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 注意力可视化配置
N_VIZ_LAYERS = 4  # 可视化前4层
N_VIZ_HEADS = 4   # 每层可视化前4个头


# ============================================================
# 注意力提取 Hook
# ============================================================

class AttentionCaptureHook:
    """Hook to capture attention weights from SmolVLA's eager_attention_forward."""
    
    def __init__(self, vlm_with_expert):
        self.vlm_with_expert = vlm_with_expert
        self.attentions = []
        self.hooks = []
        self.layer_counter = 0
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on the self_attn modules to capture Q, K, V before attention computation."""
        vlm_model = self.vlm_with_expert.get_vlm_model()
        text_layers = vlm_model.text_model.layers
        
        # Hook on each VLM layer's self_attn to capture inputs
        for layer_idx, layer in enumerate(text_layers):
            # Hook on the entire self_attn module's forward
            hook = layer.self_attn.register_forward_hook(
                self._make_vlm_hook(layer_idx)
            )
            self.hooks.append(hook)
        
        # Hook on expert layers
        for layer_idx, layer in enumerate(self.vlm_with_expert.lm_expert.layers):
            if layer is not None:
                hook = layer.self_attn.register_forward_hook(
                    self._make_expert_hook(layer_idx)
                )
                self.hooks.append(hook)
    
    def _make_vlm_hook(self, layer_idx):
        """Create hook for VLM self-attention layer."""
        def hook_fn(module, args, kwargs, output):
            # args contains: (hidden_states, ...), but we need to capture from the forward call
            # The self_attn forward receives hidden_states and optional args
            pass
        return hook_fn
    
    def _make_expert_hook(self, layer_idx):
        """Create hook for expert self-attention layer."""
        def hook_fn(module, args, kwargs, output):
            pass
        return hook_fn
    
    def clear(self):
        self.attentions = []
        self.layer_counter = 0
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class AttentionExtractor:
    """Extract attention weights by wrapping the eager_attention_forward method."""
    
    def __init__(self, vlm_with_expert):
        self.vlm_with_expert = vlm_with_expert
        self.original_method = vlm_with_expert.eager_attention_forward
        self.captured_attentions = []
        self.layer_idx = 0
        self._wrap_method()
    
    def _wrap_method(self):
        """Wrap the eager_attention_forward to capture attention weights."""
        extractor = self
        
        def wrapped_attention_forward(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            # Convert attention_mask to boolean BEFORE calling original method
            if attention_mask.dtype != torch.bool:
                attention_mask_bool = attention_mask.bool()
            else:
                attention_mask_bool = attention_mask
            
            # Call original method with converted mask
            att_output = extractor.original_method(
                attention_mask_bool, batch_size, head_dim, query_states, key_states, value_states
            )
            
            # Capture attention weights
            num_att_heads = extractor.vlm_with_expert.num_attention_heads
            num_key_value_heads = extractor.vlm_with_expert.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads
            sequence_length = key_states.shape[1]
            
            # Expand key_states to match query shape
            key_states_exp = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            key_states_exp = key_states_exp.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )
            
            # Compute attention weights
            query_states_fp32 = query_states.to(dtype=torch.float32).transpose(1, 2)
            key_states_fp32 = key_states_exp.to(dtype=torch.float32).transpose(1, 2)
            
            att_weights = torch.matmul(query_states_fp32, key_states_fp32.transpose(2, 3))
            att_weights *= head_dim ** -0.5
            
            # Apply mask
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask_bool[:, None, :, :], att_weights, big_neg)
            probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)
            
            # Store attention weights
            extractor.captured_attentions.append((extractor.layer_idx, probs.detach().cpu()))
            extractor.layer_idx += 1
            
            return att_output
        
        # Replace the method
        import types
        extractor.vlm_with_expert.eager_attention_forward = types.MethodType(
            lambda self, *args, **kwargs: wrapped_attention_forward(*args, **kwargs),
            extractor.vlm_with_expert
        )
    
    def reset(self):
        """Reset captured attentions and layer counter."""
        self.captured_attentions = []
        self.layer_idx = 0
    
    def get_attentions(self):
        """Get captured attentions organized by layer."""
        # Group by layer index
        layer_attentions = {}
        for layer_idx, attn_weights in self.captured_attentions:
            if layer_idx not in layer_attentions:
                layer_attentions[layer_idx] = []
            layer_attentions[layer_idx].append(attn_weights)
        
        # Return list of attention tensors per layer
        result = []
        for layer_idx in sorted(layer_attentions.keys()):
            # Average attention across calls in the same layer (if multiple)
            avg_attn = torch.stack(layer_attentions[layer_idx]).mean(dim=0)
            result.append(avg_attn)
        
        return result
    
    def restore(self):
        """Restore original method."""
        self.vlm_with_expert.eager_attention_forward = self.original_method


# ============================================================
# 注意力提取
# ============================================================

def extract_attention_from_model(policy, batch, device):
    """
    Extract attention weights from SmolVLA model using wrapped attention method.
    Runs a forward pass through the model to capture attention weights.
    """
    vlm_with_expert = policy.model.vlm_with_expert
    
    # Create attention extractor
    extractor = AttentionExtractor(vlm_with_expert)
    
    try:
        # Prepare inputs using policy methods (not policy.model)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        
        # Run a forward pass (training mode to get attention)
        # We use the model's forward method which calls vlm_with_expert.forward
        # This will capture attention weights through our wrapped method
        extractor.reset()
        
        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        
        # Create dummy actions and timestep for suffix
        bsize = state.shape[0]
        chunk_size = policy.config.chunk_size
        max_action_dim = policy.config.max_action_dim
        dummy_actions = torch.zeros(bsize, chunk_size, max_action_dim, device=device)
        dummy_timestep = torch.zeros(bsize, device=device)
        
        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks = policy.model.embed_suffix(
            dummy_actions, dummy_timestep
        )
        
        # Combine prefix and suffix
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        # Run forward through vlm_with_expert to capture attention
        # This simulates what happens in the model's forward pass
        with torch.no_grad():
            outputs, _ = vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
            )
        
        # Get captured attentions
        attentions = extractor.get_attentions()
        
        print(f"Captured {len(attentions)} layers of attention weights")
        if len(attentions) > 0:
            print(f"Attention[0] shape: {attentions[0].shape}")
        
        return attentions
        
    except Exception as e:
        print(f"Error during attention extraction: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        # Restore original method
        extractor.restore()


# ============================================================
# 环境设置
# ============================================================

def create_metaworld_input(task, seed, device):
    """Create metaworld environment and get first frame input."""
    env = MetaworldEnv(
        task=task,
        camera_name="corner2,gripperPOV",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
    )
    
    obs, info = env.reset(seed=seed)
    
    # 获取top和wrist图像 - 根据metaworld.py的_format_raw_obs返回格式
    top_image = obs["pixels/top"]  # HWC format, uint8
    wrist_image = obs["pixels/wrist"]  # HWC format, uint8
    state = obs["agent_pos"]  # 4-dim state
    
    # 保存原始图像用于可视化（保持HWC格式）
    top_image_np = top_image.copy()
    wrist_image_np = wrist_image.copy()
    
    # 转换为模型需要的格式 - 添加batch和time维度
    # 模型期望: (batch, time, channels, height, width) 或 (batch, channels, height, width)
    # 先转换为CHW格式
    top_img_chw = np.transpose(top_image.copy(), (2, 0, 1))  # HWC -> CHW
    wrist_img_chw = np.transpose(wrist_image.copy(), (2, 0, 1))
    
    # 转换为tensor并添加维度
    top_img_tensor = torch.from_numpy(top_img_chw).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 3, H, W)
    wrist_img_tensor = torch.from_numpy(wrist_img_chw).unsqueeze(0).unsqueeze(0).float().to(device)
    
    state_tensor = torch.from_numpy(state.copy()).unsqueeze(0).float().to(device)
    
    task_description = env.task_description
    
    return {
        "observation.images.camera1": top_img_tensor,
        "observation.images.camera2": wrist_img_tensor,
        "observation.state": state_tensor,
        "task": task_description,
        "top_image_np": top_image_np,
        "wrist_image_np": wrist_image_np,
        "state": state,
    }


def prepare_batch_for_model(model_data, policy, device):
    """Prepare batch in the format expected by the policy."""
    # 根据模型的image_features配置构建batch
    batch = {}
    
    # 处理图像 - 使用模型配置中的image_features
    for img_key in policy.config.image_features:
        if img_key in model_data:
            batch[img_key] = model_data[img_key]
    
    # 处理state
    batch["observation.state"] = model_data["observation.state"]
    
    # 处理language/task
    processor = policy.model.vlm_with_expert.processor
    task = model_data["task"]
    
    text_inputs = processor.tokenizer(
        task,
        return_tensors="pt",
        padding="max_length",
        max_length=processor.tokenizer.model_max_length if hasattr(processor.tokenizer, 'model_max_length') else 512,
        truncation=True,
    )
    
    batch["observation.language.tokens"] = text_inputs["input_ids"].to(device)
    batch["observation.language.attention_mask"] = text_inputs["attention_mask"].to(device)
    
    return batch


# ============================================================
# 注意力可视化
# ============================================================

def visualize_attention(attentions, top_image, wrist_image, model_name, output_path):
    """Visualize attention maps overlaid on images."""
    if not attentions:
        print(f"No attention maps to visualize for {model_name}")
        return
    
    n_layers = len(attentions)
    n_layers_viz = min(N_VIZ_LAYERS, n_layers)
    
    if n_layers_viz == 0:
        print(f"No valid attention layers for {model_name}")
        return
    
    # 获取第一个attention的shape来确定head数量
    first_attn = attentions[0]
    n_heads = first_attn.shape[1] if first_attn.ndim >= 2 else 1
    n_heads_viz = min(N_VIZ_HEADS, n_heads)
    
    # 创建图形
    fig_cols = n_heads_viz
    fig_rows = n_layers_viz * 2 + 1  # +1 for original images
    
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 4, fig_rows * 3.5))
    
    if fig_rows == 1 and fig_cols == 1:
        axes = np.array([[axes]])
    elif fig_rows == 1:
        axes = axes[np.newaxis, :]
    elif fig_cols == 1:
        axes = axes[:, np.newaxis]
    
    # 第一行显示原始图像
    ax_wrist = axes[0, 0]
    ax_wrist.imshow(wrist_image)
    ax_wrist.set_title("Wrist Camera View", fontsize=12, fontweight='bold')
    ax_wrist.axis("off")
    
    if fig_cols > 1:
        ax_top = axes[0, 1]
        ax_top.imshow(top_image)
        ax_top.set_title("Top Camera View", fontsize=12, fontweight='bold')
        ax_top.axis("off")
    
    # 隐藏第一行其他子图
    for c in range(2, fig_cols):
        axes[0, c].axis("off")
    
    # 获取图像尺寸
    h_wrist, w_wrist = wrist_image.shape[:2]
    h_top, w_top = top_image.shape[:2]
    
    # 可视化注意力
    for layer_idx in range(n_layers_viz):
        attn = attentions[layer_idx]
        
        for head_idx in range(n_heads_viz):
            # 获取注意力权重 (batch, heads, seq_q, seq_k)
            if attn.ndim == 4:
                attn_map = attn[0, head_idx].cpu().float().numpy()
            else:
                continue
            
            # 取最后一个query token的注意力分布（通常是最有信息的）
            if attn_map.ndim == 2:
                # attn_map shape: (seq_q, seq_k)
                # Take attention from the last query token
                attn_1d = attn_map[-1]  # Last row
            else:
                attn_1d = attn_map.flatten()
            
            # 将1D注意力重塑为2D网格（近似图像空间）
            n_tokens = len(attn_1d)
            grid_size = int(np.ceil(np.sqrt(n_tokens)))
            attn_2d = np.zeros((grid_size, grid_size))
            attn_2d.flat[:n_tokens] = attn_1d
            
            # 归一化
            attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
            
            # 调整到图像尺寸 - Wrist
            attn_wrist = cv2.resize(attn_2d, (w_wrist, h_wrist), interpolation=cv2.INTER_CUBIC)
            attn_wrist_norm = (attn_wrist - attn_wrist.min()) / (attn_wrist.max() - attn_wrist.min() + 1e-8)
            heatmap_wrist = cv2.applyColorMap((attn_wrist_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_wrist = cv2.cvtColor(heatmap_wrist, cv2.COLOR_BGR2RGB)
            overlay_wrist = cv2.addWeighted(wrist_image, 0.5, heatmap_wrist, 0.5, 0)
            
            # 调整到图像尺寸 - Top
            attn_top = cv2.resize(attn_2d, (w_top, h_top), interpolation=cv2.INTER_CUBIC)
            attn_top_norm = (attn_top - attn_top.min()) / (attn_top.max() - attn_top.min() + 1e-8)
            heatmap_top = cv2.applyColorMap((attn_top_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_top = cv2.cvtColor(heatmap_top, cv2.COLOR_BGR2RGB)
            overlay_top = cv2.addWeighted(top_image, 0.5, heatmap_top, 0.5, 0)
            
            # 绘制Wrist注意力
            row_wrist = layer_idx * 2 + 1
            col = head_idx
            ax = axes[row_wrist, col]
            ax.imshow(overlay_wrist)
            ax.set_title(f"Layer {layer_idx} / Head {head_idx}\n(Wrist Attention)", fontsize=10)
            ax.axis("off")
            
            # 绘制Top注意力
            row_top = row_wrist + 1
            ax2 = axes[row_top, col]
            ax2.imshow(overlay_top)
            ax2.set_title(f"Layer {layer_idx} / Head {head_idx}\n(Top Attention)", fontsize=10)
            ax2.axis("off")
    
    plt.suptitle(f"Attention Visualization - {model_name}", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved attention visualization to {output_path}")


# ============================================================
# 主函数
# ============================================================

def process_model(model_path, device):
    """Process a single model: load, extract attention, visualize."""
    print(f"\n{'='*80}")
    print(f"Processing model: {model_path}")
    print(f"{'='*80}")
    
    # 从路径提取模型名称
    model_name = Path(model_path).parent.parent.name  # e.g., "random_112_seed42"
    output_path = OUTPUT_DIR / f"{model_name}_attention.png"
    
    # 加载模型
    print(f"Loading model from {model_path}...")
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy = policy.to(device)
    policy.eval()
    print(f"Model loaded successfully. Config type: {policy.config.type}")
    print(f"Image features: {policy.config.image_features}")
    
    # 创建MetaWorld输入
    print(f"Creating MetaWorld environment with seed {SEED}...")
    model_data = create_metaworld_input(TASK, SEED, device)
    print(f"Task: {model_data['task']}")
    print(f"State shape: {model_data['state'].shape}")
    print(f"Top image shape: {model_data['top_image_np'].shape}")
    print(f"Wrist image shape: {model_data['wrist_image_np'].shape}")
    
    # 准备batch
    print("Preparing batch for model...")
    batch = prepare_batch_for_model(model_data, policy, device)
    print(f"Batch keys: {batch.keys()}")
    
    # 提取注意力
    print("Extracting attention maps...")
    try:
        attentions = extract_attention_from_model(policy, batch, device)
        print(f"Extracted {len(attentions)} attention layers")
        
        if len(attentions) > 0:
            print(f"Attention[0] shape: {attentions[0].shape}")
    except Exception as e:
        print(f"Error extracting attention: {e}")
        import traceback
        traceback.print_exc()
        attentions = []
    
    # 可视化
    if attentions:
        print("Visualizing attention maps...")
        visualize_attention(
            attentions,
            model_data["top_image_np"],
            model_data["wrist_image_np"],
            model_name,
            output_path
        )
    else:
        print(f"WARNING: No attention maps extracted for {model_name}")
    
    # 清理
    del policy
    torch.cuda.empty_cache()
    
    return output_path


def main():
    """Main function to process all models."""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Extract and visualize attention maps from SmolVLA models")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for inference (default: cuda)")
    args = parser.parse_args()
    
    # 设置设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        device = "cpu"
    print(f"Using device: {device}")
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 转换路径为绝对路径
    abs_model_paths = []
    for path in MODEL_PATHS:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            abs_model_paths.append(str(p))
        else:
            print(f"WARNING: Model path does not exist: {p}")
    
    if not abs_model_paths:
        print("ERROR: No valid model paths found. Please update MODEL_PATHS in the script.")
        sys.exit(1)
    
    print(f"Found {len(abs_model_paths)} model(s) to process")
    
    # 处理每个模型
    results = []
    for model_path in abs_model_paths:
        output_path = process_model(model_path, device)
        results.append((model_path, output_path))
    
    # 打印总结
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for model_path, output_path in results:
        status = "SUCCESS" if output_path and output_path.exists() else "FAILED"
        print(f"[{status}] {model_path}")
        if output_path:
            print(f"  -> {output_path}")


if __name__ == "__main__":
    main()