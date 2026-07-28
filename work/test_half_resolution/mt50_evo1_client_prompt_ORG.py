# mt50_evo1_client.py
import os
import websockets
import random,sys
import numpy as np
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
import asyncio
import json
import os,torch
from typing import List, Optional, Dict, Set
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from PIL import Image
import numpy as np

#server_url = "ws://0.0.0.0:9000"
server_url = "ws://10.10.16.19:9000"

class Normalizer:
    def __init__(self, stats_or_path):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path
        
        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < 24:
                pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > 24:
                raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
            return x
        
        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")
        
        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]
        
        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.force_min = pad_to_24(robot_stats["observation.force"]["min"])
        self.force_max = pad_to_24(robot_stats["observation.force"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])
        
        # import pdb;pdb.set_trace()
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_min = self.state_min.to(state.device, dtype=state.dtype)
        state_max = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)
    
    def normalize_force(self, state: torch.Tensor) -> torch.Tensor:
        state_min = self.force_min.to(state.device, dtype=state.dtype)
        state_max = self.force_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)
    
    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min
    
    def denormalize_state(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.state_min.to(action.device, dtype=action.dtype)
        action_max = self.state_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min
    
    def denormalize_force(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.force_min.to(action.device, dtype=action.dtype)
        action_max = self.force_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min

normalizer = Normalizer('./norm_stats.json')

def _load_pkl( pkl_path):
    """
    兼容不同 PyTorch 版本。
    如果你的 pkl 都是自己生成的，weights_only=False 是可以的。
    """
    try:
        return torch.load(pkl_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(pkl_path, map_location="cpu")


def encode_image_uint8_list(img_bgr: np.ndarray):
    return img_bgr.astype(np.uint8).tolist()


async def _amain():
    ds_files = os.listdir('./train_embeddings_cache')
    
   
    for pkl_path in ds_files:
        pkl_path = './train_embeddings_cache/'+pkl_path
        if not pkl_path.endswith('.pkl'):
            continue
        
        jpg_path = pkl_path.replace('.pkl',".jpg")
        
        arr = _load_pkl(pkl_path)
        
        fused_tokens = arr[0]
        state = arr[1]
        actions_gt = arr[2]
        action_mask = arr[3]
        embodiment_ids = arr[4]
        timestep = arr[5]
        timestamp = arr[6]
        cache_filepath = arr[7]
        prompt = arr[8]
        y_2 = arr[9]
        input_latents = arr[10]
        state_mask = arr[11]
        pkl_path = str(pkl_path)
        
        #import pdb;pdb.set_trace()
        state1 = torch.cat([state[:, :6], torch.zeros((50, 24 - 6), device=y_2.device)], dim=1)
        force1 = torch.cat([state[:, 6:21], torch.zeros((50, 24 - 15), device=y_2.device)], dim=1)
        state = normalizer.denormalize_state(state1)
        force = normalizer.denormalize_force(force1)
        state = torch.cat([state[:, :6], force[:, :15]], dim=1)
        state = torch.cat([state, torch.zeros((50, 24 - state.shape[1]), device=y_2.device)], dim=1)
        
        async with websockets.connect(server_url, max_size=100_000_000, ping_interval=30,ping_timeout=120 ) as ws:
            assert prompt is not None and len(prompt) > 0, "prompt should be non-empty"
            dummy_img = np.ones((640, 240, 3), dtype=np.uint8)*255
            
            payload = {
                "image": [dummy_img.astype(np.uint8).tolist()] *2, # -4 step，0 step # 当前的和前4帧的图片
                "state": state[-2:,:21].tolist(), #原始值 # -4 step，0 step
                "action": actions_gt[-2:].tolist(),
                "prompt": prompt,
                "steps": 11,
                "seed": 22,
                "g_scale": 1,
                "video_name": '',
                "image_mask": [1, 1, 0],
                "action_mask": [1, 1, 1, 1,1,1] + [0] * 18,
                "num_loop": 1, # 1或者2
            }
            await ws.send(json.dumps(payload))
            data = json.loads(await ws.recv())
            action = np.asarray(data['act'], dtype=np.float32) # action直接用
            state1 = np.asarray(data['sta'], dtype=np.float32)
            state = state1[:,:6]
            force = state1[:,6:21]
            import pdb;pdb.set_trace()
            
           












if __name__ == "__main__":
    seed_everything(11)
    asyncio.run(_amain())



# if __name__ == "__main__":
#     N_REPEAT = 1
#     for run_id in range(N_REPEAT):
#         print(f"\n\n===== 🌟 Run {run_id + 1}/{N_REPEAT} =====")
#         asyncio.run(_amain())
