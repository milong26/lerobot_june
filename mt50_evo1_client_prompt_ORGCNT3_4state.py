# mt50_evo1_client.py
import os
# 强制Mujoco使用EGL离屏渲染
os.environ["MUJOCO_GL"] = "osmesa"
# OpenGL绑定EGL平台
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import random,sys
import numpy as np
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
import asyncio
import json
import os
from typing import List, Optional, Dict, Set

import cv2
import gymnasium as gym
import metaworld  # noqa: F401

import websockets



import datetime

import hashlib

import hashlib

def flatten_obs_for_hash(obs) -> np.ndarray:
    if isinstance(obs, dict):
        if "observation" in obs:
            arr = np.asarray(obs["observation"], dtype=np.float32).ravel()
        else:
            arr = np.concatenate([np.asarray(v).ravel() for v in obs.values()]).astype(np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32).ravel()
    return arr


def make_scene_signature(env, scene_seed: int):
    obs, _ = reset_metaworld_scene(env, scene_seed)

    obs_arr = np.round(flatten_obs_for_hash(obs), 6)
    obs_md5 = hashlib.md5(obs_arr.tobytes()).hexdigest()

    img_bgr = render_single_bgr(env)
    img_md5 = hashlib.md5(img_bgr.tobytes()).hexdigest()

    return obs_md5, img_md5


def verify_same_scene(env, base_seed: int, task_idx: int, ep: int):
    scene_seed = make_scene_seed(base_seed, task_idx, ep)

    sig1 = make_scene_signature(env, scene_seed)
    sig2 = make_scene_signature(env, scene_seed)

    print(f"[verify] task={task_idx} ep={ep} scene_seed={scene_seed}")
    print(f"[verify] obs same:   {sig1[0] == sig2[0]}  {sig1[0]}  {sig2[0]}")
    print(f"[verify] image same: {sig1[1] == sig2[1]}  {sig1[1]}  {sig2[1]}")
    #assert sig1[0] == sig2[0] and  sig1[1] == sig2[1]
    if not sig1[0] == sig2[0] and  sig1[1] == sig2[1]:
        os.mkdir('./log_ERROR')
        open(f'./log_ERROR/log_{base_seed}_{task_idx}_{ep}_obs{sig1[0] == sig2[0]}_image{sig1[1] == sig2[1]}','wb').clos()
    
# ===================== Logging =====================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def make_log_path(prefix="eval"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"{prefix}_{ts}.txt")

LOG_PATH = make_log_path("mt50")
# ====================================================

SHOW_WINDOW = True
SAVE_IMAGE = False
SAVE_VIDEO = True  # save the video of each episode to disk

# ===================== Debug image saving =====================
INSPECT_SAMPLE_PER_EPISODE = True        
INSPECT_DIR = "inspect_frames"           
APPLY_ROT_180 = True                     
APPLY_CENTER_CROP = True
CROP_KEEP_RATIO = 2/3                    
INSPECT_SAVE_STEP_TAG = True             
# =============================================================
if len(sys.argv)<2:
    HORIZON = 5
    g_scale = 1.0
else:
    HORIZON = int(sys.argv[1])
    g_scale = float(sys.argv[2])
        

TARGET_LEVEL = "all"   # one of "all", "easy", "medium", "hard", "very_hard"
TARGET_LEVEL = "all"#"medium"
TARGET_LEVEL = [ "easy","medium", "hard", "very_hard"][int(sys.argv[3])]
num_loop = int(sys.argv[4])
model_name = sys.argv[5]




# ===================== Debug video saving ====================
VIDEO_SAVE_DIR = f"episode_videos_HORIZON{HORIZON}_GS{g_scale}_{TARGET_LEVEL}_DF{num_loop}_[{model_name.split('/')[-1]}]"
VIDEO_FPS = 30  # Original writing frame rate (used to control playback speed; the smaller the value, the slower the playback).
VIDEO_DUP_FRAMES = 1  # Number of times to duplicate each frame when writing video (used to control playback speed; the larger the value, the slower the playback).
# =============================================================

try:
    files = os.listdir(VIDEO_SAVE_DIR)
    ok_tsks = [tsk.split('-v3_')[0]+'-v3' for tsk in files if tsk.endswith(".mp4")]
except:
    ok_tsks = []


# ===================== User Config (edit here) =====================
#SERVER_URL = "ws://172.30.48.1:9000"
#SERVER_URL = "ws://0.0.0.0:9000"
SERVER_URL = "ws://10.10.16.19:9902"
# Camera & image settings
CAMERA_NAME = "corner2"        
IMG_SIZE = (448, 448)          

# Evo1 & rollout settings
STATE_TAKE = 8

EPISODES = 100
EPISODE_HORIZON = 800
SEED = 4042


# Order source
ORDER_JSON_PATH = "mt50_order.json"      

FALLBACK_USE_FIRST_N: Optional[int] = 5
FALLBACK_IDX_LIST: Optional[List[int]] = None

# Prompt source
TASKS_JSONL_PATH = "tasks.jsonl"         
# ==================================================================

# Headless GL by default; switch to 'glfw' on a desktop if you want
#os.environ.setdefault("MUJOCO_GL", "egl")
gym.logger.min_level = gym.logger.ERROR


# ---------------- Utils ----------------
def encode_image_uint8_list(img_bgr: np.ndarray):
    return img_bgr.astype(np.uint8).tolist()

def obs_to_state(obs, take: int = STATE_TAKE) -> List[float]:
    if isinstance(obs, dict):
        if "observation" in obs:
            arr = np.asarray(obs["observation"], dtype=np.float32).ravel()
        else:
            parts = [np.asarray(v).ravel() for v in obs.values()]
            arr = np.concatenate(parts).astype(np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32).ravel()
    return arr[:min(take, arr.shape[0])].tolist()

def fix_camera_angle(rgb: np.ndarray) -> np.ndarray:
    
    return cv2.rotate(rgb, cv2.ROTATE_180)

def center_crop_keep_ratio(rgb: np.ndarray, keep_ratio: float) -> np.ndarray:
    
    h, w = rgb.shape[:2]
    keep_ratio = float(keep_ratio)
    keep_ratio = max(1e-6, min(1.0, keep_ratio))  
    new_h = max(1, int(round(h * keep_ratio)))
    new_w = max(1, int(round(w * keep_ratio)))
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return rgb[y0:y0 + new_h, x0:x0 + new_w, :]

def render_single_bgr(env) -> np.ndarray:
  
    rgb = env.render()                               
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)   

   
    if APPLY_ROT_180:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        rgb = np.ascontiguousarray(rgb)

    
    if APPLY_CENTER_CROP and (0.0 < CROP_KEEP_RATIO < 1.0):
        h, w = rgb.shape[:2]
        keep = float(CROP_KEEP_RATIO)
        new_h = max(1, int(round(h * keep)))
        new_w = max(1, int(round(w * keep)))
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        rgb = rgb[y0:y0 + new_h, x0:x0 + new_w, :].copy()
        rgb = np.ascontiguousarray(rgb)

   
    if IMG_SIZE is not None:
        rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(rgb)

    
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)

    
    if 'SHOW_WINDOW' in globals() and SHOW_WINDOW:
        try:
            cv2.imshow("MetaWorld", bgr)
            cv2.waitKey(1)   
        except Exception:
           
            pass

    return bgr

def create_video_writer(env, video_name: str):
    """
    create and return a cv2.VideoWriter object for saving episode videos.
    """
    os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
    probe_frame = render_single_bgr(env)  # Render one frame first to get the dimensions.
    h0, w0 = probe_frame.shape[:2]
    frame_size = (w0, h0)
    video_path = os.path.join(VIDEO_SAVE_DIR, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, frame_size)
    # Write the detection frame as the first frame.
    for _ in range(VIDEO_DUP_FRAMES):
        video_writer.write(probe_frame)
    return video_writer

def write_video(video_writer, img_bgr: np.ndarray):
    """
    write a frame to the given cv2.VideoWriter object.
    """
    try:
        if video_writer is not None:
            for _ in range(VIDEO_DUP_FRAMES):
                video_writer.write(img_bgr)
    except Exception as e:
        log_write(f"[video][ERROR] writer.write failed: {e}")

def save_episode_video(writer, video_name: str, task_idx: int, slug: str, ep_num: int):
    """save the video to disk and close video writer."""
    if writer is None:
        return
    try:
        video_path = os.path.join(VIDEO_SAVE_DIR, video_name)
        writer.release()
        log_write(f"[video] task={task_idx} slug={slug} ep={ep_num} saved video frames {video_path}")
    except Exception as e:
        log_write(f"[video][ERROR] closing writer failed: {e}")


async def evo1_infer(ws, Image_HIST, STATE_HIST,ACTION_HIST, prompt: Optional[str] = None,video_name: Optional[str] = None,steps=-1,seed=0) -> np.ndarray:
    assert prompt is not None and len(prompt) > 0, "prompt should be non-empty"
    #dummy_img = np.zeros((480, 480, 3), dtype=np.uint8)
    #Image_HIST = [Image_HIST[-1]]
    #STATE_HIST = [STATE_HIST[-1]]
    
    img = [encode_image_uint8_list(v) for v in Image_HIST]
    STATE_HIST2 = STATE_HIST#[STATE_HIST[0]]*(2-len(STATE_HIST))+STATE_HIST
    ACTION_HIST2 = [ACTION_HIST[0]] * (2 - len(ACTION_HIST)) + ACTION_HIST
    #import pdb; pdb.set_trace()
    payload = {
        "image": [img[0]]*(2-len(img))+img,
        "state": np.concatenate(STATE_HIST2, axis=-1).tolist(),
        "action": np.concatenate(ACTION_HIST2, axis=-1).tolist(),
        "prompt": prompt,
        "steps":steps,
        "seed": seed,
        "g_scale":g_scale,
        "video_name": video_name,
        "image_mask": [1, 0, 0],
        "action_mask": [1, 1, 1, 1] + [0]*20,
        "num_loop":num_loop,
        "model":model_name
    }
    await ws.send(json.dumps(payload))
    data = json.loads(await ws.recv())
    #import pdb;pdb.set_trace()
    return np.asarray(data['act'], dtype=np.float32),np.asarray(data['sta'], dtype=np.float32)


def save_sent_bgr_frame(img_bgr: np.ndarray, ep_num: int, idx: int, slug: str, step: Optional[int] = None):

    os.makedirs(INSPECT_DIR, exist_ok=True)
    tag = f"step{step:04d}" if (INSPECT_SAVE_STEP_TAG and step is not None) else "stepNA"
    out = os.path.join(INSPECT_DIR, f"ep{ep_num:03d}_idx{idx}_{slug}_{tag}.png")
    img_bgr_safe = np.ascontiguousarray(img_bgr)  
    cv2.imwrite(out, img_bgr_safe)
    h, w = img_bgr_safe.shape[:2]
    print(f"[inspect] saved {out}  size={w}x{h}  (identical to VLA input)")

def log_write(text: str):
    
    print(text)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ---------------- Prompt loader ----------------
class PromptBook:

    def __init__(self, jsonl_path: str):
        self.by_idx: Dict[int, str] = {}
        self.by_slug: Dict[str, str] = {}
        self.seq: List[str] = []

        if not os.path.exists(jsonl_path):
            print(f"[WARN] {jsonl_path} not found; prompts will be empty.")
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        for i, obj in enumerate(lines):
            task_txt = str(obj.get("task", "")).strip()
            if "idx" in obj:
                try:
                    self.by_idx[int(obj["idx"])] = task_txt
                except Exception:
                    pass
            if "slug" in obj:
                try:
                    self.by_slug[str(obj["slug"])] = task_txt
                except Exception:
                    pass
            self.seq.append(task_txt)

    def get(self, idx: int, slug: Optional[str] = None) -> str:
        if idx in self.by_idx:
            return self.by_idx[idx]
        if slug is not None and slug in self.by_slug:
            return self.by_slug[slug]
        if 0 <= idx < len(self.seq):
            return self.seq[idx]
        return ""


PROMPTS = PromptBook(TASKS_JSONL_PATH)

def make_scene_seed(base_seed: int, task_idx: int, ep: int) -> int:
    # 不要用 Python hash()，因为 hash() 可能受 PYTHONHASHSEED 影响
    return int((base_seed * 1_000_003 + task_idx * 10_007 + ep) % (2**31 - 1))

def reset_metaworld_scene(env, scene_seed: int):
    """
    MetaWorld SawyerXYZEnv 的 reset(seed=...) 可能忽略 seed。
    因此必须先 env.seed(scene_seed)，再 env.reset()。
    """

    # 1. MetaWorld 专用 seed
    if hasattr(env, "seed"):
        env.seed(scene_seed)

    # 2. 有些 wrapper 把真正 env 放在 unwrapped
    uenv = getattr(env, "unwrapped", env)
    if uenv is not env and hasattr(uenv, "seed"):
        uenv.seed(scene_seed)

    # 3. 让 _get_state_rand_vec 使用 env.np_random，而不是全局 np.random
    for obj in (env, uenv):
        if hasattr(obj, "seeded_rand_vec"):
            obj.seeded_rand_vec = True
        if hasattr(obj, "_freeze_rand_vec"):
            obj._freeze_rand_vec = False
        if hasattr(obj, "_last_rand_vec"):
            obj._last_rand_vec = None

    # 4. 注意这里不要传 seed
    obs, info = env.reset()
    return obs, info
# ---------------- Order & groups loader ----------------
def load_order_and_groups(total_envs: int):
   
    if os.path.exists(ORDER_JSON_PATH):
        with open(ORDER_JSON_PATH, "r") as f:
            data = json.load(f)
        ordered_indices = list(map(int, data["ordered_indices"]))
     
        groups = {k: set(v) for k, v in data["groups"].items()}
        idx_to_slug = {int(k): v for k, v in data["idx_to_slug"].items()}
        print(f"[INFO] Loaded order from {ORDER_JSON_PATH} (len={len(ordered_indices)})")
        log_write(f"[INFO] Metaworld Evaluation Begins ...")
        return ordered_indices, groups, idx_to_slug

  
    if FALLBACK_IDX_LIST:
        idx_list = [i for i in FALLBACK_IDX_LIST if 0 <= i < total_envs]
    elif FALLBACK_USE_FIRST_N:
        idx_list = list(range(min(FALLBACK_USE_FIRST_N, total_envs)))
    else:
        idx_list = list(range(total_envs))
    print("[WARN] mt50_order.json not found; falling back to:", idx_list)
    
    idx_to_slug = {i: f"task-{i}" for i in idx_list}
    groups = {"easy": set(), "medium": set(), "hard": set(), "very_hard": set()}
    return idx_list, groups, idx_to_slug

def _base_env(sub):
    return getattr(sub, "unwrapped", sub)

from Save2Load import *
# ---------------- Core eval (MT50 only, ordered by mt50_order.json) ----------------
async def eval_mt50_with_groups(server_url: str,
                                num_eval_episodes: int = EPISODES,
                                episode_horizon: int = EPISODE_HORIZON,
                                seed: int = SEED):
  
    # 1) Build MT50 with fixed camera
    envs = gym.make_vec(
        "Meta-World/MT50",
        vector_strategy="sync",
        seed=seed,
        render_mode="rgb_array",
        camera_name=CAMERA_NAME,
    )
    total_envs = len(envs.envs)

    # 2) Load ordered idx list & groups
    ordered_indices, groups, idx_to_slug = load_order_and_groups(total_envs)
    ordered_indices = [i for i in ordered_indices if 0 <= i < total_envs]

    
    if TARGET_LEVEL.lower() != "all":
        allowed_slugs = groups.get(TARGET_LEVEL.lower(), set())
        before = len(ordered_indices)
        ordered_indices = [i for i in ordered_indices if idx_to_slug.get(i, "") in allowed_slugs]
        print(f"[INFO] Filtered tasks: keep only {TARGET_LEVEL} ({len(ordered_indices)}/{before})")


    # 3) Accumulators
    success_counts: Dict[int, int] = {i: 0 for i in ordered_indices}
    trials_counts: Dict[int, int] = {i: 0 for i in ordered_indices}
    group_success = {k: 0 for k in ["easy", "medium", "hard", "very_hard"]}
    group_trials  = {k: 0 for k in ["easy", "medium", "hard", "very_hard"]}
    g_seed = 0

    # 4) Main loop
    async with websockets.connect(server_url, max_size=100_000_000, ping_interval=30,ping_timeout=120 ) as ws:
        empty_closed_width = None
        for idx in ordered_indices:
            sub = envs.envs[idx]
            
            if empty_closed_width is None:
                obs, _ = sub.reset(seed=seed)
                width_fn = lambda sub: get_gripper_width_by_sites(sub, left_site_name="rightEndEffector", right_site_name="leftEndEffector")
                
                close_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                empty_closed_width, obs = calibrate_empty_closed_width(
                    sub,
                    None,
                    close_action,
                    width_fn=width_fn,
                    n_steps=30,
                )
                print("empty_closed_width:", empty_closed_width)
            #import pdb;pdb.set_trace()
            
            
            slug = idx_to_slug.get(idx, f"task-{idx}")

            
            task_prompt = PROMPTS.get(idx, slug=slug)
            
            #if task_prompt not in ['Dunk the basketball into the basket', 'Grasp the puck from one bin and place it into another bin']:
            #    continue
                
            

            gname_for_task = None
            for gname in group_trials.keys():
                if slug in groups.get(gname, set()):
                    gname_for_task = gname
                    break

            for ep in range(num_eval_episodes):
                '''for obj in (sub, getattr(sub, "unwrapped", None)):
                    fn = getattr(obj, "iterate_goal_position", None)
                    if callable(fn):
                        try: fn()
                        except Exception: pass
                        break'''
                video_name = f"task{idx:02d}_{slug}_ep{ep + 1:03d}.mp4"
                # import pdb;pdb.set_trace()
                if f"task{idx:02d}_{slug}" in ok_tsks:
                    continue

                inspect_choice = INSPECT_SAMPLE_PER_EPISODE
                saved_this_episode = False
                
                scene_seed = make_scene_seed(seed, idx, ep)
                
                verify_same_scene(sub, seed, idx, ep)
                
                #obs, _ = sub.reset(seed=scene_seed)
                #scene_seed = make_scene_seed(seed, idx, ep)
                obs, _ = reset_metaworld_scene(sub, scene_seed)
                try:
                    sub.action_space.seed(scene_seed)
                except Exception:
                    pass
                
                
                #obs, _ = sub.reset(seed=seed + ep)
                trials_counts[idx] += 1
                if gname_for_task is not None:
                    group_trials[gname_for_task] += 1

                steps = 0
                done = False
                cur_success = False
                
                video_writer = None if not SAVE_VIDEO else create_video_writer(sub, video_name)

                try:
                    a0 = np.zeros(sub.action_space.shape, dtype=np.float32)
                    a0 = np.clip(a0, sub.action_space.low, sub.action_space.high)
                    obs, _, _, _, _ = sub.step(a0)
                except Exception:
                    pass
                
                Image_HIST = []
                STATE_HIST = []
                ACTION_HIST = []
                
                #reset_recovery = make_perfect_sub_checkpoint(sub, obs, steps)
                reset_recovery = make_robot_gripper_checkpoint(sub)
                
                rollback_happened = 0
                rollback_limited = 0

                while steps < episode_horizon and not done:
                    img_bgr = render_single_bgr(sub)
                    if len(Image_HIST)>2:
                        Image_HIST33 = [Image_HIST[0],img_bgr]
                    else:
                        Image_HIST33 = [img_bgr,img_bgr]
                    
                    if SAVE_VIDEO:
                        write_video(video_writer, img_bgr)

                    if SAVE_IMAGE and inspect_choice and (not saved_this_episode):
                        save_sent_bgr_frame(
                            img_bgr, ep_num=ep + 1, idx=idx, slug=slug,
                            step=steps if INSPECT_SAVE_STEP_TAG else None
                        )
                        saved_this_episode = True

                    state_vec = obs_to_state(obs)
                    #
                    if len(STATE_HIST) > 2:
                        #STATE_HIST33=STATE_HIST+[np.array(state_vec)]#
                        STATE_HIST33 = [STATE_HIST[0], np.array(state_vec)]  #
                        ACTION_HIST33 =[ ACTION_HIST[0], np.array([0, 0, 0, 0])]
                    else:
                        STATE_HIST33 = [np.array(state_vec),np.array(state_vec),np.array(state_vec),np.array(state_vec),np.array(state_vec)]
                        STATE_HIST33 = [np.array(state_vec), np.array(state_vec)]
                        ACTION_HIST33 =[np.array([0, 0, 0, 0]),np.array([0, 0, 0, 0])]
                    
                    #
                    

                 
                    actions,states = await evo1_infer(ws, Image_HIST33, STATE_HIST33,ACTION_HIST33,
                                                      prompt=task_prompt,video_name=video_name,steps=ep*1000+steps,
                                                      seed=(g_seed+rollback_happened) % 10)
                    
                    STA_DIFF_THR = np.array([0.05, 0.05, 0.05, 0.30], dtype=np.float32)
                    sta_diff_list = []
                    # 初始恢复点：当前 sub 状态
                   
                    
                    
                    hand_sig = states[:,3]
                    LOC_HORIZON = HORIZON
                    state_vec = obs_to_state(obs)

                    # 获得夹爪宽度
                    width = get_gripper_width_by_sites(sub, left_site_name="rightEndEffector", right_site_name="leftEndEffector")
                    close_gripper = not (width>empty_closed_width+0.005) #关上
                    #
                    print('grisp:' ,actions[:,3].mean().item(), ' std: ',actions[:,3].std().item())
                    if close_gripper and (width+0.002)*10<states[:,3].mean():
                        #import pdb;pdb.set_trace()
                        rollback_limited = rollback_limited + 1
                        print(
                            f"[ROLLBACK] idx={idx}, rollback_limited={rollback_limited}, close_gripper={close_gripper},{(width+0.002)*10} "
                            f"states[:,3].mean()={states[:,3].mean()}"
                        )
                    else:
                        rollback_limited = 0
                    if rollback_limited > 3 and rollback_happened < 10: #需要复位TODO: 可以增加force的判断 预测的状态和实际不一致
                        # g_seed=g_seed+1
                        print(
                            f"[DO ROLLBACK] idx={idx}, slug={slug}, i={i},{g_seed} "
                            f"sta_diff={sta_diff}, thr={STA_DIFF_THR}"
                        )
                        # 回滚 sub 内部状态
                        '''obs, _ = restore_perfect_sub_checkpoint(
                            sub,
                            reset_recovery,
                            refresh_derived=False,
                        )'''
                        restore_info = restore_robot_gripper_checkpoint(
                            sub,
                            reset_recovery,
                            restore_ctrl=True,
                            restore_act=True,
                            restore_mocap=True,
                            align_mocap_to_hand=True,
                            refresh_derived=True,
                            verify_others_unchanged=True,
                        )
                        
                        # 同步回滚你的 Python 侧历史
                        Image_HIST = []
                        STATE_HIST = []
                        ACTION_HIST = []
                        sta_diff_list = []
                        rollback_limited = 0
                        rollback_happened = rollback_happened + 1
                        
                        try:
                            a0 = np.zeros(sub.action_space.shape, dtype=np.float32)
                            a0 = np.clip(a0, sub.action_space.low, sub.action_space.high)
                            obs, _, terminated, truncated, info = sub.step(a0)
                            if terminated or truncated:
                                done = True
                                break
                        except Exception:
                            if "truncate==True" in str(e):
                                done = True
                                break
                        continue
                    
                    #######################
                    
                    if np.abs(hand_sig[HORIZON]-hand_sig[0])>0.1:
                        LOC_HORIZON = 5 #step减小
                        print(
                            f"[SLOW DOWN] idx={idx}, slug={slug},rollback_limited={rollback_limited}, dff={hand_sig[HORIZON]}-{hand_sig[0]},{state_vec[3]}, "
                        )
                        
                        #if state_vec[3]<hand_sig[HORIZON] and actions[:15,3].sum()>0:
                        
                        
                        
                        
                   
                    for i in range(LOC_HORIZON):
                        a4 = np.asarray(actions[i][:4], dtype=np.float32)
                        a4 = np.clip(a4, sub.action_space.low, sub.action_space.high)
                        
                        state_vec = obs_to_state(obs)
                        
                        sta_diff = np.abs(np.array(state_vec)[:4] - states[i, :4])
                        
                        if sta_diff[:3].max() > 0.05 and i > 0 and False:
                            print(
                                f"[SKIP] idx={idx}, slug={slug}, i={i}, "
                                f"sta_diff={sta_diff}, thr={STA_DIFF_THR}"
                            )
                            break
                        
                        sta_diff_list.append(sta_diff.mean())
                        
                        STATE_HIST.append(np.array(state_vec))
                        img_bgr = render_single_bgr(sub)
                        Image_HIST.append(img_bgr)
                        ACTION_HIST.append(a4)
                        
                        write_video(video_writer, img_bgr)
                        
                        #import pdb;pdb.set_trace()
                        try:
                            obs, _, terminated, truncated, info = sub.step(a4)
                        except ValueError as e:
                            if "truncate==True" in str(e):
                                done = True
                                break
                            #raise
                        
                        steps += 1

                        if isinstance(info, dict) and info.get("success", 0) == 1:
                            cur_success = True
                            success_counts[idx] += 1
                            if gname_for_task is not None:
                                group_success[gname_for_task] += 1
                            done = True
                            break

                        if terminated or truncated or steps >= episode_horizon:
                            done = True
                            break
                    #print([float(v) for v in sta_diff_list ])
                    #for _ in range(10):
                    #    obs, _, _, _, _ = sub.step(a0)
                    
                    Image_HIST = Image_HIST[-4:]
                    #if len(STATE_HIST)>50:
                    #    import pdb;pdb.set_trace()
                    STATE_HIST = STATE_HIST[-4:]
                    ACTION_HIST = ACTION_HIST[-4:]
                    
                    #write_video(video_writer, np.zeros_like(img_bgr))

                
                # close video writer
                if done and SAVE_VIDEO:
                    final_frame = render_single_bgr(sub)
                    write_video(video_writer, final_frame)
                    save_episode_video(video_writer, video_name, idx, slug, ep + 1)
                    video_path = os.path.join(VIDEO_SAVE_DIR, video_name)
                    if cur_success:
                        os.system(f'mv {video_path}  {video_path}.success.mp4')
                
          
            s = success_counts[idx]
            t = trials_counts[idx]
            task_rate = s / max(1, t)
            msg = (f"[Task {idx} {slug}] {task_prompt} finished {num_eval_episodes} episodes -> "
                  f"success_rate={task_rate:.3f}  (s={s}, t={t})")
            log_write(msg)

    envs.close()

    # 5) Build metrics
    per_task: Dict[str, float] = {}
    for idx in ordered_indices:
        slug = idx_to_slug.get(idx, f"task-{idx}")
        s, t = success_counts[idx], trials_counts[idx]
        per_task[slug] = (s / t) if t > 0 else 0.0

    per_group: Dict[str, float] = {}
    for gname in ["easy", "medium", "hard", "very_hard"]:
        s, t = group_success[gname], group_trials[gname]
        per_group[gname] = (s / t) if t > 0 else 0.0

    overall = (sum(success_counts.values()) /
               max(1, sum(trials_counts.values())))

    return per_task, per_group, overall


# ---------------- Entrypoint ----------------
async def _amain():
    per_task, per_group, overall = await eval_mt50_with_groups(
        server_url=SERVER_URL,
        num_eval_episodes=EPISODES,
        episode_horizon=EPISODE_HORIZON,
        seed=SEED,
    )

    # Pretty print
    # print("\n==== Per-task success rate ====")
    # for slug, rate in per_task.items():
    #     print(f"{slug:24s}  {rate:.3f}")

    # print("\n==== Difficulty buckets ====")
    # print(f"easy      : {per_group.get('easy', 0.0):.3f}")
    # print(f"medium    : {per_group.get('medium', 0.0):.3f}")
    # print(f"hard      : {per_group.get('hard', 0.0):.3f}")
    # print(f"very_hard : {per_group.get('very_hard', 0.0):.3f}")

    avg = (per_group.get('easy', 0.0) + per_group.get('medium', 0.0) + per_group.get('hard', 0.0) + per_group.get('very_hard', 0.0)) / 4
    # print(f"\n==== Overall Average as Success Rate ====\n{avg:.3f}")

    # log
    log_write(f"\n==== Evaluation Log ====\nLog file: {LOG_PATH}")
    log_write(f"Target difficulty: {TARGET_LEVEL}")
    log_write(f"Server URL: {SERVER_URL}")
    log_write(f"Episodes per task: {EPISODES}")
    log_write(f"Episode horizon: {EPISODE_HORIZON}")
    log_write(f"HORIZON: {HORIZON}")
    log_write(f"Seed: {SEED}\n")
    

    log_write("==== Per-task success rate ====")
    for slug, rate in per_task.items():
        log_write(f"{slug:24s}  {rate:.3f}")

    log_write("\n==== Difficulty buckets ====")
    log_write(f"easy      : {per_group.get('easy', 0.0):.3f}")
    log_write(f"medium    : {per_group.get('medium', 0.0):.3f}")
    log_write(f"hard      : {per_group.get('hard', 0.0):.3f}")
    log_write(f"very_hard : {per_group.get('very_hard', 0.0):.3f}")

    log_write(f"\n==== Overall Average as Success Rate ====\n{avg:.3f}")



if __name__ == "__main__":
    seed_everything(SEED)
    asyncio.run(_amain())



# if __name__ == "__main__":
#     N_REPEAT = 1
#     for run_id in range(N_REPEAT):
#         print(f"\n\n===== 🌟 Run {run_id + 1}/{N_REPEAT} =====")
#         asyncio.run(_amain())
