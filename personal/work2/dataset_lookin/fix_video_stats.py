"""
修复视频数据集的 stats.json
从实际视频数据中计算 camera keys 的真实统计值
如果原来的数据集已经丢失了
python personal/work2/dataset_lookin/fix_video_stats.py
"""

import json
import numpy as np
from pathlib import Path
import subprocess
import cv2
from tqdm import tqdm

# 数据集路径
DATASET_ROOT = Path("/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3")
STATS_FILE = DATASET_ROOT / "meta" / "stats.json"
META_DIR = DATASET_ROOT / "meta"
DATA_DIR = DATASET_ROOT / "data"

# Camera keys
CAMERA_KEYS = ["observation.images.top", "observation.images.wrist"]

def get_video_files(camera_key: str) -> list[Path]:
    """获取指定 camera 的所有视频文件"""
    # 视频路径格式: videos/{camera_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4
    video_dir = DATASET_ROOT / "videos" / camera_key
    if not video_dir.exists():
        return []
    
    video_files = list(video_dir.glob("**/*.mp4"))
    return sorted(video_files)

def extract_frames_from_video(video_path: Path) -> list[np.ndarray]:
    """从视频中提取所有帧,使用 ffmpeg 以支持更多编码格式(如 AV1)"""
    frames = []
    
    # 方法1: 尝试使用 ffmpeg 直接提取帧(支持 AV1 等所有编码)
    try:
        import subprocess
        
        # 使用 ffmpeg 将视频帧输出到 stdout
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', 'format=rgb24',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-'
        ]
        
        # 获取视频信息
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  警告: ffprobe 失败 {video_path}: {result.stderr}")
            return frames
        
        info = result.stdout.strip().split(',')
        if len(info) < 3:
            print(f"  警告: 无法获取视频信息 {video_path}")
            return frames
        
        width, height, total_frames = int(info[0]), int(info[1]), int(info[2])
        
        if total_frames == 0:
            return frames
        
        # 使用 ffmpeg 提取所有帧
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )
        
        frame_size = width * height * 3
        
        for _ in range(total_frames):
            raw_frame = proc.stdout.read(frame_size)
            if len(raw_frame) == frame_size:
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                # 归一化到 [0, 1]
                frame_rgb = frame.astype(np.float32) / 255.0
                frames.append(frame_rgb)
        
        proc.wait(timeout=60)
        
        if frames:
            return frames
            
    except Exception as e:
        print(f"  警告: ffmpeg 方法失败 {video_path}: {e}")
    
    # 方法2: 回退到 OpenCV(逐帧读取,不使用 seek)
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  警告: 无法打开视频 {video_path}")
            return frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV 读取的是 BGR,转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 归一化到 [0, 1]
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_rgb)
        
        cap.release()
    except Exception as e:
        print(f"  警告: OpenCV 方法也失败 {video_path}: {e}")
    
    return frames

def compute_camera_stats(camera_key: str) -> dict:
    """计算单个 camera 的统计值,使用流式处理以节省内存"""
    print(f"\n计算 {camera_key} 的统计值...")
    
    video_files = get_video_files(camera_key)
    if not video_files:
        print(f"  警告: 没有找到 {camera_key} 的视频文件")
        return None
    
    print(f"  找到 {len(video_files)} 个视频文件")
    
    # 使用流式处理,避免将所有帧加载到内存
    total_frames_processed = 0
    total_pixels = None  # 记录每个通道的总像素数 (C,)
    
    # 用于在线计算统计值的变量
    sum_pixels = None  # 用于计算 mean
    sum_sq_pixels = None  # 用于计算 std
    global_min = None
    global_max = None
    
    # 为了计算分位数,采样部分帧(如果帧数太多)
    sampled_frames = []
    max_sample_frames = 1000  # 最多采样1000帧用于分位数计算
    
    for video_path in tqdm(video_files, desc=f"  处理 {camera_key}"):
        frames = extract_frames_from_video(video_path)
        if not frames:
            continue
        
        for frame in frames:
            # frame shape: (H, W, C)
            # 转换为 (C, H, W) 格式
            frame_chw = np.transpose(frame, (2, 0, 1))  # (C, H, W)
            
            # 获取帧的 H, W
            h, w = frame_chw.shape[1], frame_chw.shape[2]
            pixels_per_frame = h * w
            
            # 更新统计值
            if sum_pixels is None:
                total_pixels = np.array([pixels_per_frame, pixels_per_frame, pixels_per_frame], dtype=np.float64)
                sum_pixels = frame_chw.sum(axis=(1, 2)).astype(np.float64)  # (C,)
                sum_sq_pixels = (frame_chw ** 2).sum(axis=(1, 2)).astype(np.float64)  # (C,)
                global_min = frame_chw.min(axis=(1, 2))  # (C,)
                global_max = frame_chw.max(axis=(1, 2))  # (C,)
            else:
                total_pixels += pixels_per_frame
                sum_pixels += frame_chw.sum(axis=(1, 2))
                sum_sq_pixels += (frame_chw ** 2).sum(axis=(1, 2))
                global_min = np.minimum(global_min, frame_chw.min(axis=(1, 2)))
                global_max = np.maximum(global_max, frame_chw.max(axis=(1, 2)))
            
            # 采样帧用于分位数计算
            if len(sampled_frames) < max_sample_frames:
                sampled_frames.append(frame_chw)
            
            total_frames_processed += 1
    
    if total_frames_processed == 0:
        print(f"  警告: 没有从 {camera_key} 中提取到帧")
        return None
    
    print(f"  共处理 {total_frames_processed} 帧")
    print(f"  总像素数(每通道): {total_pixels}")
    
    # 计算 mean 和 std (除以总像素数,而不是帧数!)
    mean = (sum_pixels / total_pixels).tolist()  # (C,)
    variance = (sum_sq_pixels / total_pixels) - (sum_pixels / total_pixels) ** 2
    std = np.sqrt(np.maximum(variance, 0)).tolist()  # (C,)
    
    # min 和 max
    min_vals = global_min.tolist()  # (C,)
    max_vals = global_max.tolist()  # (C,)
    
    # 分位数: 使用采样的帧
    sampled_array = np.array(sampled_frames)  # (N, C, H, W)
    sampled_flat = sampled_array.reshape(-1, sampled_array.shape[1])  # (N*H*W, C)
    
    q01 = np.percentile(sampled_flat, 1, axis=0).tolist()
    q10 = np.percentile(sampled_flat, 10, axis=0).tolist()
    q50 = np.percentile(sampled_flat, 50, axis=0).tolist()
    q90 = np.percentile(sampled_flat, 90, axis=0).tolist()
    q99 = np.percentile(sampled_flat, 99, axis=0).tolist()
    
    stats = {
        "min": min_vals,
        "max": max_vals,
        "mean": mean,
        "std": std,
        "count": total_frames_processed,
        "q01": q01,
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "q99": q99
    }
    
    print(f"  统计值计算完成:")
    print(f"    mean: {mean}")
    print(f"    std: {std}")
    print(f"    min: {min_vals}")
    print(f"    max: {max_vals}")
    
    return stats

def main():
    print(f"修复数据集: {DATASET_ROOT}")
    print(f"Stats 文件: {STATS_FILE}")
    
    if not STATS_FILE.exists():
        print("错误: stats.json 不存在!")
        return
    
    # 加载现有 stats
    with open(STATS_FILE, 'r') as f:
        stats = json.load(f)
    
    print(f"\n现有 stats keys:")
    for key in stats.keys():
        print(f"  - {key}")
    
    # 检查缺失的 camera keys
    missing_keys = [key for key in CAMERA_KEYS if key not in stats]
    
    if not missing_keys:
        print("\n所有 camera keys 已存在，无需修复!")
        return
    
    print(f"\n缺失的 camera keys:")
    for key in missing_keys:
        print(f"  - {key}")
    
    # 创建备份
    backup_file = STATS_FILE.with_suffix('.json.backup')
    print(f"\n创建备份: {backup_file}")
    import shutil
    shutil.copy(STATS_FILE, backup_file)
    
    # 计算并添加缺失的 camera stats
    for key in missing_keys:
        camera_stats = compute_camera_stats(key)
        if camera_stats:
            stats[key] = camera_stats
            print(f" 已添加: {key}")
        else:
            print(f" 无法计算: {key}")
    
    # 保存更新后的 stats
    print(f"\n保存更新后的 stats...")
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n修复完成!")
    print(f"现在 stats 包含 {len(stats)} 个 keys")
    
    # 验证
    print(f"\n验证 camera keys:")
    for key in CAMERA_KEYS:
        if key in stats:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key} (仍然缺失)")

if __name__ == "__main__":
    main()