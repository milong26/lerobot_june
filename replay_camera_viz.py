#!/usr/bin/env python
"""
Replay dataset episode first frame as reference overlay on live RealSense camera feed.

直接运行: python replay_camera_viz.py

Keyboard Controls (在 matplotlib 窗口中按):
    q / ESC - Quit
    s       - Save current camera frame
    r       - Toggle reference image visibility
    + / =   - Adjust reference image opacity (increase)
    -       - Adjust reference image opacity (decrease)
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from lerobot.datasets import LeRobotDataset
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.cameras import ColorMode

# ==================== 配置 ====================
REPO_ID = "test/first"
ROOT = "/home/qwe/.cache/huggingface/lerobot/ep10/cylinder_standing"
EPISODE_INDEX = 0
CAMERA_SERIAL = "806312060427"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
OUTPUT_DIR = "./replay_output"
INITIAL_OPACITY = 0.3
# ==============================================


def get_first_frame(dataset: LeRobotDataset, episode_index: int) -> np.ndarray:
    """Get the first frame of the specified episode from the dataset."""
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]

    camera_keys = dataset.meta.camera_keys
    if not camera_keys:
        raise ValueError("No camera keys found in dataset")

    first_camera_key = camera_keys[1]
    print(f"  Camera key: {first_camera_key}")

    frame = dataset[from_idx][first_camera_key]

    if hasattr(frame, "numpy"):
        frame = frame.numpy()

    print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"  Frame min: {frame.min()}, max: {frame.max()}")

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    # CHW -> HWC
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)

    print(f"  After conversion - shape: {frame.shape}, dtype: {frame.dtype}")

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"Invalid frame dimensions: {frame.shape}")

    return frame


def on_key_press(event):
    global quit_flag, show_reference, opacity, save_frame_flag
    key = event.key
    if key in ["q", "escape"]:
        quit_flag = True
    elif key == "s":
        save_frame_flag = True
    elif key == "r":
        show_reference = not show_reference
        print(f"Reference image: {'ON' if show_reference else 'OFF'}")
    elif key in ["+", "equal"]:
        opacity = min(1.0, opacity + 0.05)
        print(f"Opacity increased to: {opacity:.2f}")
    elif key == "-":
        opacity = max(0.0, opacity - 0.05)
        print(f"Opacity decreased to: {opacity:.2f}")


def main():
    global quit_flag, show_reference, opacity, save_frame_flag

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {REPO_ID}")
    print(f"Root: {ROOT}")
    print(f"Episode: {EPISODE_INDEX}")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=ROOT,
        episodes=[EPISODE_INDEX],
    )

    print("Extracting first frame from dataset...")
    reference_frame = get_first_frame(dataset, EPISODE_INDEX)

    reference_pil = PILImage.fromarray(reference_frame)
    reference_pil = reference_pil.resize((CAMERA_WIDTH, CAMERA_HEIGHT), PILImage.LANCZOS)
    reference_frame = np.array(reference_pil)

    # RGB -> BGR (相机返回的是 BGR，需要统一格式才能正确叠加)
    reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_RGB2BGR)

    reference_path = output_dir / "reference_frame.png"
    reference_pil.save(str(reference_path))
    print(f"Reference frame saved to: {reference_path}")
    print(f"Reference frame size: {reference_frame.shape}")

    print(f"\nConnecting to RealSense camera (serial: {CAMERA_SERIAL})...")
    camera_config = RealSenseCameraConfig(
        serial_number_or_name=CAMERA_SERIAL,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        color_mode=ColorMode.BGR,
        use_depth=False,
    )

    camera = RealSenseCamera(camera_config)
    camera.connect()
    print("Camera connected successfully!")

    opacity = INITIAL_OPACITY
    show_reference = True
    frame_count = 0
    quit_flag = False
    save_frame_flag = False

    print("\nKeyboard Controls:")
    print("  q / ESC - Quit")
    print("  s       - Save current camera frame")
    print("  r       - Toggle reference image visibility")
    print("  + / -   - Adjust reference image opacity")
    print(f"\nInitial opacity: {opacity:.2f}")

    fig, ax = plt.subplots(figsize=(CAMERA_WIDTH / 100, CAMERA_HEIGHT / 100), dpi=100)
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.manager.set_window_title("RealSense Camera with Reference Overlay")
    ax.axis("off")
    plt.show(block=False)

    try:
        while not quit_flag:
            start_time = time.perf_counter()

            try:
                camera_frame = camera.read()
            except Exception as e:
                print(f"Error reading frame: {e}")
                continue

            display_frame = camera_frame.copy()

            if show_reference:
                display_frame = cv2.addWeighted(display_frame, 1.0, reference_frame, opacity, 0)

            info_text = f"Frame: {frame_count} | Opacity: {opacity:.2f} | Ref: {'ON' if show_reference else 'OFF'}"
            cv2.putText(
                display_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            ax.clear()
            ax.imshow(display_rgb)
            ax.axis("off")
            plt.pause(0.001)

            if save_frame_flag:
                save_path = output_dir / f"camera_frame_{frame_count:06d}.png"
                cv2.imwrite(str(save_path), camera_frame)
                print(f"Saved camera frame to: {save_path}")
                save_frame_flag = False

            frame_count += 1

            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (1.0 / CAMERA_FPS) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        camera.disconnect()
        plt.close(fig)
        print("Camera disconnected. Done!")


if __name__ == "__main__":
    main()