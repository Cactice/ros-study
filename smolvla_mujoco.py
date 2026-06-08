"""
SmolVLA (450M) + MuJoCo G2 integration.

Run with:  uv run smolvla

smolvla_base was trained on SO-100 (6-DOF). Action output is 6-dim, so we drive
G2's right arm (6 of 7 joints). Left arm holds the keyframe pose.

Edit TASK to change the instruction. No separate embedding step needed.
"""

import os
import pathlib
import subprocess
import sys

import mujoco.viewer
import numpy as np
import torch
from PIL import Image

import mujoco

_HERE = pathlib.Path(__file__).parent
_MJPYTHON = str(_HERE / ".venv/bin/mjpython")

MUJOCO_DIR = _HERE / "mujoco"
SCENE = "robot_only.xml"
PRETRAINED = "lerobot/smolvla_base"

TASK = "dance randomly"

# G2 qpos addresses (verified from mj_resetDataKeyframe inspection)
BODY_QPOS = list(range(7, 12))
ARM_L_QPOS = list(range(12, 19))
ARM_R_QPOS = list(range(21, 28))
GRIPPER_L_QPOS = [20]
GRIPPER_R_QPOS = [29]

# data.ctrl indices for position actuators
BODY_CTRL = list(range(0, 5))
ARM_L_CTRL = list(range(24, 31))
ARM_R_CTRL = list(range(31, 38))
GRIPPER_L_CTRL = [52]
GRIPPER_R_CTRL = [53]

# smolvla_base: action_dim=6 → drive right arm joints 1-6 (skip joint 7)
ACTION_CTRL = ARM_R_CTRL[:6]  # ctrl[31:37]
STATE_QPOS = ARM_R_QPOS[:6]  # qpos[21:27]

IMG_W, IMG_H = 256, 256


def launch() -> None:
    """Entry point for `uv run smolvla` — spawns venv mjpython so the viewer works on macOS."""
    sys.exit(subprocess.run([_MJPYTHON, __file__] + sys.argv[1:]).returncode)


def init_ctrl(data: mujoco.MjData) -> None:
    for i, ci in enumerate(BODY_CTRL):
        data.ctrl[ci] = data.qpos[BODY_QPOS[i]]
    for i, ci in enumerate(ARM_L_CTRL):
        data.ctrl[ci] = data.qpos[ARM_L_QPOS[i]]
    for i, ci in enumerate(ARM_R_CTRL):
        data.ctrl[ci] = data.qpos[ARM_R_QPOS[i]]
    data.ctrl[GRIPPER_L_CTRL[0]] = data.qpos[GRIPPER_L_QPOS[0]]
    data.ctrl[GRIPPER_R_CTRL[0]] = data.qpos[GRIPPER_R_QPOS[0]]


def render_cameras(
    renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData
) -> list[Image.Image]:
    cam_names = ["track", "gripper_r_camera_link", "gripper_l_camera_link"]
    images = []
    for cam in cam_names:
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        except Exception:
            cam_id = 0
        renderer.update_scene(data, camera=cam_id)
        images.append(Image.fromarray(renderer.render()).resize((IMG_W, IMG_H)))
    return images


def img_to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)  # (3, H, W)


def load_policy(device: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"Loading SmolVLA from {PRETRAINED}...")
    policy = SmolVLAPolicy.from_pretrained(PRETRAINED)
    policy = policy.to(device)
    policy.eval()
    print(f"Loaded. action_dim={policy.config.output_features['action'].shape[0]}")
    return policy


def tokenize_task(policy, task: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    max_len = policy.config.tokenizer_max_length
    enc = tokenizer(
        task,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].bool().to(device)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    os.chdir(MUJOCO_DIR)
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    init_ctrl(data)

    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    policy = load_policy(device)

    lang_tokens, lang_mask = tokenize_task(policy, TASK, device)

    chunk_size = policy.config.n_action_steps
    action_queue: list[np.ndarray] = []

    print(f"\nTask: '{TASK}'")
    print("Starting simulation. Close the viewer window to quit.\n")

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -20
        step = 0

        while v.is_running():
            mujoco.mj_step(model, data)

            if step % chunk_size == 0:
                imgs = render_cameras(renderer, model, data)

                state = torch.tensor(
                    data.qpos[STATE_QPOS].astype(np.float32), device=device
                ).unsqueeze(0)  # (1, 6)

                obs = {
                    "observation.state": state,
                    "observation.images.camera1": img_to_tensor(imgs[0], device).unsqueeze(0),
                    "observation.images.camera2": img_to_tensor(imgs[1], device).unsqueeze(0),
                    "observation.images.camera3": img_to_tensor(imgs[2], device).unsqueeze(0),
                    "observation.language.tokens": lang_tokens,
                    "observation.language.attention_mask": lang_mask,
                    "task": [TASK],
                }

                with torch.inference_mode():
                    out = policy.select_action(obs)

                # out shape: (1, 6) or (1, chunk, 6) depending on LeRobot version
                actions = out.squeeze(0).cpu().numpy()
                action_queue = list(actions) if actions.ndim == 2 else [actions]

            if action_queue:
                act = action_queue.pop(0)
                for i, ci in enumerate(ACTION_CTRL):
                    data.ctrl[ci] = float(act[i])

            v.sync()
            step += 1


if __name__ == "__main__":
    main()
