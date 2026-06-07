"""
RDT-1B + MuJoCo G2 integration.

Pipeline:
  MuJoCo cameras (3x) + joint state -> RDT-1B -> joint position commands -> MuJoCo

T5-XXL (22GB) does not fit on M2 Mac. Options:
  1. Load precomputed embedding from lang_embed.pt  (preferred)
  2. Pass zero embedding (model runs without language guidance — for demo only)

To precompute on a CUDA machine:
    python precompute_lang.py --task "pick up the workpiece and place it on the table"
"""
import os
import pathlib
import sys
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower

MUJOCO_DIR = pathlib.Path(__file__).parent / "mujoco"
SCENE = "robot_only.xml"

# RDT config
CONFIG_PATH = pathlib.Path(__file__).parent / "configs" / "base.yaml"
PRETRAINED_RDT = "robotics-diffusion-transformer/rdt-170m"
PRETRAINED_SIGLIP = "google/siglip-so400m-patch14-384"
LANG_EMBED_PATH = pathlib.Path(__file__).parent / "lang_embed.pt"

# rdt-170m specific config (hidden_size=1024, depth=14)
RDT_170M_MODEL_CONFIG = {
    "rdt": {"hidden_size": 1024, "depth": 14, "num_heads": 32,
            "cond_pos_embed_type": "multimodal"},
    "lang_adaptor": "mlp2x_gelu",
    "img_adaptor": "mlp2x_gelu",
    "state_adaptor": "mlp3x_gelu",
    "noise_scheduler": {
        "type": "ddpm", "num_train_timesteps": 1000,
        "num_inference_timesteps": 5, "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": False, "prediction_type": "sample",
    },
}

# G2 joint indices in MuJoCo qpos (from keyframe comment in XML):
# qpos[7-11]:  body (5)
# qpos[12-18]: arm_l (7)
# qpos[19-20]: gripper_l (2)
# qpos[21-27]: arm_r (7)
# qpos[28-29]: gripper_r (2)
ARM_R_QPOS = list(range(21, 28))   # 7 joints
ARM_L_QPOS = list(range(12, 19))   # 7 joints
GRIPPER_R_QPOS = [28]
GRIPPER_L_QPOS = [19]

# RDT state vector indices
R_ARM = list(range(0, 7))
R_GRIP = [10]
L_ARM = list(range(50, 57))
L_GRIP = [60]


def build_state_vec(data: mujoco.MjData) -> np.ndarray:
    """Map MuJoCo qpos -> 128-dim RDT state vector."""
    vec = np.zeros(128, dtype=np.float32)
    vec[R_ARM] = data.qpos[ARM_R_QPOS]
    vec[R_GRIP] = data.qpos[GRIPPER_R_QPOS]
    vec[L_ARM] = data.qpos[ARM_L_QPOS]
    vec[L_GRIP] = data.qpos[GRIPPER_L_QPOS]
    return vec


def apply_action(data: mujoco.MjData, action: np.ndarray) -> None:
    """Write RDT 128-dim action into MuJoCo ctrl."""
    # RDT predicts joint positions; map back to qpos indices via ctrl
    for i, qi in enumerate(ARM_R_QPOS):
        data.qpos[qi] = float(action[R_ARM[i]])
    for i, qi in enumerate(ARM_L_QPOS):
        data.qpos[qi] = float(action[L_ARM[i]])
    data.qpos[GRIPPER_R_QPOS[0]] = float(action[R_GRIP[0]])
    data.qpos[GRIPPER_L_QPOS[0]] = float(action[L_GRIP[0]])


def render_cameras(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData):
    """Render 3 cameras: exterior, right wrist, left wrist. Returns list of PIL Images."""
    images = []
    cam_names = ["track", "gripper_r_camera_link", "gripper_l_camera_link"]
    for cam in cam_names:
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        except Exception:
            cam_id = 0
        renderer.update_scene(data, camera=cam_id)
        img = renderer.render()
        images.append(Image.fromarray(img).resize((384, 384)))
    return images


def load_model(device: str):
    """Load SigLIP vision encoder and RDT policy."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)  # only used for dataset/common params

    print("Loading SigLIP vision encoder...")
    vision_tower = SiglipVisionTower(PRETRAINED_SIGLIP, args=None)
    vision_tower.load_model()
    vision_tower = vision_tower.to(device)

    num_patches = vision_tower.num_patches
    img_cond_len = (
        config["common"]["img_history_size"]
        * config["common"]["num_cameras"]
        * num_patches
    )

    print(f"Loading RDT policy ({PRETRAINED_RDT})...")
    from huggingface_hub import hf_hub_download
    policy = RDTRunner(
        action_dim=128,
        pred_horizon=64,
        config=RDT_170M_MODEL_CONFIG,
        lang_token_dim=4096,
        img_token_dim=1152,
        state_token_dim=128,
        max_lang_cond_len=1024,
        img_cond_len=img_cond_len,
        img_pos_embed_config=[("image", (2, 3, -num_patches))],
        lang_pos_embed_config=[("lang", -1024)],
        dtype=torch.float32,
    )
    ckpt = hf_hub_download(PRETRAINED_RDT, "pytorch_model.bin")
    state_dict = torch.load(ckpt, map_location="cpu")
    policy.load_state_dict(state_dict, strict=True)
    policy = policy.to(device)
    policy.eval()

    return vision_tower, policy, config


def get_lang_embed(device: str, config: dict) -> torch.Tensor:
    """Load precomputed embedding or return zeros."""
    lang_len = config["dataset"]["tokenizer_max_length"]
    lang_dim = config["model"]["lang_token_dim"]
    if LANG_EMBED_PATH.exists():
        print(f"Loading language embedding from {LANG_EMBED_PATH}")
        embed = torch.load(LANG_EMBED_PATH, map_location=device)
    else:
        print("WARNING: No lang_embed.pt found. Using zero embedding (no language guidance).")
        print("Run precompute_lang.py on a machine with enough RAM to generate it.")
        embed = torch.zeros(1, lang_len, lang_dim, dtype=torch.float32, device=device)
    return embed


@torch.inference_mode()
def encode_images(vision_tower, images: list, processor, device: str) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = vision_tower(**inputs).last_hidden_state  # (N, patches, dim)
    return feats.to(torch.float32)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    os.chdir(MUJOCO_DIR)
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    renderer = mujoco.Renderer(model, height=384, width=384)

    vision_tower, policy, config = load_model(device)
    lang_embed = get_lang_embed(device, config)

    # image processor from SigLIP
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(PRETRAINED_SIGLIP)

    chunk_size = config["common"]["action_chunk_size"]
    obs_window = deque(maxlen=2)
    action_queue: list = []

    print("\nStarting simulation. Close the viewer window to quit.")

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -20
        step = 0

        while v.is_running():
            mujoco.mj_step(model, data)

            # Infer new actions every chunk_size steps
            if step % chunk_size == 0:
                imgs = render_cameras(renderer, model, data)
                obs_window.append(imgs)
                if len(obs_window) < 2:
                    obs_window.appendleft(obs_window[0])  # pad history

                # Encode images: shape (2*3, patches, dim) -> (1, 2*3*patches, dim)
                all_imgs = [img for frame in obs_window for img in frame]
                img_feats = encode_images(vision_tower, all_imgs, processor, device)
                img_feats = img_feats.unsqueeze(0)  # (1, N*patches, dim)

                state = torch.tensor(
                    build_state_vec(data), dtype=torch.float32, device=device
                ).unsqueeze(0)  # (1, 128)
                state_mask = (state != 0).to(torch.float32)

                with torch.no_grad():
                    actions = policy.predict_action(
                        lang_tokens=lang_embed,
                        lang_attn_mask=torch.ones(
                            lang_embed.shape[:2], dtype=torch.bool, device=device
                        ),
                        img_tokens=img_feats,
                        state_tokens=state.unsqueeze(1),   # (1, 1, 128)
                        action_mask=state_mask.unsqueeze(1),  # (1, 1, 128)
                        ctrl_freqs=torch.tensor([25.0], device=device),
                    )  # (1, chunk_size, 128)
                actions = actions.squeeze(0).float().cpu().numpy()
                action_queue = list(actions)

            if action_queue:
                apply_action(data, action_queue.pop(0))

            v.sync()
            step += 1


if __name__ == "__main__":
    main()
