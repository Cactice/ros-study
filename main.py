import os
import pathlib

import jax
import mujoco.viewer

import mujoco
from mujoco import mjx

MUJOCO_DIR = pathlib.Path(__file__).parent / "mujoco"
SCENE = "robot_only.xml"


def main() -> None:
    os.chdir(MUJOCO_DIR)

    # Load model
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Put model and data on device (Metal GPU on Apple Silicon)
    mx = mjx.put_model(model)
    dx = mjx.put_data(model, data)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # JIT-compile the MJX step
    jit_step = jax.jit(mjx.step)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -20
        while v.is_running():
            # Step physics on GPU via MJX
            dx = jit_step(mx, dx)
            # Copy back to CPU for rendering
            mjx.get_data_into(data, model, dx)
            v.sync()


if __name__ == "__main__":
    main()
