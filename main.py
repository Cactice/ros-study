import os
import pathlib

import jax
import mujoco.viewer

import mujoco
from mujoco import mjx

# MJX does not yet support Metal; force CPU so JIT still compiles.
jax.config.update("jax_platform_name", "cpu")

MUJOCO_DIR = pathlib.Path(__file__).parent / "mujoco"
SCENE = "robot_only.xml"


def main() -> None:
    os.chdir(MUJOCO_DIR)

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    mx = mjx.put_model(model)
    dx = mjx.put_data(model, data)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    jit_step = jax.jit(mjx.step)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -20
        while v.is_running():
            dx = jit_step(mx, dx)
            mjx.get_data_into(data, model, dx)
            v.sync()


if __name__ == "__main__":
    main()
