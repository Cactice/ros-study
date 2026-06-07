import os
import pathlib

import mujoco
import mujoco.viewer

MUJOCO_DIR = pathlib.Path(__file__).parent / "mujoco"
SCENE = "robot_only.xml"


def main() -> None:
    os.chdir(MUJOCO_DIR)
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -20
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
