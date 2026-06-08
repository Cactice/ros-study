# AgiBot G2 + SmolVLA (MuJoCo)

Run SmolVLA on the AgiBot G2 robot in MuJoCo simulation.

## Setup

```bash
uv sync
```

## Run

```bash
uv run smolvla
```

Edit `TASK` in [smolvla_mujoco.py](smolvla_mujoco.py) to change the instruction.

## Notes

- Uses `smolvla_base` pretrained on SO-100 (6-DOF), zero-shot on G2's right arm
