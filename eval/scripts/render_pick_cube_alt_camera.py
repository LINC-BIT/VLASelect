import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs.pick_cube_alt_camera  # noqa: F401


def image_to_numpy(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCubeAltCamera-v1")
    parser.add_argument("--output", default="vis/pick_cube_alt_camera.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-backend", default="auto")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.env_id,
        obs_mode="state",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
    )
    try:
        env.reset(seed=args.seed)
        image = image_to_numpy(env.render())
        Image.fromarray(image).save(output_path)
    finally:
        env.close()

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
