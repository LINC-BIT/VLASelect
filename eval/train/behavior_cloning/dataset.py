import os
import h5py
import numpy as np
from mani_skill.utils.io_utils import load_json, dump_json
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from PIL import Image

PANDA_MASK = [True, True, True, True, True, True, False]

# taken from here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        cameras=("base_camera",),
        load_count=-1,
        normalize_states=False,
        need_states=False,
        task_name="maniskill",
        task_instruction="pick up the cube",
        preprocess_fn=None
    ):
        self.dataset_file = dataset_file
        self.cameras = cameras
        self.task_name = task_name
        self.task_instruction = task_instruction
        self.need_states = need_states
        self.preprocess_fn = preprocess_fn

        self.h5 = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)

        self.episodes = self.json_data["episodes"]

        if load_count is None or load_count < 0:
            load_count = len(self.episodes)

        # ---------- build index ----------
        self.index = []
        for eps in self.episodes[:load_count]:
            traj_key = f"traj_{eps['episode_id']}"
            T = self.h5[traj_key]["actions"].shape[0]
            for t in range(T):
                self.index.append((traj_key, t))

        # ---------- optional state normalization ----------
        self.state_mean = None
        self.state_std = None
        if normalize_states:
            self._load_or_compute_state_stats()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        traj_key, t = self.index[idx]
        traj = self.h5[traj_key]

        # ------------------------------------------------
        # RGB (choose ONE camera for VLA compatibility)
        # ------------------------------------------------
        if len(self.cameras) == 1:
            cam = self.cameras[0]  # ⭐ 明确选择一个 camera（如 base_camera）
        else:
            raise NotImplementedError('暂不支持多摄像机输入')

        rgb = traj["obs"]["sensor_data"][cam]["rgb"][t]  # (H, W, 3), uint8
        rgb = rgb.astype(np.uint8)

        # 转成 PIL.Image（OpenVLA processor 期望）
        image = Image.fromarray(rgb)

        # ------------------------------------------------
        # action
        # ------------------------------------------------
        action = traj["actions"][t].astype(np.float32)
        extra = traj["obs"]["extra"]

        sample = {
            "images": image,                     # ✅ 单张 PIL.Image
            "instruction": self.task_instruction,
            "action": action,
            "extra": {
                "tcp_pose": extra["tcp_pose"][t],
                "obj_pose": extra["obj_pose"][t],
                "goal_pos": extra["goal_pos"][t],
            }
        }

        # ------------------------------------------------
        # optional low-dim states
        # ------------------------------------------------
        if self.need_states:
            obs = traj["obs"]

            qpos = obs["agent"]["qpos"][t]
            qvel = obs["agent"]["qvel"][t]

            state = np.concatenate(
                [
                    qpos,
                    qvel,
                ],
                axis=-1,
            ).astype(np.float32)

            if self.state_mean is not None:
                state = (state - self.state_mean) / self.state_std

            sample["states"] = state

        return sample
    # =====================================================
    # Action statistics (OpenVLA norm_stats)
    # =====================================================
    def compute_action_stats(self, max_samples=None):
        actions = []

        total = len(self.index)
        if max_samples is not None:
            total = min(total, max_samples)

        for traj_key, t in tqdm(self.index[:total], desc="Computing action stats"):
            a = self.h5[traj_key]["actions"][t]
            actions.append(a)

        actions = np.asarray(actions, dtype=np.float32)

        stats = {
            "min": actions.min(axis=0),
            "max": actions.max(axis=0),
            "mean": actions.mean(axis=0),
            "std": actions.std(axis=0) + 1e-6,
            "q01": np.quantile(actions, 0.01, axis=0),
            "q99": np.quantile(actions, 0.99, axis=0),
        }
        return stats

    def export_data_stat(self, max_samples=None, force_recompute=False, robot="panda"):
        """
        Compute or load action norm stats for OpenVLA.

        Behavior:
        1. If norm_stats.json exists and force_recompute=False:
        - load and return it
        2. Else:
        - compute stats
        - save to json
        - return stats
        """

        out_path = os.path.join(
            os.path.dirname(self.dataset_file), "norm_stats.json"
        )

        # ---------- try load ----------
        if os.path.exists(out_path) and not force_recompute:
            print(f"[export_data_stat] Loading existing stats from {out_path}")
            stats = load_json(out_path)
            return stats

        # ---------- compute ----------
        print("[export_data_stat] Computing action stats...")
        action_stats = self.compute_action_stats(max_samples=max_samples)

        export = {
            self.task_name: {
                "action": {k: v.tolist() for k, v in action_stats.items()}
            }
        }

        if "panda" in robot:
            export[self.task_name]["action"]['mask'] = PANDA_MASK
        else:
            raise TypeError(f'目前不支持{robot}')

        # ---------- save ----------
        dump_json(out_path, export, indent=2)
        print(f"[export_data_stat] Saved to {out_path}")

        return export

    # =====================================================
    # State stats with cache
    # =====================================================
    def _load_or_compute_state_stats(self):
        cache_path = os.path.join(
            os.path.dirname(self.dataset_file), "state_stats.json"
        )
        if os.path.exists(cache_path):
            cache = load_json(cache_path)
            self.state_mean = np.asarray(cache["mean"])
            self.state_std = np.asarray(cache["std"])
            print(f"[state_stats] Loaded from {cache_path}")
        else:
            self._compute_state_stats()
            dump_json(
                cache_path,
                {
                    "mean": self.state_mean.tolist(),
                    "std": self.state_std.tolist(),
                },
                indent=2,
            )
            print(f"[state_stats] Saved to {cache_path}")

    def _compute_state_stats(self):
        states = []

        for traj_key, t in tqdm(self.index, desc="Computing state stats"):
            traj = self.h5[traj_key]
            obs = traj["obs"]

            qpos = obs["agent"]["qpos"][t]
            qvel = obs["agent"]["qvel"][t]
            extra = obs["extra"]

            state = np.concatenate(
                [
                    qpos,
                    qvel,
                    extra["tcp_pose"][t],
                    extra["obj_pose"][t],
                    extra["goal_pos"][t],
                ],
                axis=-1,
            )
            states.append(state)

        states = np.asarray(states, dtype=np.float32)
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-6
    
if __name__ == "__main__":
    m = ManiSkillDataset('datasets/PickCube-v1/motionplanning/trajectory.rgb+state_dict.pd_ee_delta_pose.physx_cpu.h5', task_name="PickCube-v1", normalize_states=True)
    print(m.export_data_stat())