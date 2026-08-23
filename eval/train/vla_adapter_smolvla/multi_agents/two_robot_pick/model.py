import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from PIL import Image
from transformers import AutoProcessor

from train.reinforcement_learning.utils import RunningMeanStd
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.tiny_vla import SharedTinyVLA4DActor

from prismatic.vla.action_tokenizer import ActionTokenizer


TASK_PROMPT = (
    "coordinate two robot arms to pick up the red cube and move it to the target goal position."
)
_DEBUG_RGB_SAVED = False
LLM_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
VISION_LORA_TARGET_MODULES = [
    "qkv",
    "proj",
    "fc1",
    "fc2",
    "fc3",
]


def build_agent_role_prompt(agent_name: str) -> str:
    if agent_name.endswith("-0"):
        role = "the left robot arm"
        partner = "the right robot arm"
    else:
        role = "the right robot arm"
        partner = "the left robot arm"
    return (
        f"act as {role}; coordinate with {partner} to pick up the red cube "
        "and move it to the target goal position."
    )


def _normalize_planner_subtask_sequence(
    value: Any,
    batch_size: int,
    *,
    field_name: str,
) -> List[Optional[str]]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, str):
        text = value.strip()
        return [text or None] * batch_size
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == 1 and batch_size != 1:
            return _normalize_planner_subtask_sequence(value[0], batch_size, field_name=field_name)
        if len(value) != batch_size:
            raise ValueError(f"{field_name} expects {batch_size} planner subtasks, got {len(value)}")
        normalized = []
        for item in value:
            if item is None:
                normalized.append(None)
            elif isinstance(item, str):
                text = item.strip()
                normalized.append(text or None)
            else:
                text = str(item).strip()
                normalized.append(text or None)
        return normalized
    raise TypeError(
        f"{field_name} must be None, str, list, tuple, or ndarray, got {type(value).__name__}"
    )


def extract_planner_subtasks_from_batch(
    batch: Mapping[str, Any],
    agent_names: Sequence[str],
    batch_size: int,
) -> Optional[Dict[str, List[Optional[str]]]]:
    planner_subtasks: Dict[str, List[Optional[str]]] = {}
    found = False

    shared_value = batch.get("planner_subtasks")
    if isinstance(shared_value, Mapping):
        for name in agent_names:
            if name in shared_value:
                planner_subtasks[name] = _normalize_planner_subtask_sequence(
                    shared_value[name],
                    batch_size,
                    field_name=f"planner_subtasks[{name}]",
                )
                found = True

    for name in agent_names:
        key = f"planner_subtask_{name}"
        if key in batch:
            planner_subtasks[name] = _normalize_planner_subtask_sequence(
                batch[key],
                batch_size,
                field_name=key,
            )
            found = True

    if not found:
        return None

    for name in agent_names:
        planner_subtasks.setdefault(name, [None] * batch_size)

    if not any(any(text is not None for text in texts) for texts in planner_subtasks.values()):
        return None
    return planner_subtasks


def resolve_attention_implementation(requested: str) -> str:
    valid = {"eager", "sdpa", "flash_attention_2"}
    if requested not in valid:
        raise ValueError(f"Unsupported attention implementation: {requested}. Expected one of {sorted(valid)}")
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("[setup] flash_attn is not installed; falling back to SDPA attention")
        return "sdpa"
    return requested


def ensure_package(package_name: str, package_dir: Path) -> None:
    if package_name in sys.modules:
        return
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _as_tensor(value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _ensure_2d(tensor: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
    if tensor.ndim == 0:
        if batch_size is None:
            return tensor.reshape(1, 1)
        return tensor.expand(batch_size).reshape(batch_size, 1)
    if tensor.ndim == 1:
        if batch_size is not None and tensor.shape[0] == batch_size:
            return tensor.unsqueeze(-1)
        return tensor.unsqueeze(0)
    return tensor


def maybe_save_debug_rgb(obs: Dict[str, Any], output_path: Union[str, Path] = "debug_rgb.png") -> None:
    global _DEBUG_RGB_SAVED
    if _DEBUG_RGB_SAVED:
        return
    sensor_data = obs.get("sensor_data", {})
    if "base_camera" not in sensor_data or "rgb" not in sensor_data["base_camera"]:
        return

    rgb = sensor_data["base_camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        image = rgb[0] if rgb.ndim == 4 else rgb
        image = image[..., :3].detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    else:
        image = np.asarray(rgb)
        if image.ndim == 4:
            image = image[0]
        image = image[..., :3].astype(np.uint8, copy=False)

    Image.fromarray(image).save(output_path)
    _DEBUG_RGB_SAVED = True


def extract_rgb_batch_from_obs(obs: Dict[str, Any]) -> Union[np.ndarray, torch.Tensor]:
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        return rgb[..., :3]
    return np.asarray(rgb)[..., :3]


def extract_agent_state_from_obs(obs: Dict[str, Any], agent_name: str) -> torch.Tensor:
    is_left = agent_name.endswith("-0")
    tcp_key = "left_arm_tcp" if is_left else "right_arm_tcp"
    tcp_to_cube_key = "left_arm_tcp_to_cube_pos" if is_left else "right_arm_tcp_to_cube_pos"

    qpos = _ensure_2d(_as_tensor(obs["agent"][agent_name]["qpos"], dtype=torch.float32))
    batch_size = qpos.shape[0]
    qvel = _ensure_2d(_as_tensor(obs["agent"][agent_name]["qvel"], dtype=torch.float32))
    tcp = _ensure_2d(_as_tensor(obs["extra"][tcp_key], dtype=torch.float32))
    tcp_to_cube = _ensure_2d(_as_tensor(obs["extra"][tcp_to_cube_key], dtype=torch.float32))
    cube_pose = _ensure_2d(_as_tensor(obs["extra"]["cube_pose"], dtype=torch.float32))
    cube_to_goal = _ensure_2d(_as_tensor(obs["extra"]["cube_to_goal_pos"], dtype=torch.float32))
    stage = _ensure_2d(_as_tensor(obs["extra"]["stage"], dtype=torch.float32), batch_size=batch_size)

    return torch.cat(
        [qpos, qvel, tcp, tcp_to_cube, cube_pose, cube_to_goal, stage],
        dim=-1,
    )


def extract_global_state_from_obs(obs: Dict[str, Any], agent_names: List[str]) -> torch.Tensor:
    left_name, right_name = agent_names
    left_qpos = _ensure_2d(_as_tensor(obs["agent"][left_name]["qpos"], dtype=torch.float32))
    batch_size = left_qpos.shape[0]
    left_qvel = _ensure_2d(_as_tensor(obs["agent"][left_name]["qvel"], dtype=torch.float32))
    right_qpos = _ensure_2d(_as_tensor(obs["agent"][right_name]["qpos"], dtype=torch.float32))
    right_qvel = _ensure_2d(_as_tensor(obs["agent"][right_name]["qvel"], dtype=torch.float32))
    left_tcp = _ensure_2d(_as_tensor(obs["extra"]["left_arm_tcp"], dtype=torch.float32))
    right_tcp = _ensure_2d(_as_tensor(obs["extra"]["right_arm_tcp"], dtype=torch.float32))
    cube_pose = _ensure_2d(_as_tensor(obs["extra"]["cube_pose"], dtype=torch.float32))
    left_tcp_to_cube = _ensure_2d(_as_tensor(obs["extra"]["left_arm_tcp_to_cube_pos"], dtype=torch.float32))
    right_tcp_to_cube = _ensure_2d(_as_tensor(obs["extra"]["right_arm_tcp_to_cube_pos"], dtype=torch.float32))
    cube_to_goal = _ensure_2d(_as_tensor(obs["extra"]["cube_to_goal_pos"], dtype=torch.float32))
    stage = _ensure_2d(_as_tensor(obs["extra"]["stage"], dtype=torch.float32), batch_size=batch_size)

    return torch.cat(
        [
            left_qpos,
            left_qvel,
            right_qpos,
            right_qvel,
            left_tcp,
            right_tcp,
            cube_pose,
            left_tcp_to_cube,
            right_tcp_to_cube,
            cube_to_goal,
            stage,
        ],
        dim=-1,
    )


def build_batch_from_obs(obs: Dict[str, Any], agent_names: List[str]) -> Dict[str, Any]:
    maybe_save_debug_rgb(obs)
    batch = {
        "rgb": extract_rgb_batch_from_obs(obs),
        "global_state": extract_global_state_from_obs(obs, agent_names),
    }
    for name in agent_names:
        batch[f"agent_states_{name}"] = extract_agent_state_from_obs(obs, name)
    return batch


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDiscreteActorHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, num_bins: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.logit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        action_features: torch.Tensor,
        state_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = action_features.shape
        action_features = self.context_encoder(
            action_features.reshape(batch_size * seq_len, 1, hidden_dim)
        ).reshape(batch_size, seq_len, hidden_dim)
        expanded_state = state_feature.unsqueeze(1).expand(-1, seq_len, -1)
        expanded_context = context_feature if context_feature.ndim == 3 else context_feature.unsqueeze(1)
        expanded_context = expanded_context.expand(-1, seq_len, -1)
        fused = torch.cat([action_features, expanded_state, expanded_context], dim=-1)
        return self.logit_head(fused) * self.residual_scale


class SharedVLA4DActor(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        state_dim: int,
        env_action_dim: int = 4,
        prompt: str = TASK_PROMPT,
        attention_implementation: str = "sdpa",
        image_size: Optional[int] = None,
        use_lora: bool = False,
        use_vision_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        train_vision_backbone: bool = False,
        vision_token_pool_size: Optional[int] = None,
        policy_mode: str = "residual",
    ):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.state_dim = state_dim
        self.env_action_dim = env_action_dim
        self.use_lora = use_lora
        self.use_vision_lora = use_vision_lora
        self.image_size = image_size
        self.train_vision_backbone = train_vision_backbone
        self.vision_token_pool_size = vision_token_pool_size
        self.policy_mode = policy_mode
        self.prompt_task_text = str(prompt).strip()
        self.left_role_task_text = build_agent_role_prompt("agent-0")
        self.right_role_task_text = build_agent_role_prompt("agent-1")
        self.prompt = f"In: What action should the robot take to {self.prompt_task_text}\nOut: "
        self.left_role_prompt = f"In: What action should the robot take to {self.left_role_task_text}\nOut: "
        self.right_role_prompt = f"In: What action should the robot take to {self.right_role_task_text}\nOut: "
        self.max_planner_prompt_tokens = 128

        self.processor = AutoProcessor.from_pretrained(str(self.model_dir), trust_remote_code=True)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and self.image_size is not None:
            size_cfg = getattr(image_processor, "size", None)
            if isinstance(size_cfg, dict):
                if "height" in size_cfg:
                    size_cfg["height"] = int(self.image_size)
                if "width" in size_cfg:
                    size_cfg["width"] = int(self.image_size)
                if "shortest_edge" in size_cfg:
                    size_cfg["shortest_edge"] = int(self.image_size)
            elif isinstance(size_cfg, int):
                image_processor.size = int(self.image_size)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

        prompt_tokens = self.processor.tokenizer(self.prompt, return_tensors="pt")
        self.register_buffer("prompt_input_ids", prompt_tokens["input_ids"], persistent=False)
        self.register_buffer("prompt_attention_mask", prompt_tokens["attention_mask"], persistent=False)
        role_prompt_tokens = self.processor.tokenizer(
            [self.left_role_prompt, self.right_role_prompt],
            return_tensors="pt",
            padding=True,
        )
        self.register_buffer("role_prompt_input_ids", role_prompt_tokens["input_ids"], persistent=False)
        self.register_buffer("role_prompt_attention_mask", role_prompt_tokens["attention_mask"], persistent=False)

        ensure_package("local_multi_vla_pkg", self.model_dir)
        config_mod = load_module_from_path(
            "local_multi_vla_pkg.configuration_prismatic",
            self.model_dir / "configuration_prismatic.py",
        )
        model_mod = load_module_from_path(
            "local_multi_vla_pkg.modeling_prismatic",
            self.model_dir / "modeling_prismatic.py",
        )

        self.vla = model_mod.OpenVLAForActionPrediction.from_pretrained(
            str(self.model_dir),
            config=config_mod.OpenVLAConfig.from_pretrained(str(self.model_dir)),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=resolve_attention_implementation(attention_implementation),
        )
        self.vla.set_version("v1")
        if self.use_lora:
            self.vla = self._apply_lora(
                self.vla,
                target_modules=self._get_lora_target_modules(),
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        self.full_vocab_size = int(self.vla.vocab_size)
        self.action_token_start_idx = int(self.action_tokenizer.action_token_begin_idx + 1)
        self.action_token_end_idx = int(self.action_tokenizer.action_token_end_idx)
        self.num_action_bins = int(self.action_tokenizer.vocab_size)
        self.hidden_dim = int(self.vla.llm_dim)
        self.register_buffer(
            "action_bin_centers",
            torch.from_numpy(self.action_tokenizer.bin_centers.astype(np.float32)),
            persistent=False,
        )

        self.state_projector = MLPProjector(
            input_dim=state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        ).to(dtype=torch.float32)
        self.context_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(dtype=torch.float32)
        self.actor_head = ResidualDiscreteActorHead(
            hidden_dim=self.hidden_dim,
            action_dim=env_action_dim,
            num_bins=self.num_action_bins,
        ).to(dtype=torch.float32)
        if self.policy_mode not in {"residual", "native"}:
            raise ValueError(f"Unsupported policy_mode={self.policy_mode}")

        self.eval_micro_batch_size = 32
        self._vla_trainable = True

    @property
    def device(self) -> torch.device:
        return next(self.vla.parameters()).device

    @staticmethod
    def _apply_lora(
        vla: nn.Module,
        *,
        target_modules: List[str],
        r: int,
        alpha: int,
        dropout: float,
    ) -> nn.Module:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("`peft` is required to enable VLA LoRA") from exc

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_modules,
        )
        wrapped = get_peft_model(vla, lora_cfg)
        wrapped.print_trainable_parameters()
        return wrapped

    def _get_lora_target_modules(self) -> List[str]:
        targets = list(LLM_LORA_TARGET_MODULES)
        if self.use_vision_lora:
            targets.extend(VISION_LORA_TARGET_MODULES)
        return targets

    @staticmethod
    def _is_vision_parameter_name(name: str) -> bool:
        return "vision_backbone" in name

    @staticmethod
    def _is_projector_parameter_name(name: str) -> bool:
        return ".projector." in name or name.startswith("projector.")

    def configure_trainable_modules(self, train_backbone: bool) -> None:
        self._vla_trainable = train_backbone
        if self.use_lora:
            for parameter in self.vla.parameters():
                parameter.requires_grad = False
            for name, parameter in self.vla.named_parameters():
                if "lora_" in name:
                    if not self.train_vision_backbone and self._is_vision_parameter_name(name):
                        parameter.requires_grad = False
                        continue
                    # `freeze_vla_backbone` should freeze the base VLA weights while keeping LoRA trainable.
                    parameter.requires_grad = True
                elif self._is_projector_parameter_name(name):
                    parameter.requires_grad = train_backbone
        else:
            for name, parameter in self.vla.named_parameters():
                if not self.train_vision_backbone and self._is_vision_parameter_name(name):
                    parameter.requires_grad = False
                else:
                    parameter.requires_grad = train_backbone
        actor_modules = [self.state_projector]
        if self.policy_mode == "residual":
            actor_modules.extend([self.context_projector, self.actor_head])
        for module in actor_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _vla_has_trainable_params(self) -> bool:
        return any(parameter.requires_grad for parameter in self.vla.parameters())

    @staticmethod
    def _prepare_image(rgb: Union[np.ndarray, torch.Tensor]) -> Image.Image:
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
        return Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGB")

    def _prepare_policy_inputs(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        prompt_role_ids: Optional[torch.Tensor] = None,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        pixel_values = self._prepare_pixel_values(rgbs)
        batch_size = pixel_values.shape[0]
        input_ids, attention_mask = self._resolve_prompt_inputs(
            batch_size,
            prompt_role_ids,
            planner_subtasks=planner_subtasks,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

    def _prepare_pixel_values(self, rgbs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(rgbs, torch.Tensor):
            rgb_batch = rgbs[..., :3].detach()
            if rgb_batch.device.type != "cpu" or rgb_batch.dtype != torch.uint8 or not rgb_batch.is_contiguous():
                rgb_batch = rgb_batch.to(device="cpu", dtype=torch.uint8).contiguous()
        else:
            rgb_batch = torch.from_numpy(np.asarray(rgbs)[..., :3].astype(np.uint8, copy=False)).contiguous()
        images = [self._prepare_image(rgb) for rgb in rgb_batch]
        pixel_values = self.processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
        return pixel_values.to(self.device, dtype=torch.bfloat16, non_blocking=True)

    def _resolve_prompt_inputs(
        self,
        batch_size: int,
        prompt_role_ids: Optional[torch.Tensor] = None,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if planner_subtasks is not None:
            if len(planner_subtasks) != batch_size:
                raise ValueError(f"planner_subtasks expects {batch_size} items, got {len(planner_subtasks)}")
            if prompt_role_ids is None:
                role_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
            else:
                role_ids = prompt_role_ids.to(self.device, dtype=torch.long, non_blocking=True)

            prompt_texts = []
            for idx, subtask in enumerate(planner_subtasks):
                subtask_text = None if subtask is None else str(subtask).strip()
                role_id = int(role_ids[idx].item())
                if role_id == 0:
                    base_text = self.left_role_task_text
                elif role_id == 1:
                    base_text = self.right_role_task_text
                else:
                    base_text = self.prompt_task_text
                if subtask_text:
                    prompt_texts.append(
                        f"In: What action should the robot take to {base_text}. "
                        f"Current assigned subtask: {subtask_text}\nOut: "
                    )
                else:
                    prompt_texts.append(f"In: What action should the robot take to {base_text}\nOut: ")

            prompt_tokens = self.processor.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_planner_prompt_tokens,
            )
            return (
                prompt_tokens["input_ids"].to(self.device, non_blocking=True),
                prompt_tokens["attention_mask"].to(self.device, non_blocking=True),
            )

        if prompt_role_ids is None:
            input_ids = self.prompt_input_ids.expand(batch_size, -1)
            attention_mask = self.prompt_attention_mask.expand(batch_size, -1)
        else:
            prompt_role_ids = prompt_role_ids.to(self.device, dtype=torch.long, non_blocking=True)
            input_ids = self.role_prompt_input_ids[prompt_role_ids]
            attention_mask = self.role_prompt_attention_mask[prompt_role_ids]
        return input_ids.to(self.device, non_blocking=True), attention_mask.to(self.device, non_blocking=True)

    def _action_bins_to_token_ids(self, action_bins: torch.Tensor) -> torch.Tensor:
        action_bins = action_bins.to(self.device, dtype=torch.long)
        action_bins = torch.clamp(action_bins, 0, self.num_action_bins - 1)
        return self.action_token_end_idx - action_bins - 1

    def env_actions_to_bin_indices(self, env_actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(env_actions, torch.Tensor):
            env_actions = env_actions.detach().cpu().numpy()
        env_actions = np.asarray(env_actions, dtype=np.float32)
        if env_actions.ndim == 1:
            env_actions = env_actions[None, :]
        env_actions = np.clip(env_actions, -1.0, 1.0)
        token_ids = np.asarray(self.action_tokenizer(env_actions, use_minivlm=True), dtype=np.int64)
        bin_indices = self.action_token_end_idx - token_ids - 1
        bin_indices = np.clip(bin_indices, 0, self.action_bin_centers.numel() - 1)
        return torch.from_numpy(bin_indices.astype(np.int64))

    def bin_indices_to_env_actions(self, bin_indices: torch.Tensor) -> torch.Tensor:
        if bin_indices.ndim == 1:
            bin_indices = bin_indices.unsqueeze(0)
        bin_indices = bin_indices.to(self.device, dtype=torch.long)
        bin_indices = torch.clamp(bin_indices, 0, self.action_bin_centers.numel() - 1)
        env_actions = self.action_bin_centers.to(self.device)[bin_indices].to(torch.float32)
        return torch.nan_to_num(env_actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

    def _project_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._vla_has_trainable_params():
            projected = self.vla._process_vision_features(pixel_values, language_embeddings=None, use_film=False)
        else:
            with torch.no_grad():
                projected = self.vla._process_vision_features(pixel_values, language_embeddings=None, use_film=False)
        return self._pool_vision_tokens(projected)

    def _pool_vision_tokens(self, projected_patch_embeddings: torch.Tensor) -> torch.Tensor:
        target_tokens = self.vision_token_pool_size
        if target_tokens is None:
            return projected_patch_embeddings
        if target_tokens <= 0:
            raise ValueError(f"vision_token_pool_size must be positive, got {target_tokens}")

        batch_size, num_tokens, hidden_dim = projected_patch_embeddings.shape
        if target_tokens >= num_tokens:
            return projected_patch_embeddings

        # Downsample the visual prefix before feeding it into the LM to reduce attention memory.
        pooled = F.adaptive_avg_pool1d(
            projected_patch_embeddings.transpose(1, 2),
            output_size=target_tokens,
        )
        return pooled.transpose(1, 2).reshape(batch_size, target_tokens, hidden_dim)

    @staticmethod
    def _append_state_token_to_patches(
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> torch.Tensor:
        state_token = state_feature.to(dtype=projected_patch_embeddings.dtype).unsqueeze(1)
        return torch.cat([projected_patch_embeddings, state_token], dim=1)

    def _language_model_from_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        *,
        use_cache: bool,
    ):
        def run_language_model():
            input_embeddings = self.vla.get_input_embeddings()(input_ids)
            multimodal_embeddings, multimodal_attention_mask = self.vla._build_multimodal_attention(
                input_embeddings,
                projected_patch_embeddings,
                attention_mask,
            )
            base_output = self.vla.language_model.model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                use_cache=use_cache,
                output_attentions=False,
                return_dict=True,
            )
            last_hidden_state = base_output.last_hidden_state
            logits = self.vla.language_model.lm_head(last_hidden_state)
            return SimpleNamespace(
                logits=logits,
                last_hidden_state=last_hidden_state,
                past_key_values=base_output.past_key_values,
            )

        if self._vla_has_trainable_params():
            return run_language_model()
        with torch.no_grad():
            return run_language_model()

    def _language_model_next_token_from_cache(self, token_id: torch.Tensor, past_key_values):
        def run_language_model():
            base_output = self.vla.language_model.model(
                input_ids=token_id,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            last_hidden_state = base_output.last_hidden_state
            logits = self.vla.language_model.lm_head(last_hidden_state)
            return SimpleNamespace(
                logits=logits,
                last_hidden_state=last_hidden_state,
                past_key_values=base_output.past_key_values,
            )

        if self._vla_has_trainable_params():
            return run_language_model()
        with torch.no_grad():
            return run_language_model()

    def _compute_prompt_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self._language_model_from_prefix(
            input_ids,
            attention_mask,
            projected_patch_embeddings,
            use_cache=False,
        )
        final_hidden = output.last_hidden_state.to(torch.float32)
        prompt_hidden = final_hidden[:, -1, :]
        context_feature = self.context_projector(final_hidden.mean(dim=1))
        action_token_logits = output.logits[:, -1, self.action_token_start_idx : self.action_token_end_idx].to(
            torch.float32
        )
        base_bin_logits = torch.flip(action_token_logits, dims=[-1])
        if self.policy_mode == "residual":
            residual_logits = self.actor_head(prompt_hidden.unsqueeze(1), state_feature, context_feature).squeeze(1)
            logits = base_bin_logits + residual_logits
        else:
            logits = base_bin_logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return logits, prompt_hidden, context_feature

    def _collect_action_position_contexts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
        selected_bins: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_bins = selected_bins.to(self.device, dtype=torch.long)
        if selected_bins.ndim == 1:
            selected_bins = selected_bins.unsqueeze(1)

        prompt_hidden_per_pos: List[torch.Tensor] = []
        context_feature_per_pos: List[torch.Tensor] = []

        output = self._language_model_from_prefix(
            input_ids,
            attention_mask,
            projected_patch_embeddings,
            use_cache=True,
        )
        final_hidden = output.last_hidden_state.to(torch.float32)
        hidden_sum = final_hidden.sum(dim=1)
        hidden_count = final_hidden.shape[1]

        for action_idx in range(selected_bins.shape[1]):
            prompt_hidden = final_hidden[:, -1, :]
            context_feature = self.context_projector(hidden_sum / hidden_count)
            prompt_hidden_per_pos.append(prompt_hidden)
            context_feature_per_pos.append(context_feature)

            if action_idx + 1 < selected_bins.shape[1]:
                next_token_id = self._action_bins_to_token_ids(selected_bins[:, action_idx]).unsqueeze(1)
                output = self._language_model_next_token_from_cache(next_token_id, output.past_key_values)
                final_hidden = output.last_hidden_state.to(torch.float32)
                last_hidden = final_hidden[:, -1, :]
                hidden_sum = hidden_sum + last_hidden
                hidden_count += 1

        return (
            torch.stack(prompt_hidden_per_pos, dim=1),
            torch.stack(context_feature_per_pos, dim=1),
        )

    def _evaluate_action_bins_native_full_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        action_bins: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        action_bins = action_bins.to(self.device, dtype=torch.long)
        if action_bins.ndim == 1:
            action_bins = action_bins.unsqueeze(1)

        action_token_ids = self._action_bins_to_token_ids(action_bins)
        full_input_ids = torch.cat([input_ids, action_token_ids], dim=1)
        full_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(action_token_ids, dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )
        output = self._language_model_from_prefix(
            full_input_ids,
            full_attention_mask,
            projected_patch_embeddings,
            use_cache=False,
        )

        action_len = action_bins.shape[1]
        logits_tensor = output.logits[:, -action_len - 1 : -1, self.action_token_start_idx : self.action_token_end_idx]
        logits_tensor = torch.flip(logits_tensor.to(torch.float32), dims=[-1])
        logits_tensor = torch.nan_to_num(logits_tensor, nan=0.0, posinf=20.0, neginf=-20.0)
        logprobs_tensor = F.log_softmax(logits_tensor, dim=-1)
        gathered = torch.gather(logprobs_tensor, 2, action_bins.unsqueeze(-1)).squeeze(-1)
        probs_tensor = logprobs_tensor.exp()
        entropy = -(probs_tensor * logprobs_tensor).sum(dim=-1).mean(dim=-1)
        return {
            "log_prob": gathered.sum(dim=-1),
            "entropy": entropy,
            "token_logits": logits_tensor,
        }

    def get_action_and_stats(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        states: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        action_bins: Optional[torch.Tensor] = None,
        prompt_role_ids: Optional[torch.Tensor] = None,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = self._prepare_policy_inputs(
            rgbs,
            prompt_role_ids=prompt_role_ids,
            planner_subtasks=planner_subtasks,
        )
        projected_patch_embeddings = self._project_vision_features(model_inputs["pixel_values"])
        return self._get_action_and_stats_from_prepared_inputs(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            projected_patch_embeddings=projected_patch_embeddings,
            states=states,
            state_features=state_features,
            action_bins=action_bins,
            deterministic=deterministic,
        )

    def _get_action_and_stats_from_prepared_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        states: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if state_features is None:
            state_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
            state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
            state_feature = self.state_projector(state_tensor)
        else:
            state_feature = torch.as_tensor(state_features, device=self.device, dtype=torch.float32)
            state_feature = torch.nan_to_num(state_feature, nan=0.0, posinf=1e4, neginf=-1e4)
        projected_patch_embeddings = self._append_state_token_to_patches(projected_patch_embeddings, state_feature)

        if action_bins is not None:
            action_bins = action_bins.to(self.device, dtype=torch.long)
            if action_bins.ndim == 1:
                action_bins = action_bins.unsqueeze(1)

            if self.policy_mode == "native":
                _, prompt_hidden_out, context_feature_out = self._compute_prompt_features(
                    input_ids,
                    attention_mask,
                    projected_patch_embeddings,
                    state_feature,
                )
                action_position_prompt_hidden, action_position_context_feature = self._collect_action_position_contexts(
                    input_ids,
                    attention_mask,
                    projected_patch_embeddings,
                    state_feature,
                    action_bins,
                )
                native_stats = self._evaluate_action_bins_native_full_sequence(
                    input_ids,
                    attention_mask,
                    projected_patch_embeddings,
                    action_bins,
                )
                env_actions = self.bin_indices_to_env_actions(action_bins)
                return {
                    "env_actions": env_actions,
                    "log_prob": native_stats["log_prob"],
                    "entropy": native_stats["entropy"],
                    "token_logits": native_stats["token_logits"],
                    "action_bins": action_bins,
                    "prompt_hidden": prompt_hidden_out,
                    "context_feature": context_feature_out,
                    "state_feature": state_feature,
                    "action_position_prompt_hidden": action_position_prompt_hidden,
                    "action_position_context_feature": action_position_context_feature,
                }

            scored_log_probs: List[torch.Tensor] = []
            scored_entropies: List[torch.Tensor] = []
            scored_logits: List[torch.Tensor] = []
            prompt_hidden_out: Optional[torch.Tensor] = None
            context_feature_out: Optional[torch.Tensor] = None
            action_position_prompt_hidden: List[torch.Tensor] = []
            action_position_context_feature: List[torch.Tensor] = []

            output = self._language_model_from_prefix(
                input_ids,
                attention_mask,
                projected_patch_embeddings,
                use_cache=True,
            )
            final_hidden = output.last_hidden_state.to(torch.float32)
            hidden_sum = final_hidden.sum(dim=1)
            hidden_count = final_hidden.shape[1]

            for action_idx in range(self.env_action_dim):
                prompt_hidden = final_hidden[:, -1, :]
                context_feature = self.context_projector(hidden_sum / hidden_count)
                action_token_logits = output.logits[:, -1, self.action_token_start_idx : self.action_token_end_idx].to(
                    torch.float32
                )
                base_bin_logits = torch.flip(action_token_logits, dims=[-1])
                if self.policy_mode == "residual":
                    residual_logits = self.actor_head(
                        prompt_hidden.unsqueeze(1), state_feature, context_feature
                    ).squeeze(1)
                    logits = base_bin_logits + residual_logits
                else:
                    logits = base_bin_logits
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

                if action_idx == 0:
                    prompt_hidden_out = prompt_hidden
                    context_feature_out = context_feature

                action_position_prompt_hidden.append(prompt_hidden)
                action_position_context_feature.append(context_feature)
                categorical = torch.distributions.Categorical(logits=logits)
                selected_bin = action_bins[:, action_idx]
                scored_log_probs.append(categorical.log_prob(selected_bin))
                scored_entropies.append(categorical.entropy())
                scored_logits.append(logits)

                if action_idx + 1 < self.env_action_dim:
                    next_token_id = self._action_bins_to_token_ids(selected_bin).unsqueeze(1)
                    output = self._language_model_next_token_from_cache(next_token_id, output.past_key_values)
                    final_hidden = output.last_hidden_state.to(torch.float32)
                    last_hidden = final_hidden[:, -1, :]
                    hidden_sum = hidden_sum + last_hidden
                    hidden_count += 1

            env_actions = self.bin_indices_to_env_actions(action_bins)
            return {
                "env_actions": env_actions,
                "log_prob": torch.stack(scored_log_probs, dim=1).sum(dim=-1),
                "entropy": torch.stack(scored_entropies, dim=1).mean(dim=-1),
                "token_logits": torch.stack(scored_logits, dim=1),
                "action_bins": action_bins,
                "prompt_hidden": prompt_hidden_out,
                "context_feature": context_feature_out,
                "state_feature": state_feature,
                "action_position_prompt_hidden": torch.stack(action_position_prompt_hidden, dim=1),
                "action_position_context_feature": torch.stack(action_position_context_feature, dim=1),
            }

        generated_bins: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        generated_logits: List[torch.Tensor] = []
        prompt_hidden_out: Optional[torch.Tensor] = None
        context_feature_out: Optional[torch.Tensor] = None
        action_position_prompt_hidden: List[torch.Tensor] = []
        action_position_context_feature: List[torch.Tensor] = []

        output = self._language_model_from_prefix(
            input_ids,
            attention_mask,
            projected_patch_embeddings,
            use_cache=True,
        )
        final_hidden = output.last_hidden_state.to(torch.float32)
        hidden_sum = final_hidden.sum(dim=1)
        hidden_count = final_hidden.shape[1]

        for action_idx in range(self.env_action_dim):
            prompt_hidden = final_hidden[:, -1, :]
            context_feature = self.context_projector(hidden_sum / hidden_count)
            action_token_logits = output.logits[:, -1, self.action_token_start_idx : self.action_token_end_idx].to(
                torch.float32
            )
            base_bin_logits = torch.flip(action_token_logits, dims=[-1])
            if self.policy_mode == "residual":
                residual_logits = self.actor_head(prompt_hidden.unsqueeze(1), state_feature, context_feature).squeeze(1)
                logits = base_bin_logits + residual_logits
            else:
                logits = base_bin_logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

            if action_idx == 0:
                prompt_hidden_out = prompt_hidden
                context_feature_out = context_feature

            action_position_prompt_hidden.append(prompt_hidden)
            action_position_context_feature.append(context_feature)
            categorical = torch.distributions.Categorical(logits=logits)
            selected_bin = logits.argmax(dim=-1) if deterministic else categorical.sample()
            generated_bins.append(selected_bin)
            log_probs.append(categorical.log_prob(selected_bin))
            entropies.append(categorical.entropy())
            generated_logits.append(logits)

            if action_idx + 1 < self.env_action_dim:
                next_token_id = self._action_bins_to_token_ids(selected_bin).unsqueeze(1)
                output = self._language_model_next_token_from_cache(next_token_id, output.past_key_values)
                final_hidden = output.last_hidden_state.to(torch.float32)
                last_hidden = final_hidden[:, -1, :]
                hidden_sum = hidden_sum + last_hidden
                hidden_count += 1

        selected_bins = torch.stack(generated_bins, dim=1)
        env_actions = self.bin_indices_to_env_actions(selected_bins)

        if self.policy_mode == "native":
            native_stats = self._evaluate_action_bins_native_full_sequence(
                input_ids,
                attention_mask,
                projected_patch_embeddings,
                selected_bins,
            )
            return {
                "env_actions": env_actions,
                "log_prob": native_stats["log_prob"],
                "entropy": native_stats["entropy"],
                "token_logits": native_stats["token_logits"],
                "action_bins": selected_bins,
                "prompt_hidden": prompt_hidden_out,
                "context_feature": context_feature_out,
                "state_feature": state_feature,
                "action_position_prompt_hidden": torch.stack(action_position_prompt_hidden, dim=1),
                "action_position_context_feature": torch.stack(action_position_context_feature, dim=1),
            }

        return {
            "env_actions": env_actions,
            "log_prob": torch.stack(log_probs, dim=1).sum(dim=-1),
            "entropy": torch.stack(entropies, dim=1).mean(dim=-1),
            "token_logits": torch.stack(generated_logits, dim=1),
            "action_bins": selected_bins,
            "prompt_hidden": prompt_hidden_out,
            "context_feature": context_feature_out,
            "state_feature": state_feature,
            "action_position_prompt_hidden": torch.stack(action_position_prompt_hidden, dim=1),
            "action_position_context_feature": torch.stack(action_position_context_feature, dim=1),
        }


class MultiAgentVLAAdapterMAPPOAgent(nn.Module):
    def __init__(
        self,
        agent_names: List[str],
        state_dim: int,
        global_state_dim: int,
        action_dim: int,
        model_dir: Optional[Union[str, Path]],
        normalize_state: bool = True,
        freeze_vla_backbone: bool = False,
        critic_hidden_dim: int = 512,
        attention_implementation: str = "sdpa",
        image_size: Optional[int] = None,
        use_vla_lora: bool = False,
        use_vision_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        train_vision_backbone: bool = False,
        vision_token_pool_size: Optional[int] = None,
        policy_mode: str = "residual",
        model_backbone: str = "openvla",
        tiny_hidden_dim: int = 640,
        tiny_vision_layers: int = 7,
        tiny_decoder_layers: int = 8,
        tiny_attention_heads: int = 10,
        tiny_patch_size: int = 14,
        tiny_ffn_mult: int = 4,
        tiny_num_action_bins: int = 256,
        tiny_prompt_length: int = 24,
        enable_planner_subtasks: bool = False,
    ):
        super().__init__()
        self.agent_names = list(agent_names)
        self.num_agents = len(self.agent_names)
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.model_backbone = str(model_backbone)
        self.enable_planner_subtasks = bool(enable_planner_subtasks)
        if self.model_backbone == "openvla":
            if model_dir is None:
                raise ValueError("model_dir is required when model_backbone=openvla")
            self.actor = SharedVLA4DActor(
                model_dir=Path(model_dir),
                state_dim=state_dim,
                env_action_dim=action_dim,
                attention_implementation=attention_implementation,
                image_size=image_size,
                use_lora=use_vla_lora,
                use_vision_lora=use_vision_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                train_vision_backbone=train_vision_backbone,
                vision_token_pool_size=vision_token_pool_size,
                policy_mode=policy_mode,
            )
        elif self.model_backbone == "tiny":
            self.actor = SharedTinyVLA4DActor(
                model_dir=model_dir,
                state_dim=state_dim,
                env_action_dim=action_dim,
                attention_implementation=attention_implementation,
                image_size=image_size,
                use_lora=use_vla_lora,
                use_vision_lora=use_vision_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                train_vision_backbone=train_vision_backbone,
                vision_token_pool_size=vision_token_pool_size,
                policy_mode=policy_mode,
                tiny_hidden_dim=tiny_hidden_dim,
                tiny_vision_layers=tiny_vision_layers,
                tiny_decoder_layers=tiny_decoder_layers,
                tiny_attention_heads=tiny_attention_heads,
                tiny_patch_size=tiny_patch_size,
                tiny_ffn_mult=tiny_ffn_mult,
                tiny_num_action_bins=tiny_num_action_bins,
                tiny_prompt_length=tiny_prompt_length,
            )
        else:
            raise ValueError(f"Unsupported model_backbone={self.model_backbone}")
        self.actor.configure_trainable_modules(not freeze_vla_backbone)

        hidden_dim = self.actor.hidden_dim
        self.actor_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Identity(),
                )
                for name in self.agent_names
            }
        )
        self.actor_feature_placeholders = nn.ModuleDict({name: nn.Identity() for name in self.agent_names})
        self.actor_action_position_placeholders = nn.ModuleDict(
            {
                name: nn.ModuleList([nn.Identity() for _ in range(action_dim)])
                for name in self.agent_names
            }
        )
        self.actor_action_position_actor_placeholders = nn.ModuleDict(
            {
                name: nn.ModuleList([nn.Identity() for _ in range(action_dim)])
                for name in self.agent_names
            }
        )

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(state_dim,)) for name in self.agent_names}
            )
            self.critic_state_rms = RunningMeanStd(shape=(global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        self.critic_state_encoder = nn.Sequential(
            nn.LayerNorm(global_state_dim),
            nn.Linear(global_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(dtype=torch.float32)
        self.critic_visual_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(dtype=torch.float32)
        critic_input_dim = hidden_dim * 2
        self.critic = nn.Sequential(
            nn.LayerNorm(critic_input_dim),
            nn.Linear(critic_input_dim, critic_hidden_dim),
            nn.GELU(),
            nn.Linear(critic_hidden_dim, 1),
        ).to(dtype=torch.float32)

    def configure_trainable_modules(self, freeze_vla_backbone: bool) -> None:
        self.actor.configure_trainable_modules(not freeze_vla_backbone)
        for module in [self.critic_state_encoder, self.critic_visual_encoder, self.critic]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def trainable_parameter_summary(self) -> Dict[str, Tuple[int, int]]:
        modules = {
            "vla": self.actor.vla,
            "state_projector": self.actor.state_projector,
            "context_projector": self.actor.context_projector,
            "actor_head": self.actor.actor_head,
            "critic_state_encoder": self.critic_state_encoder,
            "critic_visual_encoder": self.critic_visual_encoder,
            "critic": self.critic,
        }
        summary = {}
        for name, module in modules.items():
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            summary[name] = (total, trainable)
        return summary

    def _normalize_actor_state(self, state: torch.Tensor, agent_name: str) -> torch.Tensor:
        if self.actor_state_rms is None:
            return state
        return self.actor_state_rms[agent_name](state)

    def _normalize_global_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.critic_state_rms is None:
            return state
        return self.critic_state_rms(state)

    def _repeat_rgb_for_agents(self, rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(rgb, torch.Tensor):
            return torch.cat([rgb] * self.num_agents, dim=0)
        return np.concatenate([rgb] * self.num_agents, axis=0)

    def _build_actor_input_dict(
        self,
        state_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = state_features.shape[0] // self.num_agents
        actor_inputs = {}
        for idx, name in enumerate(self.agent_names):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            local_feature = state_features[start:end]
            local_feature = self.actor_feature_placeholders[name](local_feature)
            actor_inputs[name] = self.actor_heads[name](local_feature)
        return actor_inputs

    def _apply_action_position_placeholders(
        self,
        actor_out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        action_position_prompt_hidden = actor_out.get("action_position_prompt_hidden")
        action_position_context_feature = actor_out.get("action_position_context_feature")
        state_feature = actor_out.get("state_feature")
        if (
            action_position_prompt_hidden is None
            or action_position_context_feature is None
            or state_feature is None
        ):
            return actor_out

        if action_position_prompt_hidden.ndim != 3 or action_position_context_feature.ndim != 3:
            return actor_out

        batch_size_per_agent = action_position_prompt_hidden.shape[0] // self.num_agents
        position_features = []
        for idx, name in enumerate(self.agent_names):
            start = idx * batch_size_per_agent
            end = (idx + 1) * batch_size_per_agent
            local_prompt_hidden = action_position_prompt_hidden[start:end]
            local_context_feature = action_position_context_feature[start:end]
            local_state_feature = state_feature[start:end]
            agent_position_features = []
            for action_idx in range(local_prompt_hidden.shape[1]):
                position_feature = torch.cat(
                    [
                        local_prompt_hidden[:, action_idx, :],
                        local_context_feature[:, action_idx, :],
                        local_state_feature,
                    ],
                    dim=-1,
                )
                position_feature = self.actor_action_position_placeholders[name][action_idx](position_feature)
                position_feature = self.actor_action_position_actor_placeholders[name][action_idx](position_feature)
                agent_position_features.append(position_feature)
            position_features.append(torch.stack(agent_position_features, dim=1))
        actor_out["action_position_features"] = torch.cat(position_features, dim=0)
        return actor_out

    def _project_actor_state_features(
        self,
        normalized_states: List[torch.Tensor],
    ) -> torch.Tensor:
        cat_states = torch.cat(normalized_states, dim=0)
        state_tensor = torch.as_tensor(cat_states, device=self.actor.device, dtype=torch.float32)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        return self.actor.state_projector(state_tensor)

    def _run_shared_actor(
        self,
        batch: Dict[str, Any],
        actions_input: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        action_bins_input: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        deterministic: bool = False,
        return_token_logits: bool = False,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        Optional[Dict[str, torch.Tensor]],
    ]:
        rgb = batch["rgb"]
        batch_size = batch[f"agent_states_{self.agent_names[0]}"].shape[0]
        planner_subtasks = None
        if self.enable_planner_subtasks:
            planner_subtasks = extract_planner_subtasks_from_batch(batch, self.agent_names, batch_size)

        normalized_states = []
        prompt_role_ids = []
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"]
            state = state.to(device=self.actor.device, dtype=torch.float32)
            normalized_states.append(self._normalize_actor_state(state, name))
            role_id = 0 if name.endswith("-0") else 1
            prompt_role_ids.append(
                torch.full((state.shape[0],), role_id, dtype=torch.long, device=self.actor.device)
            )
        cat_states = torch.cat(normalized_states, dim=0)
        prompt_role_ids = torch.cat(prompt_role_ids, dim=0)
        state_features = self._project_actor_state_features(normalized_states)
        actor_inputs = self._build_actor_input_dict(state_features)
        cat_actor_inputs = torch.cat([actor_inputs[name] for name in self.agent_names], dim=0)

        action_bins = None
        if action_bins_input is not None:
            action_bins = torch.cat(
                [
                    torch.as_tensor(action_bins_input[name], device=self.actor.device, dtype=torch.long)
                    for name in self.agent_names
                ],
                dim=0,
            )
        elif actions_input is not None:
            action_bins = torch.cat(
                [self.actor.env_actions_to_bin_indices(actions_input[name]) for name in self.agent_names],
                dim=0,
            )

        projected_patch_embeddings = self.actor._project_vision_features(self.actor._prepare_pixel_values(rgb))
        critic_visual_feature = self.critic_visual_encoder(projected_patch_embeddings.mean(dim=1).to(torch.float32).detach())
        projected_patch_embeddings = torch.cat([projected_patch_embeddings] * self.num_agents, dim=0)
        cat_planner_subtasks = None
        if planner_subtasks is not None:
            cat_planner_subtasks = []
            for name in self.agent_names:
                cat_planner_subtasks.extend(planner_subtasks[name])
        input_ids, attention_mask = self.actor._resolve_prompt_inputs(
            cat_states.shape[0],
            prompt_role_ids,
            planner_subtasks=cat_planner_subtasks,
        )

        raw_actor_out = self.actor._get_action_and_stats_from_prepared_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            projected_patch_embeddings=projected_patch_embeddings,
            states=cat_states,
            state_features=cat_actor_inputs,
            action_bins=action_bins,
            deterministic=deterministic,
        )
        actor_out = dict(raw_actor_out)
        actor_out["state_feature"] = cat_actor_inputs
        actor_out = self._apply_action_position_placeholders(actor_out)

        batch_size = normalized_states[0].shape[0]
        actions_out = {}
        log_probs = {}
        entropies = {}
        action_bins_out = {}
        token_logits_out = {} if return_token_logits else None
        for idx, name in enumerate(self.agent_names):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            actions_out[name] = actor_out["env_actions"][start:end].detach().cpu().numpy()
            log_probs[name] = actor_out["log_prob"][start:end]
            entropies[name] = actor_out["entropy"][start:end]
            action_bins_out[name] = actor_out["action_bins"][start:end]
            if return_token_logits:
                token_logits_out[name] = actor_out["token_logits"][start:end]
        return actions_out, log_probs, entropies, action_bins_out, critic_visual_feature, token_logits_out

    def _compute_value(self, batch: Dict[str, Any], critic_visual_feature: torch.Tensor) -> torch.Tensor:
        global_state = batch["global_state"].to(device=self.actor.device, dtype=torch.float32)
        global_state = self._normalize_global_state(global_state)
        global_feature = self.critic_state_encoder(global_state)
        critic_input = torch.cat([global_feature, critic_visual_feature], dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def get_action_and_value(
        self,
        batch: Dict[str, Any],
        actions_input=None,
        action_bins_input=None,
        return_action_bins: bool = False,
        return_token_logits: bool = False,
    ):
        (
            actions_out,
            log_probs,
            entropies,
            action_bins_out,
            critic_visual_feature,
            token_logits_out,
        ) = self._run_shared_actor(
            batch,
            actions_input=actions_input,
            action_bins_input=action_bins_input,
            deterministic=False,
            return_token_logits=return_token_logits,
        )
        value = self._compute_value(batch, critic_visual_feature)
        if return_action_bins and return_token_logits:
            return actions_out, log_probs, entropies, value, action_bins_out, token_logits_out
        if return_action_bins:
            return actions_out, log_probs, entropies, value, action_bins_out
        if return_token_logits:
            return actions_out, log_probs, entropies, value, token_logits_out
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch: Dict[str, Any], deterministic: bool = False) -> Dict[str, np.ndarray]:
        actions_out, _, _, _, _, _ = self._run_shared_actor(
            batch,
            actions_input=None,
            deterministic=deterministic,
        )
        return actions_out

    def get_value(self, batch: Dict[str, Any]) -> torch.Tensor:
        _, _, _, _, critic_visual_feature, _ = self._run_shared_actor(
            batch,
            actions_input=None,
            deterministic=True,
        )
        return self._compute_value(batch, critic_visual_feature)

    def forward(self, batch: Dict[str, Any]):
        _, log_probs, _, values = self.get_action_and_value(batch)
        return log_probs, values

    @torch.no_grad()
    def update_state_stats(
        self,
        obs: Dict[str, Any],
        *,
        update_actor: bool = True,
        update_critic: bool = True,
    ) -> None:
        parsed = build_batch_from_obs(obs, self.agent_names)
        if update_actor and self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(
                    parsed[f"agent_states_{name}"].to(device=self.actor.device, dtype=torch.float32)
                )
        if update_critic and self.critic_state_rms is not None:
            self.critic_state_rms.update(parsed["global_state"].to(device=self.actor.device, dtype=torch.float32))

    def freeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].freeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.freeze()

    def unfreeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].unfreeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.unfreeze()

    def checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        state = self.state_dict()
        if not self.actor.use_lora:
            return state
        filtered_state = {}
        for key, value in state.items():
            if key.startswith("actor.vla."):
                if "lora_" not in key.lower():
                    continue
            filtered_state[key] = value
        return filtered_state

    def load_checkpoint_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        current_state = self.state_dict()
        filtered_state = {}
        skipped_shape_mismatch = []
        unexpected = []

        for key, value in state_dict.items():
            if key not in current_state:
                if self.actor.use_lora and key.startswith("actor.vla.") and "lora_" not in key.lower():
                    continue
                unexpected.append(key)
                continue
            if current_state[key].shape != value.shape:
                skipped_shape_mismatch.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue
            filtered_state[key] = value

        missing, unexpected_from_load = self.load_state_dict(filtered_state, strict=False)
        unexpected.extend(unexpected_from_load)

        if self.actor.use_lora:
            missing = [
                key for key in missing
                if not (key.startswith("actor.vla.") and "lora_" not in key.lower())
            ]
            unexpected = [key for key in unexpected if "lora_" not in key.lower()]

        if skipped_shape_mismatch:
            mismatch_text = ", ".join(
                f"{key}: ckpt{src_shape}->model{dst_shape}"
                for key, src_shape, dst_shape in skipped_shape_mismatch
            )
            print(f"[Checkpoint] Skipped shape-mismatched keys: {mismatch_text}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")


def build_optimizer(args, agent: MultiAgentVLAAdapterMAPPOAgent) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": [p for p in agent.actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
            "group_name": "vla",
        },
        {
            "params": [p for p in agent.actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
            "group_name": "state_projector",
        },
        {
            "params": [p for p in agent.critic_state_encoder.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic_state_encoder",
        },
        {
            "params": [p for p in agent.critic_visual_encoder.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic_visual_encoder",
        },
        {
            "params": [p for p in agent.critic.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic",
        },
    ]
    if agent.actor.policy_mode == "residual":
        param_groups.insert(
            2,
            {
                "params": [p for p in agent.actor.context_projector.parameters() if p.requires_grad],
                "lr": args.head_learning_rate,
                "group_name": "context_projector",
            },
        )
        param_groups.insert(
            3,
            {
                "params": [p for p in agent.actor.actor_head.parameters() if p.requires_grad],
                "lr": args.head_learning_rate,
                "group_name": "actor_head",
            },
        )
    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)
