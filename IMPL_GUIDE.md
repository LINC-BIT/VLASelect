# Implementation guidance of customized VLA models, scaling strategies, and knowledge exchange granularities

## 1. Implementing a new VLA model

We develop an interface `VLAModelInterface` (`api/vla_model_interface.py`) to decouple VLASelect's implementation from VLA models. To support a new VLA model in VLASelect, developers can implement the following APIs in `VLAModelInterface`:

- Model Initialization
  - `model_name(self) -> str`<br>
    Return a stable, filesystem-safe model name for logs and checkpoint metadata.
  - `architecture_spec(self) -> ModelArchitectureSpec`<br>
    Declare policy class identity and all FBS layer paths without inspecting weights.
  - `policy_class(self) -> type[nn.Module]`<br>
    Return the concrete actor-critic class used by this adapter.
  - `register_workloads(self) -> None`<br>
    Register/import model-specific environment workloads before creating environments.
  - `build_policy(self, model_dir: Path, *, args: Any, device: torch.device) -> nn.Module`<br>
    Construct the full actor-critic policy for the resolved runner arguments.
  - `convert_to_fbs_policy(self, policy: nn.Module, *, device: torch.device, max_sparsity: float) -> nn.Module`<br>
    Insert the real FBS modules into a full policy and return that instrumented policy. This is the dynamic FBS stage from the reference implementation, including SVD decomposition, FBS module insertion, sparsity initialization, and forward-equivalence verification. Static small-policy generation is intentionally a separate shared-runner step.
  - `restore_policy_after_fbs(self, policy: nn.Module, *, device: torch.device) -> nn.Module`<br>
    Restore device, dtype, and model-specific runtime attributes after FBS conversion.
  - `inspect_environment_contract(self, env_id: str, *, device: torch.device, config: Mapping[str, Any]) -> EnvironmentContract`<br>
    Probe an environment and return state/action dimensions and controlled indices.
  - `apply_environment_contract(self, args: Any, *, env_id: str, device: torch.device) -> None`<br>
    Write the resolved state/action dimensions and action mapping into runner args. Implement this when a policy action space differs from the environment action space, for example TinyVLA's controlled action channels.
  - `make_vector_env(self, args: Any, *, device: torch.device, env_id: str, num_envs: int, record_metrics: bool = True, video_output_dir: Optional[Path] = None, video_max_steps: Optional[int] = None) -> Any`<br>
    Create a vectorized environment matching the adapter's observation/action schema.
- Forward Operations
  - `get_value(self, policy: nn.Module, batch: PolicyBatch) -> torch.Tensor`<br>
    Return one scalar bootstrap value per observation in ``batch``.
  - `get_action_and_value(self, policy: nn.Module, batch: PolicyBatch, *, action_bins: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`<br>
    Return env actions, summed log-probability, entropy, value, and discrete action bins.
  - `extract_observations(self, obs: Any) -> PolicyBatch`<br>
    Convert raw environment observations into uint8 RGB and float32 state batches.
  - `extract_rgb_batch_from_obs(self, obs: Any) -> torch.Tensor`<br>
    Extract a CPU uint8 ``[batch, height, width, 3]`` image tensor from env observations.
  - `extract_state_batch_from_obs(self, obs: Any) -> Any`<br>
    Extract a float32 ``[batch, state_dim]`` NumPy-compatible state array.
- Backward Operations
  - `configure_trainable_modules(self, policy: nn.Module, *, train_backbone: bool) -> None`<br>
    Set ``requires_grad`` for backbone and task heads according to the training schedule.
  - `build_optimizer(self, policy: nn.Module, *, config: Mapping[str, Any]) -> Optimizer`<br>
    Create named optimizer parameter groups for backbone, heads, state, and value modules.
  - `set_backbone_learning_rate(self, optimizer: Optimizer, learning_rate: float) -> None`<br>
    Update the adapter's backbone optimizer group when a warmup period ends.


## 2. Implementing a new model scaling strategy

We develop an interface `SmallModelScalingInterface`
(`api/small_model_scaling_interface.py`) to decouple VLASelect's online RL process
from the strategy used to scale the small model. To support a new model scaling strategy (e.g. knowledge distillation), developers can implement the following APIs in `SmallModelScalingInterface`:

- Scaling Sample Collection
  - `collect_sample_for_small_model_scaling(args, *, large_agent, small_agent, eval_envs, device, adapter, reference_api)`<br>
    Build the RGB, state, and action-bin sample batch used for static small-model scaling. The default implementation supports `target-batch`, `target-single`, and `target-single-traj`.
- Small Model Scaling
  - `generate_initial_small_model(*, large_agent, args, eval_envs, device, adapter, reference_api)`<br>
    Generate the initial static small model and its pruning metadata before online RL begins.
  - `regenerate_small_model_in_place(*, large_agent, small_agent, current_pruning_info, optimizer, args, eval_envs, device, adapter, reference_api)`<br>
    Regenerate the static small model during training, inherit retained channels when incremental regeneration is enabled, update the current model in place, and reset its optimizer state when configured.
  - `after_small_model_scaling(*, large_agent, small_agent, sample_batch, pruning_info, optimizer, args, device, adapter, reference_api)`<br>
    Customize a scaled model before the runner uses it. The optimizer is `None` for initial scaling and is the active PPO optimizer during regeneration. Strategy-specific hyperparameters should be stored in the interface instance during initialization rather than added to this runtime method.
- Regeneration Scheduling
  - `should_regenerate_small_model_before_rollout(schedule, update, start_update, current_success_end, success_end_at_last_regeneration, update_at_last_regeneration)`<br>
    Decide whether regeneration should run before the next rollout. The default implementation supports one-time generation, per-rollout regeneration, success-improvement thresholds, and insufficient-improvement windows.
  - `maybe_regenerate_small_model_before_rollout(*, args, update, start_update, current_success_end, success_end_at_last_regeneration, update_at_last_regeneration, large_agent, small_agent, current_pruning_info, optimizer, eval_envs, device, adapter, reference_api)`<br>
    Apply the regeneration decision and return whether regeneration occurred together with the current pruning metadata.


## 3. Implementing a new knowledge exchange granularity

We develop an interface `GranularitySmallModelScalingInterface` (`api/knowledge_exchange_granularity_interface.py`) to decouple VLASelect's implementation from knowledge exchange granularities. To support a new knowledge exchange granularity in VLASelect, developers can implement the following APIs in `GranularitySmallModelScalingInterface`:

- `group_fbs_layers(fbs_layers)`<br>
  Partition FBS modules into selectable groups (i.e. a large granularity than neuron). The default layer-grained behavior
  groups modules belonging to the same transformer layer.
- `score_groups(actor, groups)`<br>
  Compute the mean cached FBS score for each group/granularity after the generation sample has
  been evaluated.
- `select_high_score_groups(group_scores, max_sparsity)`<br>
  Select the highest-scoring groups. `max_sparsity` controls the fraction of groups/granularity
  that are reduced; retained groups keep all of their neurons.
