# VLASelect Model Interface

VLASelect supports multiple vision-language-action (VLA) models through the
`VLAModelInterface` defined in [`api/vla_model_interface.py`](../vla_model_interface.py).
The interface isolates model-specific integration code from the shared continual-learning
runner. Rollout collection, GAE, PPO updates, training metrics, checkpointing, plots, and
continual-environment scheduling remain in the common runner.

## Unified Verification

The VLA-Adapter verification entry point selects the scaling or knowledge-exchange
implementation from the command line. The two selectors are mutually exclusive:

```bash
MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify.sh \
  --scaling-method logit_distillation
MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify.sh \
  --knowledge-exchange-granularity block
```
Use `layer`, `block`, or `attention_head` for knowledge-exchange granularity.
`attention_head` keeps each attention head's contiguous QKV channels together;
`head` is accepted as a short alias by `vla_adapter_impl.py`.

With `MWE=1`, training runs until the five-minute wall-clock limit. Results are written
under model-specific directories: `api/results/vla_adapter/{scaling_methods,knowledge_exchange}`
or `api/results/tinyvla/{scaling_methods,knowledge_exchange}`. Plot completed runs with:

```bash
python api/plot_scaling_methods.py
python api/plot_knowledge_exchange.py
```

## Custom Small-Model Scaling

[`SmallModelScalingInterface`](../small_model_scaling_interface.py) separates
small-policy sampling, static FBS generation, regeneration decisions, retained-channel
inheritance, and optimizer reset from the shared training loop. `run_training` uses its
default implementation unless an implementation is explicitly supplied:

```python
run_training(adapter, args, small_model_scaling_interface=my_generator)
```

Subclass it to change the small-model scaling workflow while leaving VLA adaptation,
rollouts, PPO, feedback, checkpointing, and environment scheduling unchanged. The
[`logit_distillation_impl.py`](../small_model_scaling_interface_examples/logit_distillation_impl.py)
example reuses the default static generation flow and adds behavioral knowledge
distillation through `after_small_model_scaling`. Its distillation hyperparameters are
constructor-only. Run its verification entry point with:

```bash
MWE=1 bash api/small_model_scaling_interface_examples/knowledge_distillation_impl_verify.sh
```

Distillation strategies expose the constructor option
`randomize_student_parameters` (enabled by default). VLA-Adapter keeps the default
random initialization, while EdgeVLA and TinyVLA disable it so distillation starts
from the channels retained during FBS materialization.

An adapter must implement the API groups below. The adapter is also responsible for
declaring the model architecture explicitly. Architecture discovery from checkpoint files
is not part of the contract.

## Data Contracts

### `EnvironmentContract`

```python
@dataclass(frozen=True)
class EnvironmentContract:
    state_dim: int
    action_dim: int
    env_action_dim: int
    controlled_action_indices: Tuple[int, ...]
```

Describes the state and action dimensions expected by the policy and the environment.
`controlled_action_indices` maps policy action channels into the environment action vector.
The constructor validates positive dimensions, unique indices, and index bounds.

### `PolicyBatch`

```python
@dataclass
class PolicyBatch:
    rgbs: np.ndarray
    states: np.ndarray
```

The canonical observation batch passed to policy-facing APIs. `rgbs` contains batched
`uint8` images and `states` contains batched `float32` state features.

### `FBSLayerGroups`

```python
@dataclass(frozen=True)
class FBSLayerGroups:
    vision_qkv: Tuple[str, ...] = ()
    vision_proj: Tuple[str, ...] = ()
    vision_ff1: Tuple[str, ...] = ()
    vision_ff2: Tuple[str, ...] = ()
    language_qkv: Tuple[Tuple[str, ...], ...] = ()
    language_proj: Tuple[str, ...] = ()
    language_ff1: Tuple[Tuple[str, ...], ...] = ()
    language_ff2: Tuple[str, ...] = ()
```

Contains the exact module paths used by dynamic and static FBS conversion. Language QKV
and FFN groups are nested because decoder implementations commonly expose separate
`q_proj`/`k_proj`/`v_proj` and gated FFN projections.

### `ModelArchitectureSpec`

```python
@dataclass(frozen=True)
class ModelArchitectureSpec:
    architecture_name: str
    policy_class_name: str
    state_dim: int
    action_dim: int
    env_action_dim: int
    fbs_layers: FBSLayerGroups
    architecture_config: Mapping[str, Any] = field(default_factory=dict)
```

Declares the concrete actor-critic class, dimensions, architecture configuration, and FBS
paths independently of checkpoint availability. The constructor rejects empty identity
fields and empty transformer declarations. If a declared FBS path is missing from the
constructed policy, conversion must fail loudly instead of silently skipping that layer.

## Identity and Architecture APIs

```python
@property
def model_name(self) -> str: ...

@property
def architecture_spec(self) -> ModelArchitectureSpec: ...

@property
def policy_class(self) -> type[nn.Module]: ...
```

`model_name` is a stable filesystem-safe identifier used for logs and checkpoints.
`architecture_spec` is the mandatory architecture declaration and must not inspect loaded
weights to decide what the model is. `policy_class` returns the concrete actor-critic class
that the adapter constructs, such as `HandVLAAdapterActorCritic` or `EdgeVLAActorCritic`.

## Workload and Environment APIs

```python
def register_workloads(self) -> None: ...

def inspect_environment_contract(
    self,
    env_id: str,
    *,
    device: torch.device,
    config: Mapping[str, Any],
) -> EnvironmentContract: ...

def apply_environment_contract(
    self,
    args: Any,
    *,
    env_id: str,
    device: torch.device,
) -> None: ...

def make_vector_env(
    self,
    args: Any,
    *,
    device: torch.device,
    env_id: str,
    num_envs: int,
    record_metrics: bool = True,
    video_output_dir: Optional[Path] = None,
    video_max_steps: Optional[int] = None,
) -> Any: ...
```

`register_workloads` imports or registers model-specific environments before environment
creation. `inspect_environment_contract` probes the selected environment and returns the
actual state/action mapping. `apply_environment_contract` copies that resolved contract
into the shared runner arguments. `make_vector_env` creates the vectorized training or
evaluation environment, including reference wrappers, metric recording, and optional
video recording.

## Observation APIs

```python
def extract_observations(self, obs: Any) -> PolicyBatch: ...

def extract_rgb_batch_from_obs(self, obs: Any) -> torch.Tensor: ...

def extract_state_batch_from_obs(self, obs: Any) -> Any: ...
```

These functions translate workload-specific ManiSkill observations into the canonical
policy representation. RGB output must be a CPU `uint8` tensor shaped like
`[batch, height, width, 3]`; state output must be compatible with a `float32` array of
shape `[batch, state_dim]`. `extract_observations` is the combined convenience API used by
the shared runner.

## Policy and FBS APIs

```python
def build_policy(
    self,
    model_dir: Path,
    *,
    args: Any,
    device: torch.device,
) -> nn.Module: ...

def convert_to_fbs_policy(
    self,
    policy: nn.Module,
    *,
    device: torch.device,
    max_sparsity: float,
) -> nn.Module: ...

def restore_policy_after_fbs(
    self,
    policy: nn.Module,
    *,
    device: torch.device,
) -> nn.Module: ...
```

`build_policy` constructs the full actor-critic model from the original model directory.
The API examples require that directory and do not provide a random-initialization
fallback; a structurally different surrogate is not valid. `convert_to_fbs_policy` invokes the real
reference FBS flow, including SVD decomposition, FBS module insertion, sparsity
initialization, and forward-equivalence verification. It must use
`architecture_spec.fbs_layers`. `restore_policy_after_fbs` restores device, dtype, buffers,
and model-specific runtime attributes after conversion.

## Optimization APIs

```python
def configure_trainable_modules(
    self,
    policy: nn.Module,
    *,
    train_backbone: bool,
) -> None: ...

def build_optimizer(
    self,
    policy: nn.Module,
    *,
    config: Mapping[str, Any],
) -> Optimizer: ...

def set_backbone_learning_rate(
    self,
    optimizer: Optimizer,
    learning_rate: float,
) -> None: ...
```

`configure_trainable_modules` sets `requires_grad` for the VLA backbone and task heads
according to the training schedule. `build_optimizer` creates the named parameter groups
used by the reference PPO implementation, typically including backbone, state projector,
context, actor, and value groups. `set_backbone_learning_rate` updates the backbone group
when the warmup period ends.

## Action and Value APIs

```python
def get_action_and_value(
    self,
    policy: nn.Module,
    batch: PolicyBatch,
    *,
    action_bins: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]: ...

def get_value(
    self,
    policy: nn.Module,
    batch: PolicyBatch,
) -> torch.Tensor: ...
```

`get_action_and_value` returns five batched tensors: environment actions, summed action
log-probability, entropy, scalar value estimates, and discrete action-bin indices. It must
support both sampled rollout actions and provided `action_bins` for PPO updates;
`deterministic=True` is used for evaluation. `get_value` returns one bootstrap value per
observation and is used when computing GAE.

## Checkpoint Lifecycle API

```python
def prepare_policy_for_checkpoint_load(
    self,
    policy: nn.Module,
) -> None: ...
```

Runs model-specific preparation immediately before the shared runner loads a checkpoint,
for example switching evaluation mode or preparing device/dtype state. Checkpoint format,
save paths, and load orchestration remain owned by the shared runner.

## Implementation Checklist

When adding a new VLA adapter:

1. Implement every abstract method in `VLAModelInterface`.
2. Declare the concrete actor-critic class and architecture configuration in
   `ModelArchitectureSpec`.
3. Declare every vision and language FBS path explicitly.
4. Ensure the no-checkpoint construction path preserves the declared module structure.
5. Keep environment action mapping inside `EnvironmentContract` and the environment APIs.
6. Reuse the reference FBS, PPO, evaluation, checkpoint, and continual-learning logic.
7. Run the adapter verification script with `MWE=1` and confirm rollout metrics,
   evaluation metrics, FBS verification, and checkpoint files are produced.

## Verification Examples

```bash
MWE=1 bash ./api/vla_model_interface_examples/vla_adapter_impl_verify.sh
MWE=1 bash ./api/vla_model_interface_examples/tinyvla_impl_verify.sh
```

`MWE=1` only reduces the collected experience and wall-clock guard. The environment,
model construction, FBS conversion, continual-learning schedule, PPO updates, evaluation,
and checkpoint logic use the same implementation as the full run.
