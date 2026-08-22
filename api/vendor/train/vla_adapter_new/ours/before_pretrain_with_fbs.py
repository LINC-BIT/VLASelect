import sys; sys.path.append('.')
from train.vla_adapter_new.ours.model_with_fbs_test import convert_to_fbs_model
from train.vla_adapter_new.ours.pretrain_with_fbs import *

device = 'cuda'

def maybe_load_checkpoint(
    # args: Args,
    raw_policy: HandVLAAdapterActorCritic,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[int, int, float]:
    # if not args.resume_from:
    #     return 1, 0, -1.0
    checkpoint = torch.load('ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt', map_location=raw_policy.device)
    policy_state = strip_module_prefix(checkpoint["policy"])
    raw_policy.load_state_dict(policy_state, strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_update = int(checkpoint.get("update", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_success = float(checkpoint.get("best_success_once", -1.0))
    return start_update, global_step, best_success

raw_policy = HandVLAAdapterActorCritic(
    Path('eval/ckpt/vla_adapter_new/LIBERO-Object'),
    device=device,
    state_dim=105,
    action_dim=16,
).to(device)
start_update, global_step, best_success_once = maybe_load_checkpoint(raw_policy)
raw_policy = convert_to_fbs_model(raw_policy, device).to(device)

torch.save(raw_policy.state_dict(), 'train/vla_adapter_new/ours/pretrained_model_with_fbs.pth')
