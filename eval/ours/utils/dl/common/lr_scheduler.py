
def get_linear_schedule_with_warmup(num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return lr_lambda



def get_step_lr_schedule_with_warmup(num_warmup_steps, interval, lmba):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # elif (current_step - num_warmup_steps) % interval == 0:
        #     return lmba
        # return 1.0
        return lmba ** ((current_step - num_warmup_steps) // interval)
    return lr_lambda
