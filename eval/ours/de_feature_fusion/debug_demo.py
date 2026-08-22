import torch
import sys; sys.path.append('.')
from train.toy_cnn.model import Actor
from ours.de_feature_fusion.client import Client
import copy
import time


def run_client1():
    model1 = Actor(30, 10, 1, False)
    client1 = Client(
        name='client1',
        large_model=model1,
        layer_name_of_output_features='decoder.0',
        local_feature_dim=512,
        feature_selector_alpha=0.5,
        data_manager_url='http://localhost:8000'
    )

    num_iterations = 10
    num_steps_in_rollout = 20
    num_steps_in_updating = 5
    batch_size = 64

    small_model = copy.deepcopy(model1)
    client1.before_training_start(small_model=small_model)

    for iteration_idx in range(num_iterations):
        print(f'Client1 starts iteration {iteration_idx}')

        # rollout阶段
        
        for step_idx in range(num_steps_in_rollout):
            small_model(torch.rand(batch_size, 3, 128, 128), torch.rand(batch_size, 1, 128, 128), torch.rand(batch_size, 30)) # 模拟做forward
            rewards = torch.randn(batch_size)

            client1.after_each_forward_during_rollout(rewards)
            time.sleep(1)

        # 更新阶段
        for step_idx in range(num_steps_in_updating):
            time.sleep(1)

        client1.refresh_features() # 在每次rollout开始前（除了第一次rollout）刷新一次特征，确保使用的是其它客户端最新的特征


def run_client2():
    model2 = Actor(20, 20, 1, False)
    client2 = Client(
        name='client2',
        large_model=model2,
        layer_name_of_output_features='decoder.2',
        local_feature_dim=256,
        feature_selector_alpha=0.5,
        data_manager_url='http://localhost:8000'
    )

    num_iterations = 10
    num_steps_in_rollout = 15
    num_steps_in_updating = 6
    batch_size = 64

    small_model = copy.deepcopy(model2)
    client2.before_training_start(small_model=small_model)

    for iteration_idx in range(num_iterations):
        print(f'Client 2 starts iteration {iteration_idx}') # 用于观察两个客户端的迭代是否同步开始
        # rollout阶段
        for step_idx in range(num_steps_in_rollout):
            small_model(torch.rand(batch_size, 3, 128, 128), torch.rand(batch_size, 1, 128, 128), torch.rand(batch_size, 20)) # 模拟做forward
            rewards = torch.randn(batch_size)

            client2.after_each_forward_during_rollout(rewards)
            time.sleep(1)

        # 更新阶段
        for step_idx in range(num_steps_in_updating):
            time.sleep(1)

        client2.refresh_features() # 在每次rollout开始前（除了第一次rollout）刷新一次特征，确保使用的是其它客户端最新的特征


if __name__ == "__main__":
    client1_process = torch.multiprocessing.Process(target=run_client1)
    client2_process = torch.multiprocessing.Process(target=run_client2)
    client1_process.start()
    client2_process.start()
    client1_process.join()
    client2_process.join()
