import torch
import numpy as np
import sapien
import sys
import os
sys.path.append(os.getcwd())
from envs.pick_obj_random import matrix_to_quaternion, euler_angles_to_matrix
print(sapien.Pose(p=[-0.615, 0, 0.005], q=matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, 0, -np.pi/2]), convention="XYZ"))))