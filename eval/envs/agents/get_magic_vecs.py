import numpy as np
from scipy.spatial.transform import Rotation as R

def T_from_origin(xyz, rpy):
    Rmat = R.from_euler("xyz", rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = xyz
    return T

# ---------- 1. 构建变换 ----------
# 从urdf读取joint坐标，并转换成4x4变换矩阵
T_arm_r1 = T_from_origin(
    [-0.0035, -0.012625, -0.0685],
    [0, -1.5708, 0]
)

T_arm_r3 = T_from_origin(
    [-0.0035, -0.0045, -0.0505],
    [0, -1.5708, 0]
)

T_r1_r2 = T_from_origin(
    [0.03, -0.00075429, 0],
    [0, 0, 0]
)

T_arm_r2 = T_arm_r1 @ T_r1_r2

T_arm_l1 = T_from_origin(
    [-0.0035, 0.012375, -0.0685],
    [0, -1.5708, 0]
)

T_arm_l3 = T_from_origin(
    [-0.0035, 0.0045, -0.0505],
    [0, -1.5708, 0]
)

T_l1_l2 = T_from_origin(
    [0.03, 0.00065104, 0],
    [0, 0, 0]
)

T_arm_l2 = T_arm_l1 @ T_l1_l2

# ---------- 2. 取位置 ----------

A_r = T_arm_r1[:3, 3]
B_r = T_arm_r3[:3, 3]
D_r = T_arm_r2[:3, 3]

A_l = T_arm_l1[:3, 3]
B_l = T_arm_l3[:3, 3]
D_l = T_arm_l2[:3, 3]
# ---------- 3. 平行四边形公式 ----------

C_r = B_r + (D_r - A_r)
C_l = B_l + (D_l - A_l)

# ---------- 4. 转局部坐标 ----------

C_h_r = np.append(C_r, 1)

C_local_r3 = np.linalg.inv(T_arm_r3) @ C_h_r
C_local_r2 = np.linalg.inv(T_arm_r2) @ C_h_r

C_h_l = np.append(C_l, 1)

C_local_l3 = np.linalg.inv(T_arm_l3) @ C_h_l
C_local_l2 = np.linalg.inv(T_arm_l2) @ C_h_l

print("C_local_r3:", C_local_r3[:3])
print("C_local_r2:", C_local_r2[:3])
print("C_local_l3:", C_local_l3[:3])
print("C_local_l2:", C_local_l2[:3])
