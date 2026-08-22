import numpy as np

def triangle_wave(t, period):
    """范围 [-1,1] 的三角波"""
    x = (t % period) / period  # [0,1]
    return 4 * np.abs(x - 0.5) - 1


class UniformScanPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.t = 0

        # ⭐ 控制幅度（避免 hitting ±1）
        self.amp = 0.8

        # 每个关节不同周期（避免同步）
        self.periods = np.linspace(250, 300, action_dim)

        # 小扰动（用于拟合）
        self.noise_amp = 0.15
        self.freqs = np.linspace(0.05, 0.1, action_dim)

    def reset(self):
        self.t = 0

    def act(self, obs=None):
        action = np.zeros(self.action_dim)

        for i in range(self.action_dim):
            base = triangle_wave(self.t, self.periods[i])

            # 小正弦扰动（保证可辨识）
            perturb = self.noise_amp * np.sin(self.freqs[i] * self.t)

            action[i] = self.amp * base + perturb

        # 避免贴边
        action = np.clip(action, -0.95, 0.95)

        self.t += 1
        return action
    
    import numpy as np

def smooth_triangle(t, period):
    """平滑三角波，范围 [-1,1]"""
    return np.arcsin(np.sin(2 * np.pi * t / period)) * (2 / np.pi)


class PDJointPosScan:
    """
    适用于 pd_joint_pos 的扫描策略
    - 直接输出 target joint position
    - joint range: [-2, 2]
    """

    def __init__(self, action_dim, joint_limit=2.0):
        self.action_dim = action_dim
        self.joint_limit = joint_limit
        self.t = 0

        # 每个关节周期不同
        self.periods = np.linspace(200, 600, action_dim)

        # 初相
        self.phases = np.linspace(0, 1, action_dim, endpoint=False)

        # 扫描方向（增加多样性）
        self.directions = np.random.choice([-1, 1], action_dim)

        self.perturb_amp = 0.02

    def triangle_wave(self, x):
        """
        ✔ 线性三角波 [-1,1]
        ✔ 纯一次函数拼接
        """
        x = x - np.floor(x)  # [0,1)

        return np.where(
            x < 0.5,
            4 * x - 1,       # 上升段
            -4 * x + 3       # 下降段
        )

    def act(self, obs=None):
        action = np.zeros(self.action_dim)

        for i in range(self.action_dim):

            # =========================
            # 1️⃣ 归一化时间 + 相位
            # =========================
            x = (self.t / self.periods[i]) + self.phases[i]
            x = x - np.floor(x)
            print('t =', x)
            # =========================
            # 2️⃣ 线性三角波 [-1,1]
            # =========================
            base = self.triangle_wave(x)

            # =========================
            # 3️⃣ 方向控制
            # =========================
            # base *= self.directions[i]

            # =========================
            # 4️⃣ 映射到 [-2,2]
            # =========================
            val = self.joint_limit * base

            # =========================
            # 5️⃣ 微扰（用于拟合）
            # =========================
            # val += self.perturb_amp * (2 * np.random.rand() - 1)

            action[i] = val

        self.t += 1
        return action