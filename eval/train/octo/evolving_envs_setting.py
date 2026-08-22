import random
import copy
import json


def get_num_evolving_envs(setting_fp):
    with open(setting_fp, 'r') as f:
        setting = json.load(f)
    return len(setting)


def read_evolving_envs_setting(setting_fp, env_idx):
    with open(setting_fp, 'r') as f:
        setting = json.load(f)
    return setting[env_idx]


def generate_evolving_envs_setting(num_evolving_envs):
    """
    生成完全随机的测试环境设置（各环境相互独立）
    """
    object_types = ["cube", "sphere", "cylinder", "box"]

    base_object_size_info = {
        'cube': {'half_size': 0.03},
        'sphere': {'radius': 0.03},
        'cylinder': {'radius': 0.03, 'half_length': 0.03},
        'box': {'half_sizes': [0.03, 0.04, 0.05]},
    }

    env_settings = []

    for _ in range(num_evolving_envs):
        # 1. 随机物体类型
        object_type = random.choice(object_types)

        # 2. 随机尺寸（在基准尺寸附近扰动）
        object_size_info = copy.deepcopy(base_object_size_info)
        size_scale = random.uniform(0.7, 1.3)

        if object_type == "sphere":
            object_size_info["sphere"]["radius"] *= size_scale
        elif object_type == "cube":
            object_size_info["cube"]["half_size"] *= size_scale
        elif object_type == "cylinder":
            object_size_info["cylinder"]["radius"] *= size_scale
            object_size_info["cylinder"]["half_length"] *= size_scale
        elif object_type == "box":
            object_size_info["box"]["half_sizes"] = [
                s * size_scale for s in object_size_info["box"]["half_sizes"]
            ]

        # 3. 随机质量
        object_mass = random.uniform(0.1, 1.0)  # kg

        # 4. 随机颜色（RGBA）
        object_color = [
            random.random(),
            random.random(),
            random.random(),
            1.0
        ]

        # 5. 随机摄像头
        randomize_camera = random.choice([True, False])

        # 6. 光照相关完全随机
        ambient_light_temperature = random.randint(2000, 6500)
        directional_light_temperature = random.randint(2000, 6500)

        ambient_light_intensity = random.uniform(0.1, 1.0)

        directional_light_direction = [
            random.uniform(0.0, 0.5),
            random.uniform(0.0, 0.5),
            random.uniform(0.0, 0.5),
        ]

        shadow_scale = random.randint(5, 10)

        setting = dict(
            object_type=object_type,
            object_size_info=object_size_info,
            object_mass=object_mass,
            object_color=object_color,
            randomize_camera=randomize_camera,
            ambient_light_temperature=ambient_light_temperature,
            ambient_light_intensity=ambient_light_intensity,
            directional_light_temperature=directional_light_temperature,
            directional_light_direction=directional_light_direction,
            shadow_scale=shadow_scale,
        )

        env_settings.append(setting)

    return env_settings





if __name__ == '__main__':
    evolving_envs_setting = generate_evolving_envs_setting(10)
    # print(evolving_envs_setting)
    with open('./train/octo/evolving-envs-setting.json', 'w') as f:
        json.dump(evolving_envs_setting, f, indent=2)
    
