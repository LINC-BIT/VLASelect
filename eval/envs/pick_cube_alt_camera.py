import numpy as np

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("PickCubeAltCamera-v1", max_episode_steps=50)
class PickCubeAltCameraEnv(PickCubeEnv):
    """PickCube-v1 with a front overhead camera viewpoint."""

    camera_eye = [0.55, 0.0, 0.9]
    camera_target = [-0.08, 0.0, 0.08]

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=self.camera_eye, target=self.camera_target)
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.camera_eye, target=self.camera_target)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
