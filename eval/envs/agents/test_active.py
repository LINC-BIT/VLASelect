import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils.system.backend import parse_sim_and_render_backend
from mani_skill.utils.building import URDFLoader
loader = URDFLoader()
loader.set_scene(ManiSkillScene(backend=parse_sim_and_render_backend('cpu', 'cpu')))
robot = loader.load("envs/agents/urdfs/amazinghand_right/right_hand_final.urdf")
print(robot.active_joints_map.keys())

# import sys
# import os
# sys.path.append(os.getcwd())
# import envs.agents.amazinghand # imports your robot and registers it
# # imports the demo_robot example script and lets you test your new robot
# import mani_skill.examples.demo_robot as demo_robot_script
# import tyro

# demo_robot_script.main(tyro.cli(demo_robot_script.Args))

# loader = self.scene.create_mjcf_loader()
# asset_path = str(self.mjcf_path)
# builder = loader.parse(asset_path)["articulation_builders"][0]
# builder.build()