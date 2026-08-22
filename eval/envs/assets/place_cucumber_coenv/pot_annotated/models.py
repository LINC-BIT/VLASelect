import trimesh
import json
import numpy as np


PI = np.pi
# URDF_MATRIX = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
URDF_MATRIX = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
ROTATE_MATRIX = trimesh.transformations.euler_matrix(PI / 2,0,-PI / 2)
def create_model_data(id):
    save_path = f"./model_data{id}.json"

    # with open(save_path, 'r') as json_file:
    #     data = json.load(json_file)
        
    scene = trimesh.Scene()
    id_lst = [1,3,4,5,6,7,8,10,11,13,14]

    for id in id_lst:
        file_path = f"./textured_objs/original-{id}.obj"
        with open(file_path, 'rb') as file_obj:
            mesh = trimesh.load(file_obj, file_type='obj')
        scene.add_geometry(mesh)

    # file_path = f"./textured_objs/original-21.obj"
    # with open(file_path, 'rb') as file_obj:
    #     mesh = trimesh.load(file_obj, file_type='obj')
    # scene.add_geometry(mesh)

    # file_path = f"./textured_objs/original-22.obj"
    # with open(file_path, 'rb') as file_obj:
    #     mesh = trimesh.load(file_obj, file_type='obj')
    # scene.add_geometry(mesh)

    oriented_bounding_box = mesh.bounding_box_oriented
    red_color = [1.0, 0.0, 0.0, 0.5]  # 红色, A=1 表示不透明
    blue_color = [0.0, 0.0, 1.0, 0.5]
    green_color = [0.0, 1.0, 0.0, 0.5]
    scale = [0.2] * 3

    target_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    target_trans_matrix_sphere = URDF_MATRIX @ trimesh.transformations.translation_matrix([0,0.7,0]) @ trimesh.transformations.euler_matrix(PI / 3,0,0)
    target_sphere.apply_transform(target_trans_matrix_sphere)
    target_sphere.visual.vertex_colors = np.array([blue_color] * len(target_sphere.vertices))
    target_points_list = [(ROTATE_MATRIX @ target_trans_matrix_sphere).tolist()]
    axis_t = trimesh.creation.axis(axis_length=1.5)
    axis_t.apply_transform(target_trans_matrix_sphere)
    # scene.add_geometry(axis_t)
    
    target_matrix = (ROTATE_MATRIX @ target_trans_matrix_sphere).tolist()
    # functional_sphere = trimesh.creation.icosphere(subdivisions=0.7, radius=0.2)
    # functional_trans_matrix_sphere = trimesh.transformations.translation_matrix([0,-0.45,0]) @ trimesh.transformations.euler_matrix(0,0,PI / 2)
    # functional_sphere.apply_transform(functional_trans_matrix_sphere)
    # functional_sphere.visual.vertex_colors = np.array([green_color] * len(functional_sphere.vertices))
    # functional_matrix_1 = (ROTATE_MATRIX @ functional_trans_matrix_sphere).tolist()
    
    # # 可视化网格和坐标轴
    # scene.add_geometry(target_sphere)

    contact_points_list = []
    contact_point_discription_list = []
    orientation_point_list = []
    functional_matrix = []

    def add_contact_point(point_radius, pose:list, euler:list, discription: str):
        contact_sphere = trimesh.creation.icosphere(subdivisions=2, radius=point_radius)
        contact_trans_matrix_sphere = trimesh.transformations.translation_matrix(pose) @ trimesh.transformations.euler_matrix(euler[0], euler[1], euler[2])
        contact_sphere.apply_transform(contact_trans_matrix_sphere)
        contact_sphere.visual.vertex_colors = np.array([red_color] * len(contact_sphere.vertices))
        contact_points_list.append((ROTATE_MATRIX @ contact_trans_matrix_sphere).tolist())

        axis = trimesh.creation.axis(axis_length=0.6,origin_size= 0.01)
        axis.apply_transform(contact_trans_matrix_sphere)
        # scene.add_geometry(axis)
        # scene.add_geometry(contact_sphere)
        contact_point_discription_list.append(discription)

    # old
    add_contact_point(0.02, [0, 0.55, 0], [0 , 0, 0], "the grasp side of the lid.")
    add_contact_point(0.02, [0, 0.55, 0], [0 ,PI / 2, 0], "the grasp side of the lid.")
    add_contact_point(0.02, [0, 0.55, 0], [0 , PI , 0], "the grasp side of the lid.")
    add_contact_point(0.02, [0, 0.55, 0], [0 , -PI / 2, 0], "the grasp side of the lid.")

    add_contact_point(0.02, [0.87, 0.18, 0], [0 , PI / 2, -PI / 3], "the grasp side of the handle.")
    add_contact_point(0.02, [-0.87, 0.18, 0], [0 , - PI / 2, PI / 3], "the grasp side of the handle.")

    # add_contact_point(0.02, [0, 0.33, 0.59], [PI/2, 0, PI], "Handle center grip position for cabinet")
    # add_contact_point(0.02, [0.2, 0.33, 0.59], [PI/2, 0, 0], "Handle left grip position for cabinet")
    # add_contact_point(0.02, [0.2, 0.33, 0.59], [PI/2, 0, PI], "Handle left grip position for cabinet")
    # add_contact_point(0.02, [-0.2, 0.33, 0.59], [PI/2, 0, 0], "Handle right grip position for cabinet")
    # add_contact_point(0.02, [-0.2, 0.33, 0.59], [PI/2, 0, PI], "Handle right grip position for cabinet")

    
    # 旋转矩阵的参数顺序是 (X, Y, Z)   to endpose axis
    transform_matrix = trimesh.transformations.euler_matrix(0,0,0)
    transform_matrix = transform_matrix.tolist()

    # 方向点
    # orientation_point = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    # orientation_point_sphere = trimesh.transformations.translation_matrix([0,oriented_bounding_box.extents[1]/2,0])
    # orientation_point.apply_transform(orientation_point_sphere)
    # orientation_point.visual.vertex_colors = np.array([red_color] * len(orientation_point.vertices))
    # orientation_point_list = orientation_point_sphere.tolist()
    # scene.add_geometry(orientation_point)

    # axis1
    axis1 = trimesh.creation.axis(axis_length=2)
    axis1.apply_transform(transform_matrix)

    functional_sphere = trimesh.creation.icosphere(subdivisions=0.7, radius=0.2)
    functional_trans_matrix_sphere = trimesh.transformations.translation_matrix([0,-0.45,0]) @ trimesh.transformations.euler_matrix(0,0,PI / 2)
    functional_sphere.apply_transform(functional_trans_matrix_sphere)
    functional_sphere.visual.vertex_colors = np.array([green_color] * len(functional_sphere.vertices))
    functional_matrix_1 = (ROTATE_MATRIX @ functional_trans_matrix_sphere).tolist()
    
    functional_sphere_2 = trimesh.creation.icosphere(subdivisions=0.7, radius=0.2)
    functional_trans_matrix_sphere_2 = trimesh.transformations.translation_matrix([0,0.5,0]) @ trimesh.transformations.euler_matrix(0,0,PI / 2)
    functional_sphere_2.apply_transform(functional_trans_matrix_sphere_2)
    functional_sphere_2.visual.vertex_colors = np.array([green_color] * len(functional_sphere_2.vertices))
    functional_matrix_2 = (ROTATE_MATRIX @ functional_trans_matrix_sphere_2).tolist()
    axis = trimesh.creation.axis(axis_length=1.5)
    axis.apply_transform(functional_trans_matrix_sphere)
    axis2 = trimesh.creation.axis(axis_length=1.5)
    axis2.apply_transform(functional_trans_matrix_sphere_2)
    # scene.add_geometry(axis)
    # scene.add_geometry(axis2)

    # scene.add_geometry(functional_sphere)
    # scene.add_geometry(functional_sphere_2)

    data = {
        'center': oriented_bounding_box.centroid.tolist(),  # 中心点
        'extents': oriented_bounding_box.extents.tolist(),   # 尺寸
        'scale': scale,                                     # 缩放
        'target_pose': target_points_list,              # 目标点矩阵
        'target_matrix': [target_matrix],
        'contact_points_pose' : contact_points_list,    # 抓取点矩阵（多个）
        'transform_matrix': transform_matrix,           # 模型到标轴的旋转矩阵
        "functional_matrix": [functional_matrix_1, functional_matrix_2],         # 功能点矩阵
        'orientation_point': orientation_point_list,
        'contact_points_group': [[0,1],[2,3],[4,5]],
        'contact_points_mask': [True, True, True],
        'contact_points_discription': contact_point_discription_list,    # 抓取点描述
        'target_point_discription': ["The center of the pot."],  # 目标点描述
        'functional_point_discription': ["The base of the pot", "the lid of the pot"],
        'orientation_point_discription': [""],
        "model_type": "urdf"
    }
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, separators=(',', ': '))

    # 将坐标轴添加到场景
    # axis = trimesh.creation.axis(axis_length=1.5,origin_size= 0.05)
    # scene.add_geometry(axis1)
    scene.show()

if __name__ == "__main__":
    id = ""
    create_model_data(id)