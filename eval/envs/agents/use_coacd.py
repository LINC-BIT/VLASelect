import os
import trimesh
import coacd
import numpy as np

def process_stl_folder(input_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".stl"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing: {filename}")

            # 读取 mesh
            mesh = trimesh.load(input_path, force="mesh")

            # 转换为 coacd mesh
            coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

            # 运行 CoACD
            parts = coacd.run_coacd(
                coacd_mesh,
                # threshold=0.02,            # 很小的凹误差
                # resolution=5000,           # 高体素分辨率
                # preprocess_resolution=200,
                # mcts_nodes=40,
                # mcts_iterations=400,
                # mcts_max_depth=5,
                # max_convex_hull=-1,        # 不限制
                # pca=True,
                # merge=False,               # 不合并
                # decimate=False,            # 不简化
                # max_ch_vertex=1024,        # 允许更多顶点
                # extrude=False,
                # apx_mode="ch",
                # seed=0
            )

            # 将所有 convex hull 合并为一个 trimesh
            meshes = []

            for part in parts:
                vertices, faces = part   # ✅ 正确解包
                part_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces
                )

                meshes.append(part_mesh)

            scene = trimesh.Scene()
            np.random.seed(0)
            for p in meshes:
                p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
                scene.add_geometry(p)

            # 保存结果
            scene.export(output_path)

            print(f"Saved to: {output_path}")
    print("All files processed.")


if __name__ == "__main__":
    input_folder = "envs/agents/urdfs/dofbot_se/meshes/collision_ori"
    output_folder = "envs/agents/urdfs/dofbot_se/meshes/collision"

    process_stl_folder(input_folder, output_folder)