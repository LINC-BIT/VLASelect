import cv2
import numpy as np
import os
import glob
from PIL import Image

# ================= 配置区域 =================
# 棋盘格的内部角点数量 (列数, 行数)
# 注意：这是指黑白块交界的角点数量，不是方块的数量
# 例如：如果你打印的是 8x6 的方块，那么角点通常是 7x5 或 9x6，请根据实际调整
CHECKERBOARD = (7, 10) 

# 图像文件夹路径 (支持 jpg, png)
IMAGE_PATH = "images/*.jpg" 
# ===========================================
def get_images(num=15):
    camera = cv2.VideoCapture(0,cv2.CAP_V4L2)
    path = './images'
    i = 0
    while i < num:
        ret, frame = camera.read()
        save_path = os.path.join(path, f'{i}.jpg')
        cv2.imshow('show', frame)
        if cv2.waitKey(2) & 0xFF == ord('y'):
            Image.fromarray(frame).save(save_path)
            i += 1

def main():
    # 定义世界坐标系中的点 (例如 z=0 平面上的点)
    # 这里假设每个方格的物理尺寸为 1个单位 (例如 25mm)，也可以设为实际毫米数
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # 用于存储所有图像的世界坐标点和图像坐标点
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # 获取图片列表
    images = glob.glob(IMAGE_PATH)
    
    if not images:
        print(f"错误: 在路径 '{IMAGE_PATH}' 下未找到图片！")
        return

    print(f"找到 {len(images)} 张图片，开始处理...")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        # cv2.CALIB_CB_ADAPTIVE_THRESH 适应光照变化
        # cv2.CALIB_CB_FAST_CHECK 快速检查
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

        # 如果找到了角点
        if ret == True:
            objpoints.append(objp)

            # 亚像素级角点优化，提高精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            imgpoints.append(corners_subpix)

            # 可视化检测结果 (可选，方便调试)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners_subpix, ret)
            # 缩放图片以便显示
            h, w = img.shape[:2]
            display_img = cv2.resize(img, (int(w/2), int(h/2)))
            cv2.imshow('Detected Corners', display_img)
            cv2.waitKey(500) # 暂停0.5秒查看效果
        else:
            print(f"警告: 在图片 {fname} 中未检测到棋盘格。")

    cv2.destroyAllWindows()

    # ================= 执行标定 =================
    if len(objpoints) > 0:
        print("\n正在计算相机参数...")
        
        # calibrateCamera 返回重投影误差、相机矩阵、畸变系数、旋转向量和平移向量
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("---------- 标定结果 ----------")
        print(f"平均重投影误差: {ret}")
        print(f"\n相机内参矩阵 (Camera Matrix):\n{mtx}")
        print(f"\n畸变系数 (Distortion Coefficients):\n{dist.flatten()}")
        
        # 获取图像尺寸
        h, w = img.shape[:2]
        
        # ================= 去畸变演示 =================
        # 计算新的相机矩阵，用于裁剪掉黑边
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # 选择一张图进行去畸变展示
        test_img_path = images[0]
        img = cv2.imread(test_img_path)
        
        # 应用去畸变
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # 裁剪多余的黑边
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        cv2.imwrite('undistorted_result.jpg', dst)
        print(f"\n去畸变后的图片已保存为 'undistorted_result.jpg'")
        
        # 显示对比 (原图 vs 去畸变)
        cv2.imshow('Original', cv2.resize(img, (640, 480)))
        cv2.imshow('Undistorted', cv2.resize(dst, (640, 480)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存参数到文件 (可选)
        np.savez("camera_params.npz", mtx=mtx, dist=dist, newcameramtx=newcameramtx)
        print("参数已保存到 'camera_params.npz'")

    else:
        print("错误: 没有检测到任何有效的棋盘格，标定失败。")

if __name__ == "__main__":
    get_images()
    main()