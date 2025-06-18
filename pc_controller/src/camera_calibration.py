#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os
import sys
import time
import json


class ManualCalibration:
    def __init__(self):
        # 棋盘格内角点尺寸
        self.pattern_size = (8, 5)  # 内角点数量 (宽度, 高度)
        self.square_size = 2.72  # 方格尺寸（厘米）

        # 标定相关变量
        self.objpoints = []  # 3D点坐标
        self.imgpoints = []  # 2D点坐标
        self.image_size = None  # 图像尺寸
        self.images_marked = []  # 已经标记的图像路径

        # 用于保存手动标记的点
        self.current_points = []
        self.current_image = None
        self.current_image_path = None
        self.point_counter = 0
        self.window_name = "手动标记角点"

        # 图像路径
        self.image_path = r"E:\Projects\LLM-Robotic\calibration_image"

        # 保存路径
        self.save_folder = "calibration_results"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # 保存手动标记数据的文件
        self.manual_points_file = os.path.join(self.save_folder, "manual_points.json")

        # 图像增强设置
        self.enable_enhancement = True  # 是否启用图像增强
        self.enhancement_method = 'sharpening'  # 默认使用锐化方法

        # 如果之前有标记，加载它们
        self.load_saved_points()

    def enhance_image(self, image):
        """增强图像，提高边缘清晰度"""
        if not self.enable_enhancement:
            return image

        enhanced = image.copy()

        if self.enhancement_method == 'sharpening':
            # 锐化卷积核
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        elif self.enhancement_method == 'canny':
            # 通过Canny边缘增强
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            edges = cv2.Canny(gray, 50, 150)

            # 将边缘叠加到原图上
            if len(enhanced.shape) == 3:
                # 彩色图像情况下
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                enhanced = cv2.addWeighted(enhanced, 0.7, edges_colored, 0.3, 0)
            else:
                # 灰度图像
                enhanced = cv2.addWeighted(enhanced, 0.7, edges, 0.3, 0)

        return enhanced

    def toggle_enhancement(self):
        """切换不同的图像增强方法"""
        # 简化为主要在Sharpening和Canny之间切换
        methods = ['sharpening', 'canny', 'none']
        current_index = methods.index(self.enhancement_method) if self.enhancement_method in methods else 0
        next_index = (current_index + 1) % len(methods)
        self.enhancement_method = methods[next_index]

        if self.enhancement_method == 'none':
            self.enable_enhancement = False
        else:
            self.enable_enhancement = True

        print(f"图像增强方法已切换为: {self.enhancement_method}")
        return self.enhancement_method

    def load_saved_points(self):
        """加载之前保存的手动标记点"""
        if os.path.exists(self.manual_points_file):
            try:
                with open(self.manual_points_file, 'r') as f:
                    data = json.load(f)
                    # 将列表转回NumPy数组
                    self.imgpoints = [np.array(points, dtype=np.float32) for points in data.get('imgpoints', [])]
                    self.images_marked = data.get('images_marked', [])

                    # 根据已标记的图像重建对象点
                    self.objpoints = []
                    for _ in range(len(self.imgpoints)):
                        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
                        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
                        objp = objp * self.square_size
                        self.objpoints.append(objp)

                    # 尝试获取图像尺寸
                    if self.image_size is None and self.images_marked:
                        for img_path in self.images_marked:
                            if os.path.exists(img_path):
                                img = cv2.imread(img_path)
                                if img is not None:
                                    self.image_size = (img.shape[1], img.shape[0])
                                    break

                    print(f"已加载 {len(self.imgpoints)} 张已标记图像")
            except Exception as e:
                print(f"加载保存的标记点时出错: {e}")

    def save_points(self):
        """保存当前标记的所有点"""
        # 将NumPy数组转换为列表以便JSON序列化
        imgpoints_list = []
        for points in self.imgpoints:
            imgpoints_list.append(points.tolist())  # 转换为Python列表

        data = {
            'imgpoints': imgpoints_list,
            'images_marked': self.images_marked
        }

        with open(self.manual_points_file, 'w') as f:
            json.dump(data, f)
        print(f"标记数据已保存到 {self.manual_points_file}")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于手动标记角点"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 计算当前应标记的点所在的行和列
            total_points = self.pattern_size[0] * self.pattern_size[1]
            if self.point_counter < total_points:
                # 添加点
                self.current_points.append([float(x), float(y)])
                self.point_counter += 1

                # 在图像上绘制点，不显示序号
                cv2.circle(self.current_image, (x, y), 1, (0, 0, 255), -1)  # 点的大小改为1
                cv2.imshow(self.window_name, self.current_image)

                # 计算当前点的行列位置（用于指导用户）
                row = (self.point_counter - 1) // self.pattern_size[0]
                col = (self.point_counter - 1) % self.pattern_size[0]
                print(f"已标记点 {self.point_counter}/{total_points} (行 {row + 1}, 列 {col + 1})")

                # 如果所有点都标记完了
                if self.point_counter == total_points:
                    print("所有点标记完成！按 'S' 保存, 'R' 重新开始, 'Q' 放弃并继续")

    def mark_image(self, image_path):
        """手动标记图像中的角点"""
        self.current_image_path = image_path
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return False

        # 获取图像尺寸
        if self.image_size is None:
            self.image_size = (img.shape[1], img.shape[0])

        # 应用图像增强
        enhanced_img = self.enhance_image(img)

        # 保存原始图像的副本
        self.current_image = enhanced_img.copy()
        self.original_image = img.copy()  # 保存未增强的原始图像
        self.current_points = []
        self.point_counter = 0

        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # 显示图像并指导
        cv2.imshow(self.window_name, self.current_image)
        print("\n" + "=" * 50)
        print(f"开始标记图像: {os.path.basename(image_path)}")
        print(f"请按照从左到右、从上到下的顺序标记 {self.pattern_size[0]}x{self.pattern_size[1]} 个内角点")
        print("按 'S' 保存标记, 'R' 重新开始, 'Q' 放弃并继续, 'ESC' 退出程序")
        print(f"按 'E' 切换图像增强方法: [当前: {self.enhancement_method}] -> 在锐化/边缘检测/无增强间切换")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC键退出整个程序
                cv2.destroyWindow(self.window_name)
                return None

            elif key == ord('s') or key == ord('S'):  # 保存当前标记
                total_points = self.pattern_size[0] * self.pattern_size[1]
                if self.point_counter == total_points:
                    # 保存点
                    self.imgpoints.append(np.array(self.current_points, dtype=np.float32))
                    self.images_marked.append(image_path)

                    # 创建对象点
                    objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
                    objp = objp * self.square_size
                    self.objpoints.append(objp)

                    print(f"已保存当前图像的标记点，总共已标记 {len(self.imgpoints)} 张图像")
                    # 保存所有点
                    self.save_points()
                    cv2.destroyWindow(self.window_name)  # 只关闭当前窗口，不完全退出
                    return True
                else:
                    print(f"尚未标记完所有点! 当前: {self.point_counter}/{total_points}")

            elif key == ord('r') or key == ord('R'):  # 重新开始当前图像
                enhanced_img = self.enhance_image(self.original_image.copy())
                self.current_image = enhanced_img.copy()
                self.current_points = []
                self.point_counter = 0
                cv2.imshow(self.window_name, self.current_image)
                print("已重置当前图像的标记")

            elif key == ord('e') or key == ord('E'):  # 切换图像增强方法
                method = self.toggle_enhancement()
                enhanced_img = self.enhance_image(self.original_image.copy())

                # 保存当前已标记的点
                temp_points = self.current_points.copy()
                temp_counter = self.point_counter

                # 应用新的增强方法，但保留已标记的点
                self.current_image = enhanced_img.copy()

                # 重新绘制已标记的点
                for i, (x, y) in enumerate(temp_points):
                    cv2.circle(self.current_image, (int(x), int(y)), 1, (0, 0, 255), -1)

                self.current_points = temp_points
                self.point_counter = temp_counter

                cv2.imshow(self.window_name, self.current_image)
                print(f"已切换图像增强方法为: {method}，保留已标记的点")

            elif key == ord('q') or key == ord('Q'):  # 放弃当前图像
                cv2.destroyWindow(self.window_name)  # 只关闭当前窗口，不完全退出
                return False

    def calibrate_camera(self):
        """使用标记点标定相机"""
        if len(self.imgpoints) < 3:
            print("至少需要3张已标记图像才能进行标定")
            return False

        # 确保图像尺寸已正确设置
        if self.image_size is None or any(s <= 0 for s in self.image_size):
            print("错误：图像尺寸无效，尝试重新检测图像尺寸")
            # 尝试从已标记图像中获取有效尺寸
            for img_path in self.images_marked:
                img = cv2.imread(img_path)
                if img is not None:
                    self.image_size = (img.shape[1], img.shape[0])
                    print(f"已重新检测图像尺寸: {self.image_size}")
                    break

            # 如果仍然无法获取有效尺寸，则退出
            if self.image_size is None or any(s <= 0 for s in self.image_size):
                print("无法获取有效的图像尺寸，标定失败")
                return False

        print("\n" + "=" * 50)
        print(f"开始用 {len(self.imgpoints)} 张图像进行相机标定...")
        print(f"使用图像尺寸: {self.image_size}")

        # 执行相机标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None)

        if not ret:
            print("标定失败!")
            return False

        # 优化相机矩阵
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, self.image_size, 1, self.image_size)

        # 保存标定结果
        calibration_result = {
            'camera_matrix': mtx.tolist(),
            'dist_coeffs': dist.tolist(),
            'optimal_camera_matrix': newcameramtx.tolist(),
            'roi': roi,
            'image_size': self.image_size
        }

        result_file = os.path.join(self.save_folder, "camera_params.json")
        with open(result_file, 'w') as f:
            json.dump(calibration_result, f, indent=4)

        # 计算重投影误差
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            # 确保数据类型匹配
            imgpoints2 = imgpoints2.reshape(-1, 2)
            imgpoints1 = self.imgpoints[i].reshape(-1, 2)
            error = cv2.norm(imgpoints1, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error = mean_error / len(self.objpoints)
        print(f"标定完成! 重投影误差: {mean_error}")
        print(f"相机参数已保存到 {result_file}")

        # 检查标定质量
        self.check_calibration(mtx, dist, newcameramtx, roi)

        return True

    def check_calibration(self, camera_matrix, dist_coeffs, new_camera_matrix=None, roi=None):
        """检查标定质量，展示未失真图像"""
        if not self.images_marked:
            return

        print("\n" + "=" * 50)
        print("正在生成未失真图像用于检查标定质量...")

        # 创建保存文件夹
        undistort_folder = os.path.join(self.save_folder, "undistorted")
        if not os.path.exists(undistort_folder):
            os.makedirs(undistort_folder)

        # 为每张已标记的图像生成未失真版本
        for img_path in self.images_marked:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 方法1：使用优化的相机矩阵
            if new_camera_matrix is not None:
                dst1 = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

                # 剪裁图像
                if roi and all(v > 0 for v in roi):
                    x, y, w, h = roi
                    dst1 = dst1[y:y + h, x:x + w]

                out_file1 = os.path.join(undistort_folder, f"undistorted_optimal_{os.path.basename(img_path)}")
                cv2.imwrite(out_file1, dst1)

            # 方法2：使用原始相机矩阵
            dst2 = cv2.undistort(img, camera_matrix, dist_coeffs)
            out_file2 = os.path.join(undistort_folder, f"undistorted_regular_{os.path.basename(img_path)}")
            cv2.imwrite(out_file2, dst2)

            # 方法3：使用重映射
            h, w = img.shape[:2]
            mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), 5)
            dst3 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            out_file3 = os.path.join(undistort_folder, f"undistorted_remap_{os.path.basename(img_path)}")
            cv2.imwrite(out_file3, dst3)

        print(f"未失真图像已保存到 {undistort_folder}")
        print("生成了3种不同的去畸变结果，可比较哪种效果最好")

    def run(self):
        """运行手动标定流程"""
        # 寻找标定图像，仅查找png格式
        images = sorted(glob.glob(os.path.join(self.image_path, '*.png')))
        if not images:
            print(f"在 {self.image_path} 目录中没有找到PNG图像")
            return

        print(f"找到 {len(images)} 张标定图像")

        # 过滤掉已经标记的图像
        images_to_mark = [img for img in images if img not in self.images_marked]
        print(f"其中 {len(self.images_marked)} 张已标记, {len(images_to_mark)} 张待标记")

        # 对每个图像进行手动标记
        for img_path in images_to_mark:
            result = self.mark_image(img_path)
            if result is None:  # 用户选择退出
                break

        # 如果有足够图像，进行标定
        if len(self.imgpoints) >= 3:
            self.calibrate_camera()
        else:
            print("标记的图像不足，需要至少3张图像才能进行标定")


if __name__ == "__main__":
    calibrator = ManualCalibration()
    calibrator.run() 