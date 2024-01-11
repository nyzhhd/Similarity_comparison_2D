#均方误差 (MSE): 一种简单直观的方法，通过计算两张图片对应像素之间的平方差的平均值来衡量它们的差异。MSE值越低，表明图片越相似。
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import time
import matplotlib.pyplot as plt
import os


def mse(imageA, imageB):
    '''
    计算两个图像之间的规范化均方误差。
    输入
    imageA: 第一个图像
    imageB: 第二个图像
    输出
    similarity: 规范化的均方误差，作为相似度度量，范围在0到1之间。
    '''
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    max_err = 255.0 ** 2
    similarity = 1 - (err / max_err)
    return similarity

def find_defects_by_comparison_sift(original_image,template_image):
    '''
    输入
    original_image: 巡检时候拍摄的图像，即待检测图像
    template_image: 模板库的模板图像
    输出
    ssim: original_image和template_image的相似度，值为0-1。
    '''
    original_image = cv2.medianBlur(original_image, 3)
    template_image = cv2.medianBlur(template_image, 3)
    #调整亮度和对比度
    original_image = cv2.convertScaleAbs(original_image, alpha=1.2, beta=20)
    template_image = cv2.convertScaleAbs(template_image, alpha=1.2, beta=20)
    # 确保两张图片具有相同的尺寸
    if original_image.shape != template_image.shape:
        # 如果尺寸不同，将第二张图片调整为与第一张图片相同的尺寸
        template_image = cv2.resize(template_image, (original_image.shape[1], original_image.shape[0]))

    # 使用SIFT特征提取器
    sift = cv2.SIFT_create()
    # 提取特征点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(original_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)
    # 使用FLANN匹配器进行特征匹配
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {})

    #先看两幅图片有多少个特征匹配点
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 在原始图像中绘制匹配的特征
    matched_image = cv2.drawMatches(original_image, keypoints1, template_image, keypoints2, good_matches, None)

    #根据特征匹配来使得第二幅图像与第一幅图像对齐
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        # 应用变换矩阵以对齐第二幅图像
        image2_stabilized = cv2.warpAffine(template_image, M, (template_image.shape[1], template_image.shape[0]))
    else:
        image2_stabilized = template_image

    # 灰度化
    gray_image1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2_stabilized, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    # gray_image1 = cv2.equalizeHist(gray_image1)
    # gray_image2 = cv2.equalizeHist(gray_image2)

    # 计算相似性度量
    mse_sim= mse(gray_image1, gray_image2)  #------------------------------------- 利用mse函数比较两个图的结构相似度

    # 显示结果图像
    # cv2.imshow("Matched Image Sift", matched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return mse_sim

def find_defects_by_comparison_Bolt(original_image,threshold):
        have_defect = False
        gk_norm_folder = 'gk_mb_20231113'        #-----------------------------------------------#模板库 ： 存放正常的模板
        mse_list=[]
        for filename in os.listdir(gk_norm_folder):
            image_norm_path = os.path.join(gk_norm_folder, filename)
            template_image = cv2.imread(image_norm_path)
            try:
               mse_sim = find_defects_by_comparison_sift(original_image, template_image)
            except:
                mse_sim =0
            mse_list.append(mse_sim )
        mse_avg = sum(mse_list) / len(mse_list)
        mse_max = max(mse_list)
        print("最大相似度：",mse_max)
        print("平均相似度：",mse_avg)

        if mse_avg>threshold:
            #print(f"{filename} 未检测到缺陷")
            have_defect=False
        else:
            #print(f"{filename} 检测到缺陷")
            have_defect=True
        return have_defect,mse_avg


def add_jitter(values, jitter_strength=0.1):
    # 为垂直坐标添加随机偏移量
    jitter = jitter_strength * (np.random.rand(len(values)) - 0.5)
    return np.array(values) + jitter

if __name__ == "__main__":
    threshold = 0.522
    gk_folder = 'gk'  # 需要检测图像的文件夹
    defect_data = []  # 收集有缺陷的图像的相似度数据
    no_defect_data = []  # 收集无缺陷的图像的相似度数据

    start_time = time.time()
    total_images = 0
    defect_count = 0

    for filename in os.listdir(gk_folder):
        print(f'+++++++++++++++++{total_images}+++++++++++++++++++++++')
        image_path = os.path.join(gk_folder, filename)
        original_image = cv2.imread(image_path)
        have_defect, mse_sim = find_defects_by_comparison_Bolt(original_image, threshold)
        total_images += 1

        # 根据文件名判断图像是否真实有缺陷
        real_defect = filename.startswith("F")

        # 收集数据
        if real_defect:
            defect_data.append(mse_sim)
        else:
            no_defect_data.append(mse_sim)

        # 打印检测结果
        if have_defect:
            defect_count += 1
            print(f"{image_path} 检测到缺陷")
        else:
            print(f"{image_path} 未检测到缺陷")

    end_time = time.time()
    single_time = (end_time - start_time) / total_images
    print(gk_folder, f"检测出{defect_count}/{total_images}个图片有缺陷，每张图片用时{single_time}s")

    # 添加轻微的随机偏移量
    defect_labels_jittered = add_jitter([1] * len(defect_data))
    no_defect_labels_jittered = add_jitter([0] * len(no_defect_data))

    # 绘制散点图
    plt.figure(figsize=(10, 6))  # 调整图形大小
    defect_jitter = add_jitter([1] * len(defect_data))
    no_defect_jitter = add_jitter([0] * len(no_defect_data))

    # 绘制所有点
    plt.scatter(defect_data, defect_jitter, color='red', label='Real Defect')
    plt.scatter(no_defect_data, no_defect_jitter, color='green', label='No Real Defect')

    # 标记代表性点的标签
    # 例如，选择每种类别中MSE最高和最低的点
    if defect_data:
        plt.text(defect_data[0], defect_jitter[0], f'{defect_data[0]:.2f}', fontsize=8, ha='right')
        plt.text(defect_data[-1], defect_jitter[-1], f'{defect_data[-1]:.2f}', fontsize=8, ha='right')

    if no_defect_data:
        plt.text(no_defect_data[0], no_defect_jitter[0], f'{no_defect_data[0]:.2f}', fontsize=8, ha='right')
        plt.text(no_defect_data[-1], no_defect_jitter[-1], f'{no_defect_data[-1]:.2f}', fontsize=8, ha='right')

    plt.axvline(x=threshold, color='blue', linestyle='--', label='Threshold')
    plt.xlabel('MSE Similarity')
    plt.ylabel('Real Defect Presence (Jittered)')
    plt.title('Real Defect Detection Based on MSE Similarity')
    plt.legend()
    plt.show()
        
