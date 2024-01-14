# 结构相似性指数 (SSIM): 一种更先进的方法，用于测量两张图片的视觉结构、亮度和对比度的相似度。SSIM的值在0到1之间，值越接近1，表示图片越相似。对于需要高度视觉相似度的应用，SSIM可能是更好的选择。
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import time
import matplotlib.pyplot as plt
import os

def find_defects_by_comparison_sift(original_image,template_image):
    '''
    输入
    original_image: 巡检时候拍摄的图像，即待检测图像
    template_image: 模板库的模板图像
    输出
    ssim: original_image和template_image的相似度，值为0-1。
    '''
    # original_image = cv2.medianBlur(original_image, 3)
    # template_image = cv2.medianBlur(template_image, 3)
    # #调整亮度和对比度
    # original_image = cv2.convertScaleAbs(original_image, alpha=1.2, beta=20)
    # template_image = cv2.convertScaleAbs(template_image, alpha=1.2, beta=20)
    # 确保两张图片具有相同的尺寸
    if original_image.shape != template_image.shape:
        # 如果尺寸不同，将第二张图片调整为与第一张图片相同的尺寸
        template_image = cv2.resize(template_image, (original_image.shape[1], original_image.shape[0]))

    # # 使用SIFT特征提取器
    # sift = cv2.SIFT_create()
    # # 提取特征点和描述符
    # keypoints1, descriptors1 = sift.detectAndCompute(original_image, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)
    # # 使用FLANN匹配器进行特征匹配
    # flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {})
    #
    # #先看两幅图片有多少个特征匹配点
    # matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good_matches.append(m)
    #
    # # 在原始图像中绘制匹配的特征
    # matched_image = cv2.drawMatches(original_image, keypoints1, template_image, keypoints2, good_matches, None)
    #
    # #根据特征匹配来使得第二幅图像与第一幅图像对齐
    # if len(good_matches) >= 4:
    #     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
    #     # 应用变换矩阵以对齐第二幅图像
    #     image2_stabilized = cv2.warpAffine(template_image, M, (template_image.shape[1], template_image.shape[0]))
    # else:
    #     image2_stabilized = template_image

    # 灰度化
    gray_image1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    # gray_image1 = cv2.equalizeHist(gray_image1)
    # gray_image2 = cv2.equalizeHist(gray_image2)

    # 计算相似性度量

    ssim = compare_ssim(gray_image1, gray_image2)  #------------------------------------- 利用compare_ssim函数比较两个图的结构相似度

    # 显示结果图像
    # cv2.imshow("Matched Image Sift", matched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return ssim

def find_defects_by_comparison_Bolt(original_image, threshold, num_remove=1):
    have_defect = False
    gk_norm_folder = 'gk_mb_20231113'  # 模板库路径
    mse_list = []
    for filename in os.listdir(gk_norm_folder):
        image_norm_path = os.path.join(gk_norm_folder, filename)
        template_image = cv2.imread(image_norm_path)
        try:
            mse_sim = find_defects_by_comparison_sift(original_image, template_image)
        except:
            mse_sim = 0
        mse_list.append(mse_sim)

    # 排序并去除最高和最低的num_remove个值
    mse_list.sort()
    if len(mse_list) > 2 * num_remove:
        mse_list = mse_list[num_remove:-num_remove]
    mse_avg = sum(mse_list) / len(mse_list) if mse_list else 0

    mse_max = max(mse_list) if mse_list else 0
    print("最大相似度：", mse_max)
    print("平均相似度：", mse_avg)

    have_defect = mse_avg <= threshold
    return have_defect, mse_avg


def add_jitter(values, jitter_strength=0.1):
    # 为所有点添加随机偏移量
    jitter = jitter_strength * (np.random.rand(len(values)) - 0.5)
    return jitter

if __name__ == "__main__":
    TP = 0  # 真正例
    FP = 0  # 假正例
    TN = 0  # 真负例
    FN = 0  # 假负例
    threshold = 0.35
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

        if have_defect and real_defect:
            TP += 1
        elif have_defect and not real_defect:
            FP += 1
        elif not have_defect and not real_defect:
            TN += 1
        elif not have_defect and real_defect:
            FN += 1

    end_time = time.time()
    single_time = (end_time - start_time) / total_images
    print(f'\n+++++++++++++++++++SUM++++++++++++++++++++++++')
    print(gk_folder, f"文件夹检测出{defect_count}/{total_images}个图片有缺陷\n每张图片用时{single_time}s")
    # 计算TPR和FPR
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

    print(f"正确检出率（TPR）: {TPR:.2f}")
    print(f"误检率（FPR）: {FPR:.2f}")

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
    plt.xlabel('SSIM Similarity')
    plt.ylabel('Real Defect Presence (Jittered)')
    plt.title('Real Defect Detection Based on SSIM Similarity')
    plt.legend()
    plt.show()

        
