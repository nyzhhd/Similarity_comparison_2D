import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import shutil
import os
from sklearn.cluster import KMeans

def find_defects_by_comparison_sift(original_image,template_image):
    '''
    输入
    original_image: 原始保存图像
    template_image: 巡检拍摄图像
    threshold_feature:通过特征点个数比较方法阈值，阈值越大更容易检测出缺陷，阈值越小越不容易检测出缺陷，对于防尘套破损这样不容易发现的错误阈值应该设高
    threshold_ssim:通过图像相似性比较方法阈值，阈值越大更容易检测出缺陷，阈值越小越不容易检测出缺陷，对于防尘套破损这样不容易发现的错误阈值应该设高
    其中original_image和template_image要保证是一个地方做故障前和做故障后的图，图像可以发生倾斜

    输出
    have_defect: True代表可能存在有缺陷。False代表没有检测出差别。
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

    # 使用SIFT特征提取器
    # sift = cv2.SIFT_create()
    # # 提取特征点和描述符
    # keypoints1, descriptors1 = sift.detectAndCompute(original_image, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)
    # # 使用FLANN匹配器进行特征匹配
    # flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {})

    # #先看两幅图片有多少个特征匹配点
    # matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good_matches.append(m)

    # # 在原始图像中绘制匹配的特征
    # matched_image = cv2.drawMatches(original_image, keypoints1, template_image, keypoints2, good_matches, None)

    # #根据特征匹配来使得第二幅图像与第一幅图像对齐
    # if len(good_matches) >= 4:
    #     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
    #     # 应用变换矩阵以对齐第二幅图像
    #     image2_stabilized = cv2.warpAffine(template_image, M, (template_image.shape[1], template_image.shape[0]))
    # else:
    #     image2_stabilized = template_image


    # 计算相似性度量
    #灰度化
    gray_image1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    #直方图均衡化
    #gray_image1 = cv2.equalizeHist(gray_image1)
    #gray_image2 = cv2.equalizeHist(gray_image2)
    ssim = compare_ssim(gray_image1, gray_image2)



    # 显示结果图像
    # cv2.imshow("Matched Image Sift", matched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return ssim


# if __name__ == "__main__":
#     pass
#     import os

#     threshold=0.24
#     gk_folder = 'bolthead'
#     min_avg_simm = float('inf')
#     min_avg_simm_image = ''
#     i=0
#     avgs=0

#     for filename1 in os.listdir(gk_folder):
#         print ('----------------------{i}-----------------------')
#         reference_image_path = os.path.join(gk_folder, filename1)
#         reference_image = cv2.imread(reference_image_path)
#         reference_simm_list = []
#         for filename2 in os.listdir(gk_folder):
#             if filename2 != filename1:
#                 image_path = os.path.join(gk_folder, filename2)
#                 image = cv2.imread(image_path)
#                 try:
#                     simm = find_defects_by_comparison_sift(reference_image, image)
#                 except:
#                     simm = 0
#                 reference_simm_list.append(simm)
#         avg_simm = sum(reference_simm_list) / len(reference_simm_list)
#         print ('-----------',min(reference_simm_list))
#         print ('-----------',max(reference_simm_list))
#         print('-----------',avg_simm)
#         avgs=avg_simm+avgs
#         i=i+1
#         if avg_simm < threshold:
#             print(f"与其它图片比较相似度低于阈值的图片：{(reference_image_path)},相似度：{avg_simm}")
#         if avg_simm < min_avg_simm:
#             min_avg_simm = avg_simm
#             min_avg_simm_image = reference_image_path
#     print(f"最小平均相似度图片：{min_avg_simm_image},相似度：{min_avg_simm}")
#     print(avgs/i)


from skimage.feature import hog

from skimage import color


def compute_hog_features(image):
    # 将图像转换为灰度图像
    gray_image = color.rgb2gray(image)

    # 使用HOG计算图像的特征
    features, _ = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return features

if __name__ == "__main__":
    gk_folder = 'bolthead'
    output_folder_template = 'template_images2'
    output_folder_others = 'other_images2'
    num_template_images = 40  # 你想选择的模板图片数量

    if not os.path.exists(output_folder_template):
        os.makedirs(output_folder_template)

    if not os.path.exists(output_folder_others):
        os.makedirs(output_folder_others)

    images = []
    filenames = []

    for filename in os.listdir(gk_folder):
        print(filename)
        image_path = os.path.join(gk_folder, filename)

        # 读取图像
        image = cv2.imread(image_path)

        # 将图像调整为相同的大小
        image = cv2.resize(image, (100, 100))

        # 计算HOG特征
        features = compute_hog_features(image)

        images.append(features)
        filenames.append(filename)

    images = np.array(images)

    # 使用 KMeans 聚类
    kmeans = KMeans(n_clusters=num_template_images, random_state=0)
    kmeans.fit(images)

    # 获取每个簇的中心（代表性图像）
    representative_indices = kmeans.predict(images)

    # 复制代表性图片到 template_images 文件夹
    for index in representative_indices:
        filename = filenames[index]
        image_path = os.path.join(gk_folder, filename)
        print(f"Copying representative image {image_path} to {output_folder_template}")
        shutil.copy(image_path, os.path.join(output_folder_template, os.path.basename(image_path)))

    # 复制其他图片到 other_images 文件夹
    for i, filename in enumerate(os.listdir(gk_folder)):
        image_path = os.path.join(gk_folder, filename)

        if i not in representative_indices:
            print(f"Copying other image {image_path} to {output_folder_others}")
            shutil.copy(image_path, os.path.join(output_folder_others, filename))


# import cv2
# import os
# import shutil
# import numpy as np
# from sklearn.cluster import KMeans
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error as mse
# from skimage import exposure
#
#
# def extract_hog_features(image_path):
#     # 读取图像
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # 计算 HOG 特征
#     fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
#
#     # 使用直方图均衡化增强图像
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
#     # 确保 HOG 特征向量的长度是固定的
#     fd_fixed_length = np.zeros(987)  # 根据你之前报错的形状，将 987 替换为 HOG 特征向量的固定长度
#     fd_fixed_length[:len(fd)] = fd
#
#     return fd_fixed_length, hog_image_rescaled
#
#
# def calculate_ssim_mse(image1, image2):
#     # 计算 SSIM
#     ssim_value, _ = ssim(image1, image2, full=True)
#
#     # 计算 MSE
#     mse_value = mse(image1, image2)
#
#
#     return ssim_value, mse_value
#
#
# if __name__ == "__main__":
#     gk_folder = 'bolthead'
#     output_folder_template = 'template_images2'
#     output_folder_others = 'other_images2'
#     num_template_images = 40  # 你想选择的模板图片数量
#
#     if not os.path.exists(output_folder_template):
#         os.makedirs(output_folder_template)
#
#     if not os.path.exists(output_folder_others):
#         os.makedirs(output_folder_others)
#
#     images = []
#     filenames = []
#
#     for filename in os.listdir(gk_folder):
#         print(filename)
#         image_path = os.path.join(gk_folder, filename)
#
#         # 提取 HOG 特征
#         features, hog_image = extract_hog_features(image_path)
#
#         images.append(features)
#         filenames.append(filename)
#
#     images = np.array(images)
#
#     # 使用 KMeans 聚类
#     kmeans = KMeans(n_clusters=num_template_images, random_state=0)
#     kmeans.fit(images)
#
#     # 获取每个簇的中心（代表性图像）
#     representative_indices = kmeans.predict(images)
#
#     # 复制代表性图片到 template_images 文件夹
#     for index in representative_indices:
#         filename = filenames[index]
#         image_path = os.path.join(gk_folder, filename)
#         print(f"Copying representative image {image_path} to {output_folder_template}")
#         shutil.copy(image_path, os.path.join(output_folder_template, os.path.basename(image_path)))
#
#     # 复制其他图片到 other_images 文件夹，并计算 SSIM 和 MSE
#     for i, filename in enumerate(os.listdir(gk_folder)):
#         image_path = os.path.join(gk_folder, filename)
#
#         if i not in representative_indices:
#             print(f"Copying other image {image_path} to {output_folder_others}")
#             shutil.copy(image_path, os.path.join(output_folder_others, filename))
#
#             # 计算 SSIM 和 MSE
#             other_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             # 调整 HOG 特征图的大小以匹配 other_image
#             hog_image_rescaled = cv2.resize(hog_image, (other_image.shape[1], other_image.shape[0]))
#
#             # 计算 SSIM 和 MSE
#             ssim_value, mse_value = calculate_ssim_mse(hog_image_rescaled, other_image)
#
#             print(f"SSIM between representative image and {filename}: {ssim_value}")
#             print(f"MSE between representative image and {filename}: {mse_value}")
