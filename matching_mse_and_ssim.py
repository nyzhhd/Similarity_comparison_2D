import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import time
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import joblib


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
    # 确保相似度在0到1之间
    similarity = max(0, min(similarity, 1))
    return similarity

def find_defects_by_comparison_sift_mse(original_image,template_image):
    '''
    输入
    original_image: 巡检时候拍摄的图像，即待检测图像
    template_image: 模板库的模板图像
    输出
    ssim: original_image和template_image的相似度，值为0-1。
    '''
    # 确保两张图片具有相同的尺寸
    if original_image.shape != template_image.shape:
        # 如果尺寸不同，将第二张图片调整为与第一张图片相同的尺寸
        template_image = cv2.resize(template_image, (original_image.shape[1], original_image.shape[0]))

    # 计算相似性度量
    mse_sim= mse(original_image, template_image)  #------------------------------------- 利用mse函数比较两个图的结构相似度
    return mse_sim

def find_defects_by_comparison_sift_ssim(original_image,template_image):
    '''
    输入
    original_image: 巡检时候拍摄的图像，即待检测图像
    template_image: 模板库的模板图像
    输出
    ssim: original_image和template_image的相似度，值为0-1。
    '''
    # 确保两张图片具有相同的尺寸
    if original_image.shape != template_image.shape:
        # 如果尺寸不同，将第二张图片调整为与第一张图片相同的尺寸
        template_image = cv2.resize(template_image, (original_image.shape[1], original_image.shape[0]))


    # 灰度化
    gray_image1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # 计算相似性度量

    ssim = compare_ssim(gray_image1, gray_image2)  #------------------------------------- 利用compare_ssim函数比较两个图的结构相似度

    # 显示结果图像
    # cv2.imshow("Matched Image Sift", matched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return ssim

def find_defects_by_comparison_Bolt_mse(original_image, threshold):
    have_defect = False
    gk_norm_folder = 'template_images'  # 模板库：存放正常的模板
    mse_list = []

    max_mse = 0
    max_mse_filename = ""

    for filename in os.listdir(gk_norm_folder):
        image_norm_path = os.path.join(gk_norm_folder, filename)
        template_image = cv2.imread(image_norm_path)

        try:
            mse_sim = find_defects_by_comparison_sift_mse(original_image, template_image)
        except:
            mse_sim = 0

        mse_list.append(mse_sim)

        if mse_sim > max_mse:
            max_mse = mse_sim
            max_mse_filename = filename

    mse_avg = sum(mse_list) / len(mse_list)
    print("mse最大相似度：", max_mse)
    print("mse平均相似度：", mse_avg)
    print(f"mse为{max_mse}来自{max_mse_filename} ")

    if max_mse > threshold:
        #print(f"{max_mse_filename} 未检测到缺陷")
        have_defect = False
    else:
        #print(f"{max_mse_filename} 检测到缺陷")
        have_defect = True

    return have_defect, max_mse



def find_defects_by_comparison_Bolt_ssim(original_image, threshold):
    have_defect = False
    gk_norm_folder = 'template_images'  # 模板库：存放正常的模板
    mse_list = []

    max_ssim = 0
    max_ssim_filename = ""

    for filename in os.listdir(gk_norm_folder):
        image_norm_path = os.path.join(gk_norm_folder, filename)
        template_image = cv2.imread(image_norm_path)

        try:
            ssim = find_defects_by_comparison_sift_ssim(original_image, template_image)
        except:
            ssim = 0

        mse_list.append(ssim)

        if ssim > max_ssim:
            max_ssim = ssim
            max_ssim_filename = filename

    ssim_avg = sum(mse_list) / len(mse_list)
    print("ssim最大相似度：", max_ssim)
    print("ssim平均相似度：", ssim_avg)
    print(f"ssim为{max_ssim}来自{max_ssim_filename} ")

    if max_ssim > threshold:
        #print(f"{max_ssim_filename} 未检测到缺陷")
        have_defect = False
    else:
        #print(f"{max_ssim_filename} 检测到缺陷")
        have_defect = True

    return have_defect, max_ssim




if __name__ == "__main__":
    threshold = 0.522
    gk_folder = 'other_images'
    mse_data = []  # MSE相似度数据
    ssim_data = [] # SSIM相似度数据
    defect_labels = []  # 标记是否有缺陷

    for filename in os.listdir(gk_folder):
        print(f'----------------{filename}------------------')  
        image_path = os.path.join(gk_folder, filename)
        original_image = cv2.imread(image_path)
        _, mse_sim = find_defects_by_comparison_Bolt_mse(original_image, threshold)
        _, ssim = find_defects_by_comparison_Bolt_ssim(original_image, threshold)

        real_defect = filename.startswith("F")
        defect_labels.append(real_defect)
        mse_data.append(mse_sim)
        ssim_data.append(ssim)

    # 准备数据
    X = np.column_stack((mse_data, ssim_data))
    y = np.array(defect_labels)

    # 创建SVM模型和参数网格
    svc = SVC(probability=True, kernel='rbf')
    parameters = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': [1, 0.1, 0.01, 0.001]
    }
    # model = make_pipeline(StandardScaler(), svc)
    #
    # # 进行网格搜索
    # clf = GridSearchCV(model, parameters, cv=5)
    # clf.fit(X, y)
    #
    # # 获取最佳参数并重新训练模型
    # best_model = clf.best_estimator_
    # print("Best parameters found: ", clf.best_params_)

    ## 保存模型到文件
    #joblib.dump(best_model, 'best_model.joblib')
    best_model = joblib.load('best_model.joblib')



    # 使用最佳模型进行预测
    y_pred = best_model.predict(X)

    # 计算评估指标
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # 打印评估指标
    print("准确率（Accuracy）: {:.2f}".format(accuracy))
    print("精确度（Precision）: {:.2f}".format(precision))
    print("召回率（Recall）: {:.2f}".format(recall))
    print("F1 分数（F1 Score）: {:.2f}".format(f1))

    # 输出每个样本的实际类别和预测类别
    result_df = pd.DataFrame({'file':os.listdir(gk_folder),'Actual Defect': y, 'Predicted Defect': y_pred})
    result_df.to_csv('svm_classification_results.csv', index=False)

    # 生成预测网格
    xx, yy = np.meshgrid(np.linspace(min(mse_data), max(mse_data), 100),
                         np.linspace(min(ssim_data), max(ssim_data), 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = best_model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # 绘制散点图和分类线
    plt.figure(figsize=(10, 6))
    colors = ['red' if label else 'green' for label in defect_labels]
    plt.scatter(mse_data, ssim_data, color=colors)
    plt.contour(xx, yy, probs, levels=[0.5], cmap="Greys_r")
    plt.xlabel('MSE Similarity')
    plt.ylabel('SSIM Similarity')
    plt.title('Optimized Defect Detection Based on MSE and SSIM Similarity')
    plt.legend(['Real Defect', 'No Real Defect'])
    plt.show()

    # print(len(defect_labels==True))


    # # 加载保存的模型
    # loaded_model = joblib.load('best_model.joblib')
    # # 有一张新图像
    # new_image = cv2.imread(r'D:\adavance\resnet50\datasets\WholeCotterPin_and_Clip_and_SlopeSingleBolthead\train\norm_SlopeSingleBolthead\0004d114.jpg')
    # new_image = cv2.resize(new_image, (100, 100))  # 将图像调整为相同的大小
    #
    # # 计算新图像的特征
    # _,new_mse= find_defects_by_comparison_Bolt_mse(new_image, threshold)
    # _,new_ssim=find_defects_by_comparison_Bolt_ssim( new_image, threshold)
    #
    # # 创建新数据点
    # new_data_point = np.array([[new_mse, new_ssim]])
    #
    # # 使用保存的模型进行预测
    # prediction = loaded_model.predict(new_data_point)
    #
    # print("Prediction for the new image:", prediction)