import cv2
import os
import numpy as np
import csv
import math
import random

# 選擇block的index
def select_block_index(image_size):

    block_index = []
    
    for i in range(30):
        block_index.append(random.randint(int(image_size / 30) * i, (int(image_size / 30) * (i + 1)) - 1936))

    block_index = np.array(block_index)

    return block_index

# 計算每個block的information entropy
def calculate_information_entropy(image):
    # 計算影像的直方圖
    histogram, _ = np.histogram(image, bins=256, range=[0,256])

    information_entropy = 0

    for i in range(len(histogram)):
        if histogram[i] != 0:
            information_entropy = information_entropy - ((histogram[i] / image.shape[0]) * math.log(histogram[i] / image.shape[0], 2))

    return information_entropy

# 計算每個channel的LIN
def calculate_LIN(image):
    LIN = 0

    image_flatten = image.flatten()

    block_index = select_block_index(image_flatten.shape[0])

    for i in range(block_index.shape[0]):
        LIN = LIN + calculate_information_entropy(image_flatten[i:i+1936])

    LIN = LIN / 30

    return LIN

# 計算彩色影像的LIN
def LIN_color(image):
    red = calculate_LIN(image[:, :, 2])
    green = calculate_LIN(image[:, :, 1])
    blue = calculate_LIN(image[:, :, 0])

    return red, green, blue

# 原始圖片資料夾的路徑
origin_folder_path = "source"

# 原始圖片資料夾中所有圖片的名字
origin_image_files = os.listdir(origin_folder_path)

# 加密嵌密圖片資料夾的路徑
encrypt_folder_path = "encry"

# 加密嵌密圖片資料夾中所有圖片的名字
encrypt_image_files = os.listdir(encrypt_folder_path)

with open(f'statis/LIN_res.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    writer.writerow(['LIN (30, 1936, 0.05)', '', 'Plain', '', '', 'Cipher', '', ''])
    writer.writerow(['Image', 'Type', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'])

    for i in range(len(origin_image_files)):
        # 構建完整的文件路徑
        origin_image_path = os.path.join(origin_folder_path, origin_image_files[i]) 

        # 使用OpenCV讀取圖像
        origin_image = cv2.imread(origin_image_path, cv2.IMREAD_UNCHANGED)

        # 構建完整的文件路徑
        encrypt_image_path = os.path.join(encrypt_folder_path, origin_image_files[i][:-4] + '_enc' + origin_image_files[i][-4:]) 

        # 使用OpenCV讀取圖像
        encrypt_image = cv2.imread(encrypt_image_path, cv2.IMREAD_UNCHANGED)

        if len(origin_image.shape) == 3:
            # 計算LIN
            plain_red, plain_green, plain_blue = LIN_color(origin_image)

            # 計算LIN
            cipher_red, cipher_green, cipher_blue = LIN_color(encrypt_image)

            # 寫入csv
            writer.writerow([origin_image_files[i], 'color', str(plain_red), str(plain_green), str(plain_blue), str(cipher_red), str(cipher_green), str(cipher_blue)])
        else:
            # 計算LIN
            plain = calculate_LIN(origin_image)

            # 計算LIN
            cipher = calculate_LIN(encrypt_image)

            # 寫入csv
            writer.writerow([origin_image_files[i], 'gray', str(plain), '', '', str(cipher), '', ''])

