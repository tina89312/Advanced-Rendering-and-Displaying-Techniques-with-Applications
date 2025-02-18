import cv2
import os
import numpy as np
import csv

# 計算每個channel的VOH
def calculate_VOH(image):
    # 計算影像的直方圖
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0,256])

    VOH = 0

    for i in range(len(histogram)):
        for j in range(len(histogram)):
            VOH = VOH + ((histogram[i] - histogram[j]) ** 2) / 2

    VOH = VOH / (256 ** 2)

    return VOH

# 計算VOH
def VOH(image):
    red = calculate_VOH(image[:, :, 2])
    green = calculate_VOH(image[:, :, 1])
    blue = calculate_VOH(image[:, :, 0])

    return red, green, blue

# 原始圖片資料夾的路徑
origin_folder_path = "origin"

# 原始圖片資料夾中所有圖片的名字
origin_image_files = os.listdir(origin_folder_path)

# 加密嵌密圖片資料夾的路徑
encrypt_folder_path = "encrp"

# 加密嵌密圖片資料夾中所有圖片的名字
encrypt_image_files = os.listdir(encrypt_folder_path)

with open(f'statis/VOH_res.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    writer.writerow(['VOH', '', 'Plain', '', '', 'Cipher', '', ''])
    writer.writerow(['Image', 'Type', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'])

    for i in range(len(origin_image_files)):
        # 構建完整的文件路徑
        origin_image_path = os.path.join(origin_folder_path, origin_image_files[i]) 

        # 使用OpenCV讀取圖像
        origin_image = cv2.imread(origin_image_path, cv2.IMREAD_UNCHANGED)

        # 計算VOH
        plain_red, plain_green, plain_blue = VOH(origin_image)

        # 構建完整的文件路徑
        encrypt_image_path = os.path.join(encrypt_folder_path, encrypt_image_files[i]) 

        # 使用OpenCV讀取圖像
        encrypt_image = cv2.imread(encrypt_image_path, cv2.IMREAD_UNCHANGED)

        # 計算VOH
        cipher_red, cipher_green, cipher_blue = VOH(encrypt_image)

        # 寫入csv
        writer.writerow([origin_image_files[i], 'color', str(plain_red), str(plain_green), str(plain_blue), str(cipher_red), str(cipher_green), str(cipher_blue)])
