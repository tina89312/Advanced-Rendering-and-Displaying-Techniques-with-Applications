import cv2
import os
import numpy as np
import csv

# 計算每個channel的CHI
def calculate_CHI(image):
    # 計算影像的直方圖
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0,256])

    CHI = 0

    for i in range(len(histogram)):
        CHI = CHI + (((histogram[i] - (image.flatten().shape[0] / 256)) ** 2) / (image.flatten().shape[0] / 256))

    return CHI

# 計算CHI
def CHI(image):
    red = calculate_CHI(image[:, :, 2])
    green = calculate_CHI(image[:, :, 1])
    blue = calculate_CHI(image[:, :, 0])

    red_result = 'Success' if red < 293.248 else 'Fail'
    green_result = 'Success' if green < 293.248 else 'Fail'
    blue_result = 'Success' if blue < 293.248 else 'Fail'

    return red, green, blue, red_result, green_result, blue_result

# 原始圖片資料夾的路徑
origin_folder_path = "origin"

# 原始圖片資料夾中所有圖片的名字
origin_image_files = os.listdir(origin_folder_path)

# 加密嵌密圖片資料夾的路徑
encrypt_folder_path = "encrp"

# 加密嵌密圖片資料夾中所有圖片的名字
encrypt_image_files = os.listdir(encrypt_folder_path)

with open(f'statis/CHI_res.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    writer.writerow(['CHI', '', 'Cipher', '', '', '', '', 'Results', '', ''])
    writer.writerow(['Image', 'Type', 'Red', 'Green', 'Blue', 'alpha', 'chi value', 'Red', 'Green', 'Blue'])

    for i in range(len(encrypt_image_files)):
        # 構建完整的文件路徑
        encrypt_image_path = os.path.join(encrypt_folder_path, encrypt_image_files[i]) 

        # 使用OpenCV讀取圖像
        encrypt_image = cv2.imread(encrypt_image_path, cv2.IMREAD_UNCHANGED)

        # 計算CHI
        cipher_red, cipher_green, cipher_blue, red_result, green_result, blue_result = CHI(encrypt_image)

        # 寫入csv
        writer.writerow([origin_image_files[i], 'color', str(cipher_red), str(cipher_green), str(cipher_blue), '0.05', '293.248', red_result, green_result, blue_result])