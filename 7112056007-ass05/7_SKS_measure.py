import cv2
import os
import numpy as np
import csv

# 計算每個channel的NPCR
def calculate_NPCR(origin_image, encrypt_image):
    NPCR = 0

    for i in range(origin_image.shape[0]):
        for j in range(origin_image.shape[1]):
            if origin_image[i, j] != encrypt_image[i, j]:
                NPCR = NPCR + 1

    NPCR = NPCR / (origin_image.shape[0] * origin_image.shape[1])

    return NPCR


# 計算彩色影像的NPCR
def NPCR_color(origin_image, encrypt_image):
    red = calculate_NPCR(origin_image[:, :, 2], encrypt_image[:, :, 2])
    green = calculate_NPCR(origin_image[:, :, 1], encrypt_image[:, :, 1])
    blue = calculate_NPCR(origin_image[:, :, 0], encrypt_image[:, :, 0])

    return red, green, blue

# 計算每個channel的UACI
def calculate_UACI(origin_image, encrypt_image):
    UACI = 0

    UACI = np.sum(np.abs(origin_image.astype(np.int16) - encrypt_image.astype(np.int16)) / 255)
    
    UACI = UACI / (origin_image.shape[0] * origin_image.shape[1])

    return UACI


# 計算彩色影像的UACI
def UACI_color(origin_image, encrypt_image):
    red = calculate_UACI(origin_image[:, :, 2], encrypt_image[:, :, 2])
    green = calculate_UACI(origin_image[:, :, 1], encrypt_image[:, :, 1])
    blue = calculate_UACI(origin_image[:, :, 0], encrypt_image[:, :, 0])

    return red, green, blue

# 原始圖片資料夾的路徑
origin_folder_path = "source"

# 原始圖片資料夾中所有圖片的名字
origin_image_files = os.listdir(origin_folder_path)

# 加密嵌密圖片資料夾的路徑
encrypt_folder_path = "encry"

# 加密嵌密圖片資料夾中所有圖片的名字
encrypt_image_files = os.listdir(encrypt_folder_path)

with open(f'statis/SKS_res.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    writer.writerow(['SKS', '', '', 'Cipher NPCR', '', '', 'Cipher UACI', '', ''])
    writer.writerow(['Image1', 'Image2', 'Type', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'])

    for i in range(len(origin_image_files)):
        # 構建完整的文件路徑
        encrypt_image_path_1 = os.path.join(encrypt_folder_path, origin_image_files[i][:-4] + '_enc' + origin_image_files[i][-4:]) 

        # 構建完整的文件路徑
        encrypt_image_path_2 = os.path.join(encrypt_folder_path, origin_image_files[i][:-4] + '_enc_SKS' + origin_image_files[i][-4:]) 

        # 使用OpenCV讀取圖像
        encrypt_image_1 = cv2.imread(encrypt_image_path_1, cv2.IMREAD_UNCHANGED)

        # 使用OpenCV讀取圖像
        encrypt_image_2 = cv2.imread(encrypt_image_path_2, cv2.IMREAD_UNCHANGED)

        if len(encrypt_image_1.shape) == 3:
            # 計算NPCR
            cipher_NPCR_red, cipher_NPCR_green, cipher_NPCR_blue = NPCR_color(encrypt_image_1, encrypt_image_2)

            # 計算UACI
            cipher_UACI_red, cipher_UACI_green, cipher_UACI_blue = UACI_color(encrypt_image_1, encrypt_image_2)

            # 寫入csv
            writer.writerow([origin_image_files[i][:-4] + '_enc' + origin_image_files[i][-4:], origin_image_files[i][:-4] + '_enc_SKS' + origin_image_files[i][-4:], 'color', str(cipher_NPCR_red), str(cipher_NPCR_green), str(cipher_NPCR_blue), str(cipher_UACI_red), str(cipher_UACI_green), str(cipher_UACI_blue)])
        else:
            # 計算NPCR
            cipher_NPCR = calculate_NPCR(encrypt_image_1, encrypt_image_2)

            # 計算UACI
            cipher_UACI = calculate_UACI(encrypt_image_1, encrypt_image_2)

            # 寫入csv
            writer.writerow([origin_image_files[i][:-4] + '_enc' + origin_image_files[i][-4:], origin_image_files[i][:-4] + '_enc_SKS' + origin_image_files[i][-4:], 'gray', str(cipher_NPCR), '', '', str(cipher_UACI), '', ''])

