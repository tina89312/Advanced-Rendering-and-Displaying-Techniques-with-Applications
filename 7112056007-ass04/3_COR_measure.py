import cv2
import os
import numpy as np
import csv
import math
import random

# 抽取樣本
def sampling(image, sample_size):
    image_faltten = image.flatten()
    l = math.floor(len(image_faltten) / sample_size)

    samples_index = []

    for i in range(sample_size):
        # 區間的邊界
        a_start = i * l
        a_end = (i + 1) * l

        s = random.randint(a_start, a_end)

        while i != 0 and s == samples_index[i-1]:
            s = random.randint(a_start, a_end)
        else:
            samples_index.append(s)
    
    for i in range(len(samples_index)):
        c = samples_index[i] % image.shape[1]
        r = int((s - c) / image.shape[1])

        samples_index[i] = [r, c]

    return np.array(samples_index)

def calculate_horizontal(image, samples_index):
    mu_x = 0
    mu_y = 0

    for i in range(samples_index.shape[0]):
        mu_x = np.add(mu_x, image[samples_index[i, 0], samples_index[i, 1]])
        if  samples_index[i, 1] + 1 >= image.shape[1]:
            mu_y = np.add(mu_y, image[samples_index[i, 0], 0])
        else:
            mu_y = np.add(mu_y, image[samples_index[i, 0], samples_index[i, 1] + 1])

    mu_x = mu_x / samples_index.shape[0]
    mu_y = mu_y / samples_index.shape[0]

    r_x_y_1 = 0
    r_x_y_2 = 0
    r_x_y_3 = 0

    for i in range(samples_index.shape[0]):
        if  samples_index[i, 1] + 1 >= image.shape[1]:
            r_x_y_1 = r_x_y_1 + ((image[samples_index[i, 0], samples_index[i, 1]] - mu_x) * (image[samples_index[i, 0], 0] - mu_y))
            r_x_y_2 = r_x_y_2 + (image[samples_index[i, 0], samples_index[i, 1]] - mu_x) ** 2
            r_x_y_3 = r_x_y_3 + (image[samples_index[i, 0], 0] - mu_y) ** 2
        else:
            r_x_y_1 = r_x_y_1 + ((image[samples_index[i, 0], samples_index[i, 1]] - mu_x) * (image[samples_index[i, 0], samples_index[i, 1] + 1] - mu_y))
            r_x_y_2 = r_x_y_2 + (image[samples_index[i, 0], samples_index[i, 1]] - mu_x) ** 2
            r_x_y_3 = r_x_y_3 + (image[samples_index[i, 0], samples_index[i, 1] + 1] - mu_y) ** 2

    horizontal = r_x_y_1 / (math.sqrt(r_x_y_2) * math.sqrt(r_x_y_3))

    return horizontal

def calculate_vertical(image, samples_index):
    mu_x = 0
    mu_y = 0

    for i in range(samples_index.shape[0]):
        mu_x = np.add(mu_x, image[samples_index[i, 0], samples_index[i, 1]])
        if  samples_index[i, 0] + 1 >= image.shape[0]:
            mu_y = np.add(mu_y, image[0, samples_index[i, 1]])
        else:
            mu_y = np.add(mu_y, image[samples_index[i, 0] + 1, samples_index[i, 1]])
    
    mu_x = mu_x / samples_index.shape[0]
    mu_y = mu_y / samples_index.shape[0]

    r_x_y_1 = 0
    r_x_y_2 = 0
    r_x_y_3 = 0

    for i in range(samples_index.shape[0]):
        if  samples_index[i, 1] + 1 >= image.shape[0]:
            r_x_y_1 = r_x_y_1 + ((image[samples_index[i, 0], samples_index[i, 1]] - mu_x) * (image[0, samples_index[i, 1]] - mu_y))
            r_x_y_2 = r_x_y_2 + (image[samples_index[i, 0], samples_index[i, 1]] - mu_x) ** 2
            r_x_y_3 = r_x_y_3 + (image[0, samples_index[i, 1]] - mu_y) ** 2
        else:
            r_x_y_1 = r_x_y_1 + ((image[samples_index[i, 0], samples_index[i, 1]] - mu_x) * (image[samples_index[i, 0] + 1, samples_index[i, 1]] - mu_y))
            r_x_y_2 = r_x_y_2 + (image[samples_index[i, 0], samples_index[i, 1]] - mu_x) ** 2
            r_x_y_3 = r_x_y_3 + (image[samples_index[i, 0] + 1, samples_index[i, 1]] - mu_y) ** 2

    vertical = r_x_y_1 / (math.sqrt(r_x_y_2) * math.sqrt(r_x_y_3))

    return vertical

def calculate_diagonal(image, samples_index):
    mu_x = 0
    mu_y = 0

    for i in range(samples_index.shape[0]):
        mu_x = np.add(mu_x, image[samples_index[i, 0], samples_index[i, 1]])
        if  samples_index[i, 1] + 1 >= image.shape[1] and samples_index[i, 0] + 1 >= image.shape[0]:
            mu_y = np.add(mu_y, image[0, 0])
        elif samples_index[i, 1] + 1 >= image.shape[1]:
            mu_y = np.add(mu_y, image[samples_index[i, 0] + 1, 0])
        elif samples_index[i, 0] + 1 >= image.shape[0]:
            mu_y = np.add(mu_y, image[0, samples_index[i, 1] + 1])
        else:
            mu_y = np.add(mu_y, image[samples_index[i, 0] + 1, samples_index[i, 1] + 1])
    
    mu_x = mu_x / samples_index.shape[0]
    mu_y = mu_y / samples_index.shape[0]

    r_x_y_1 = 0
    r_x_y_2 = 0
    r_x_y_3 = 0

    for i in range(samples_index.shape[0]):
        if  samples_index[i, 1] + 1 >= image.shape[1] and samples_index[i, 0] + 1 >= image.shape[0]:
            r_x_y_1 = r_x_y_1 + ((image[samples_index[i, 0], samples_index[i, 1]] - mu_x) * (image[0, 0] - mu_y))
            r_x_y_2 = r_x_y_2 + (image[samples_index[i, 0], samples_index[i, 1]] - mu_x) ** 2
            r_x_y_3 = r_x_y_3 + (image[0, 0] - mu_y) ** 2
        elif samples_index[i, 1] + 1 >= image.shape[1]:
            r_x_y_1 = r_x_y_1 + ((image[samples_index[i, 0], samples_index[i, 1]] - mu_x) * (image[samples_index[i, 0] + 1, 0] - mu_y))
            r_x_y_2 = r_x_y_2 + (image[samples_index[i, 0], samples_index[i, 1]] - mu_x) ** 2
            r_x_y_3 = r_x_y_3 + (image[samples_index[i, 0] + 1, 0] - mu_y) ** 2
        elif samples_index[i, 0] + 1 >= image.shape[0]:
            r_x_y_1 = r_x_y_1 + ((image[samples_index[i, 0], samples_index[i, 1]] - mu_x) * (image[0, samples_index[i, 1] + 1] - mu_y))
            r_x_y_2 = r_x_y_2 + (image[samples_index[i, 0], samples_index[i, 1]] - mu_x) ** 2
            r_x_y_3 = r_x_y_3 + (image[0, samples_index[i, 1] + 1] - mu_y) ** 2
        else:
            r_x_y_1 = r_x_y_1 + ((image[samples_index[i, 0], samples_index[i, 1]] - mu_x) * (image[samples_index[i, 0] + 1, samples_index[i, 1] + 1] - mu_y))
            r_x_y_2 = r_x_y_2 + (image[samples_index[i, 0], samples_index[i, 1]] - mu_x) ** 2
            r_x_y_3 = r_x_y_3 + (image[samples_index[i, 0] + 1, samples_index[i, 1] + 1] - mu_y) ** 2

    diagonal = r_x_y_1 / (math.sqrt(r_x_y_2) * math.sqrt(r_x_y_3))

    return diagonal

# 計算每個channel的VOH
def calculate_COR(image, sample_size):
    # 抽取樣本
    samples_index = sampling(image, sample_size)

    horizontal = calculate_horizontal(image, samples_index)
    vertical = calculate_vertical(image, samples_index)
    diagonal = calculate_diagonal(image, samples_index)

    return horizontal, vertical, diagonal

# 計算COR
def COR(image, sample_size):
    red_horizontal, red_vertical, red_diagonal = calculate_COR(image[:, :, 2], sample_size)
    green_horizontal, green_vertical, green_diagonal = calculate_COR(image[:, :, 1], sample_size)
    blue_horizontal, blue_vertical, blue_diagonal = calculate_COR(image[:, :, 0], sample_size)

    return red_horizontal, red_vertical, red_diagonal, green_horizontal, green_vertical, green_diagonal, blue_horizontal, blue_vertical, blue_diagonal

# 原始圖片資料夾的路徑
origin_folder_path = "origin"

# 原始圖片資料夾中所有圖片的名字
origin_image_files = os.listdir(origin_folder_path)

# 加密嵌密圖片資料夾的路徑
encrypt_folder_path = "encrp"

# 加密嵌密圖片資料夾中所有圖片的名字
encrypt_image_files = os.listdir(encrypt_folder_path)

with open(f'statis/COR_res.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    writer.writerow(['COR', '', 'Plain', '', '', '', '', '', '', '', '', 'Cipher', '', '', '', '', '', '', '', ''])
    writer.writerow(['Sample', '8000', 'red', '', '', 'green', '', '', 'blue', '', '', 'red', '', '', 'green', '', '', 'blue', '', ''])
    writer.writerow(['Image', 'Type', 'horizontal', 'vertical', 'diagonal', 'horizontal', 'vertical', 'diagonal', 'horizontal', 'vertical', 'diagonal', 'horizontal', 'vertical', 'diagonal', 'horizontal', 'vertical', 'diagonal', 'horizontal', 'vertical', 'diagonal'])

    for i in range(len(encrypt_image_files)):
        # 構建完整的文件路徑
        origin_image_path = os.path.join(origin_folder_path, origin_image_files[i]) 

        # 使用OpenCV讀取圖像
        origin_image = cv2.imread(origin_image_path, cv2.IMREAD_UNCHANGED)

        # 計算COR
        cipher_red_horizontal, cipher_red_vertical, cipher_red_diagonal, cipher_green_horizontal, cipher_green_vertical, cipher_green_diagonal, cipher_blue_horizontal, cipher_blue_vertical, cipher_blue_diagonal = COR(origin_image, 8000)

        # 構建完整的文件路徑
        encrypt_image_path = os.path.join(encrypt_folder_path, encrypt_image_files[i]) 

        # 使用OpenCV讀取圖像
        encrypt_image = cv2.imread(encrypt_image_path, cv2.IMREAD_UNCHANGED)

        # 計算COR
        encrypt_red_horizontal, encrypt_red_vertical, encrypt_red_diagonal, encrypt_green_horizontal, encrypt_green_vertical, encrypt_green_diagonal, encrypt_blue_horizontal, encrypt_blue_vertical, encrypt_blue_diagonal = COR(encrypt_image, 8000)

        # 寫入csv
        writer.writerow([origin_image_files[i], 'color', str(cipher_red_horizontal), str(cipher_red_vertical), str(cipher_red_diagonal), str(cipher_green_horizontal), str(cipher_green_vertical), str(cipher_green_diagonal), str(cipher_blue_horizontal), str(cipher_blue_vertical), str(cipher_blue_diagonal), str(encrypt_red_horizontal), str(encrypt_red_vertical), str(encrypt_red_diagonal), str(encrypt_green_horizontal), str(encrypt_green_vertical), str(encrypt_green_diagonal), str(encrypt_blue_horizontal), str(encrypt_blue_vertical), str(encrypt_blue_diagonal)])