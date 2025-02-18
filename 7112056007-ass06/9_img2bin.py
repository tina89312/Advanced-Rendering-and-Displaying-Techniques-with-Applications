import cv2
import os


# 轉換成二進制
def Convert_to_binary(image):
    image_binary = ''
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_binary = image_binary + bin(image[i, j])[2:].zfill(8)

    return image_binary

# 將影像的二進制寫入txt
def write_txt(binary_value, file_name):
    with open(f"binar/{file_name}", "w") as file:
        # 将字符串写入文件
        file.write(binary_value)

    return

# 圖片資料夾的路徑
origin_folder_path = "origin"

# 資料夾中所有圖片的名字
origin_image_files = os.listdir(origin_folder_path)

for image_file in origin_image_files:
    # 構建完整的文件路徑
    image_path = os.path.join(origin_folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if len(image.shape) == 3:
        # 轉換成二進制
        image_binary_red = Convert_to_binary(image[:, :, 2])
        image_binary_green = Convert_to_binary(image[:, :, 1])
        image_binary_blue = Convert_to_binary(image[:, :, 0])

        # 將影像的二進制寫入txt
        write_txt(image_binary_red, image_file[:-4] + '_R_M2.txt')
        write_txt(image_binary_green, image_file[:-4] + '_G_M2.txt')
        write_txt(image_binary_blue, image_file[:-4] + '_B_M2.txt')
    else:
        # 轉換成二進制
        image_binary = Convert_to_binary(image)

        # 將影像的二進制寫入txt
        write_txt(image_binary, image_file[:-4] + '_M2.txt')

# 圖片資料夾的路徑
encry_folder_path = "encry"

# 資料夾中所有圖片的名字
encry_image_files = os.listdir(encry_folder_path)

for image_file in encry_image_files:
    # 構建完整的文件路徑
    image_path = os.path.join(encry_folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if len(image.shape) == 3:
        # 轉換成二進制
        image_binary_red = Convert_to_binary(image[:, :, 2])
        image_binary_green = Convert_to_binary(image[:, :, 1])
        image_binary_blue = Convert_to_binary(image[:, :, 0])

        # 將影像的二進制寫入txt
        write_txt(image_binary_red, image_file[:-4] + '_R_M2.txt')
        write_txt(image_binary_green, image_file[:-4] + '_G_M2.txt')
        write_txt(image_binary_blue, image_file[:-4] + '_B_M2.txt')
    else:
        # 轉換成二進制
        image_binary = Convert_to_binary(image)

        # 將影像的二進制寫入txt
        write_txt(image_binary, image_file[:-4] + '_M2.txt')
