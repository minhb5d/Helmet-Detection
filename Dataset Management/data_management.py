import os
import shutil

def organize_voc_folder(root_dir):
    jpeg_dir = os.path.join(root_dir, 'JPEGImages')
    ann_dir = os.path.join(root_dir, 'Annotations')

    # Tạo thư mục nếu chưa có
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Duyệt qua toàn bộ file trong thư mục gốc
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)
        if os.path.isfile(file_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                shutil.move(file_path, os.path.join(jpeg_dir, filename))
            elif filename.endswith('.xml'):
                shutil.move(file_path, os.path.join(ann_dir, filename))

    print(f"Đã chia xong file trong {root_dir} vào JPEGImages và Annotations.")

# Gọi hàm cho cả train và test
organize_voc_folder('/kaggle/working/helmet-detector/train')  #thay đỏi đường dẫn đến thư mục train
organize_voc_folder('/kaggle/working/helmet-detector/test') #thay đổi đường dẫn đến thư mục test
