import os
import shutil

def organize_voc_folder(root_dir):
    """
    Organize VOC style dataset:
    - Move all images (.jpg, .png) into JPEGImages folder
    - Move all annotations (.xml) into Annotations folder
    """
    jpeg_dir = os.path.join(root_dir, 'JPEGImages')
    ann_dir = os.path.join(root_dir, 'Annotations')

    # Create directories if not exist
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Move files
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith(('.jpg', '.png')):
                shutil.move(file_path, os.path.join(jpeg_dir, filename))
            elif filename.lower().endswith('.xml'):
                shutil.move(file_path, os.path.join(ann_dir, filename))

    print(f"✅ Done organizing: {root_dir}")

def main():
    # Define paths to train and test folders
    dataset_root = "../helmet-detector"  # Update this if needed

    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")

    print("📦 Organizing training data...")
    organize_voc_folder(train_dir)

    print("📦 Organizing test data...")
    organize_voc_folder(test_dir)

if __name__ == '__main__':
    main()
