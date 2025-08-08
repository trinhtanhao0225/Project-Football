import glob
import os
import json
import shutil

class FootBallDataset:
    def __init__(self, dataset_path, output_path, output_name, is_train):
        # Tạo thư mục output
        output_folder_status = os.path.join(output_path, output_name,is_train)
        os.makedirs(output_folder_status,exist_ok=True)
        self.image_path = os.path.join(output_folder_status, 'images')
        self.label_path = os.path.join(output_folder_status, 'labels')
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)

        # Tìm ảnh và file annotation
        folder_path_status = os.path.join(dataset_path, is_train)
        self.image_files = list(glob.iglob(f"{folder_path_status}/*.jpg"))
        json_files = list(glob.iglob(f"{folder_path_status}/*.json"))
        
        self.annotation_file = json_files[0]

    def move_image_create_txt(self):
        with open(self.annotation_file, "r") as f:
            coco = json.load(f)

        # Map ảnh theo id
        image_id_to_info = {img['id']: img for img in coco['images']}

        for ann in coco['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id'] -1
            bbox = ann['bbox']  

            img_info = image_id_to_info[image_id]
            file_name = img_info['file_name']
            width = img_info['width']
            height = img_info['height']

            # Chuyển bbox về YOLO format
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            w = bbox[2] / width
            h = bbox[3] / height

            # Copy ảnh
            src_img_path = os.path.join(os.path.dirname(self.annotation_file), file_name)
            dst_img_path = os.path.join(self.image_path, file_name)
            shutil.copy(src_img_path, dst_img_path)

            # Ghi file label
            label_file = os.path.splitext(file_name)[0] + ".txt"
            label_path = os.path.join(self.label_path, label_file)
            with open(label_path, "a") as f:
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__=='__main__':
    dataset = FootBallDataset(r'C:\Users\Public\Documents\Football_Project\dataset\football-players-detection.v1i.coco',r'C:\Users\Public\Documents\Football_Project\dataset','football-yolo-format','train')
    dataset.move_image_create_txt()
