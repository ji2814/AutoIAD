
import os
import pandas as pd

dataset_root = "D:/aotuIAD/dataset/MVTecAD/bottle"
output_csv_path = "workspace/dataset.csv"

data = []

# Process training data
train_good_dir = os.path.join(dataset_root, "train", "good")
for img_name in os.listdir(train_good_dir):
    if img_name.endswith(".png"):
        image_path = os.path.join("train", "good", img_name).replace("\\", "/")
        data.append({
            "image_path": image_path,
            "gt_label": "good",
            "gt_mask": "none",
            "split": "train",
            "is_normal": 1,
            "defect_type": "none"
        })

# Process testing data
test_dir = os.path.join(dataset_root, "test")
for defect_type_folder in os.listdir(test_dir):
    defect_path = os.path.join(test_dir, defect_type_folder)
    if os.path.isdir(defect_path):
        for img_name in os.listdir(defect_path):
            if img_name.endswith(".png"):
                image_path = os.path.join("test", defect_type_folder, img_name).replace("\\", "/")
                gt_label = defect_type_folder
                is_normal = 0 if defect_type_folder != "good" else 1

                gt_mask = "none"
                if not is_normal:
                    mask_name = img_name.replace(".png", "_mask.png")
                    mask_path_candidate = os.path.join(dataset_root, "ground_truth", defect_type_folder, mask_name).replace("\\", "/")
                    if os.path.exists(mask_path_candidate):
                        gt_mask = os.path.join("ground_truth", defect_type_folder, mask_name).replace("\\", "/")

                data.append({
                    "image_path": image_path,
                    "gt_label": gt_label,
                    "gt_mask": gt_mask,
                    "split": "test",
                    "is_normal": is_normal,
                    "defect_type": defect_type_folder if not is_normal else "none"
                })

df = pd.DataFrame(data)

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(output_csv_path, index=False)
print(f"Dataset CSV created at: {output_csv_path}")
