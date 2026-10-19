import os
import random
import shutil

images_path = "datasets/floods/Dataset/images"
labels_path = "datasets/floods/Dataset/labels"

train_img = "datasets/floods/train/images"
train_lbl = "datasets/floods/train/labels"

val_img = "datasets/floods/val/images"
val_lbl = "datasets/floods/val/labels"

images = os.listdir(images_path)

random.shuffle(images)

split = int(len(images) * 0.8)

train_files = images[:split]
val_files = images[split:]

for img in train_files:

    shutil.copy(
        os.path.join(images_path, img),
        os.path.join(train_img, img)
    )

    label = img.replace("image_", "label_").replace(".jpg", ".png")

    shutil.copy(
        os.path.join(labels_path, label),
        os.path.join(train_lbl, label)
    )

for img in val_files:

    shutil.copy(
        os.path.join(images_path, img),
        os.path.join(val_img, img)
    )

    label = img.replace("image_", "label_").replace(".jpg", ".png")

    shutil.copy(
        os.path.join(labels_path, label),
        os.path.join(val_lbl, label)
    )

print("Dataset split complete")