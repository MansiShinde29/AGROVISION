import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from imgaug import augmenters as iaa

# Path to dataset
base_dir = os.getcwd()
dataset_path = os.path.join(base_dir, "home", "static", "disease_detection")
model_save_path = os.path.join(base_dir, "home", "models", "disease_model.pkl")

# Ensure dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"🚨 Dataset folder not found: {dataset_path}")

# Data Augmentation
augmentor = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 1.0))
])

def augment_image(image):
    return augmentor.augment_image(image)

def extract_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    hist /= hist.sum() + 1e-6
    return hist

features = []
labels = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                for _ in range(20):  # Generate 20 samples per image
                    aug_image = augment_image(image)
                    features.append(extract_features(aug_image))
                    labels.append(category)

features = np.array(features)
labels = np.array(labels)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test) * 100

with open(model_save_path, "wb") as f:
    pickle.dump((clf, label_encoder), f)

print(f"✅ Disease Detection Model Trained & Saved Successfully with Accuracy: {accuracy:.2f}%")
