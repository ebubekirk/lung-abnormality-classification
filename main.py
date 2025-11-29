import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.ResNet34 import ResNet34
from models.EfficientNetB0 import EfficientNetB0
from models.DenseNet121 import DenseNet

# Load your dataset
def load_data(filepath):
    data = pd.read_csv(filepath)

    filenames = data['Image Index'].astype(str).values
    image_paths = [os.path.join('data/images/', fname) for fname in filenames]
    labels = data['Finding Labels'].values
    return np.array(image_paths), labels

# Preprocess images
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(path, target_size=(200, 200))  # Adjust size as needed
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
    return np.array(images) / 255.0  # Normalize images

# Train and evaluate model
def train_and_evaluate_model(model, train_data, train_labels, val_data, val_labels):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))
    loss, accuracy = model.evaluate(val_data, val_labels)
    return accuracy

def main():
    # Load data
    image_paths, labels = load_data('data/Data_Entry_2017.csv')
    images = preprocess_images(image_paths)

    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Store results
    results = {
        'ResNet34': [],
        'EfficientNetB0': [],
        'DenseNet121': []
    }

    for train_index, val_index in kf.split(images):
        train_data, val_data = images[train_index], images[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        # Initialize models
        resnet_model = ResNet34(input_channels=3, num_classes=len(np.unique(labels)))
        efficientnet_model = EfficientNetB0(num_classes=len(np.unique(labels)))
        densenet_model = DenseNet(num_classes=len(np.unique(labels)))

        # Train and evaluate each model
        print("Training ResNet34...")
        resnet_accuracy = train_and_evaluate_model(resnet_model, train_data, train_labels, val_data, val_labels)
        results['ResNet34'].append(resnet_accuracy)

        print("Training EfficientNetB0...")
        efficientnet_accuracy = train_and_evaluate_model(efficientnet_model, train_data, train_labels, val_data, val_labels)
        results['EfficientNetB0'].append(efficientnet_accuracy)

        print("Training DenseNet121...")
        densenet_accuracy = train_and_evaluate_model(densenet_model, train_data, train_labels, val_data, val_labels)
        results['DenseNet121'].append(densenet_accuracy)

    # Print average results
    for model_name, accuracies in results.items():
        print(f"{model_name} Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

if __name__ == "__main__":
    main()