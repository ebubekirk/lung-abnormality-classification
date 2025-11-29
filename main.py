import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image
from models.ResNet34 import ResNet34
from models.EfficientNetB0 import EfficientNetB0
from models.DenseNet121 import DenseNet
from helpers.preprocess import preprocess_images

# Load your dataset
def load_data(filepath):
    data = pd.read_csv(filepath)

    filenames = data['Image Index'].astype(str).values
    image_paths = [os.path.join('data/images/', fname) for fname in filenames]
    # Split labels by '|' and convert to list of lists
    labels = [label.split('|') for label in data['Finding Labels'].astype(str).values]
    return np.array(image_paths), labels

def load_images(image_paths, target_size=(200, 200)):
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    if not images:
        # return empty array with expected rank 4: (0, H, W, C)
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32)

    # stack to ensure a single dense ndarray, enforce float32 and contiguous memory
    images_np = np.stack(images).astype(np.float32)
    images_np = np.ascontiguousarray(images_np)
    return images_np

# Train and evaluate model
def evaluate_multi_label(y_true, y_probs, class_names, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    n_classes = y_true.shape[1]

    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Per-class accuracy (column-wise)
    accuracies = np.array([(y_true[:, i] == y_pred[:, i]).mean() for i in range(n_classes)])

    aucs = []
    for i in range(n_classes):
        try:
            # roc_auc_score requires at least one positive and one negative sample in y_true[:, i]
            auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        except Exception:
            auc = np.nan
        aucs.append(auc)
    aucs = np.array(aucs, dtype=float)

    metrics = {}
    for i, cname in enumerate(class_names):
        metrics[cname] = {
            'precision': float(precisions[i]),
            'recall': float(recalls[i]),
            'f1': float(f1s[i]),
            'accuracy': float(accuracies[i]),
            'auc': float(aucs[i]) if not np.isnan(aucs[i]) else None
        }
    return metrics

def train_and_evaluate_model(model, train_data, train_labels, val_data, val_labels, mlb):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Transform labels to multi-hot encoding
    train_labels_encoded = mlb.transform(train_labels)
    val_labels_encoded = mlb.transform(val_labels)

    # Ensure numeric dtypes and contiguous arrays so TF can copy to device reliably
    train_data = np.ascontiguousarray(train_data, dtype=np.float32)
    val_data = np.ascontiguousarray(val_data, dtype=np.float32)
    train_labels_encoded = np.ascontiguousarray(train_labels_encoded, dtype=np.float32)
    val_labels_encoded = np.ascontiguousarray(val_labels_encoded, dtype=np.float32)

    # quick sanity check (prints minimal info)
    print(f"train_data.shape={train_data.shape}, dtype={train_data.dtype}; train_labels.shape={train_labels_encoded.shape}, dtype={train_labels_encoded.dtype}")
    print(f"val_data.shape={val_data.shape}, dtype={val_data.dtype}; val_labels.shape={val_labels_encoded.shape}, dtype={val_labels_encoded.dtype}")

    try:
        model.fit(train_data, train_labels_encoded, epochs=10, batch_size=32, validation_data=(val_data, val_labels_encoded))
    except Exception as e:
        print("Error during model.fit:", e)
        raise

    val_probs = model.predict(val_data)
    # Return per-class metrics for this fold
    fold_metrics = evaluate_multi_label(val_labels_encoded, val_probs, mlb.classes_, threshold=0.5)
    return fold_metrics

def main():
    # Load data
    image_paths, labels = load_data('data/Data_Entry_2017.csv')
    images = load_images(image_paths)

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Store per-fold metrics per model
    results = {
        'ResNet34': [],
        'EfficientNetB0': [],
        'DenseNet121': []
    }

    for train_index, val_index in kf.split(images):
        train_data, val_data = images[train_index], images[val_index]
        train_labels, val_labels = [labels[i] for i in train_index], [labels[i] for i in val_index]

        # Initialize models
        resnet_model = ResNet34(num_classes=len(mlb.classes_))
        efficientnet_model = EfficientNetB0(num_classes=len(mlb.classes_))
        densenet_model = DenseNet(num_classes=len(mlb.classes_))

        # Train and evaluate each model
        print("Training ResNet34...")
        resnet_fold_metrics = train_and_evaluate_model(resnet_model, train_data, train_labels, val_data, val_labels, mlb)
        results['ResNet34'].append(resnet_fold_metrics)
        
        print("Training EfficientNetB0...")
        efficientnet_fold_metrics = train_and_evaluate_model(efficientnet_model, train_data, train_labels, val_data, val_labels, mlb)
        results['EfficientNetB0'].append(efficientnet_fold_metrics)

        print("Training DenseNet121...")
        densenet_fold_metrics = train_and_evaluate_model(densenet_model, train_data, train_labels, val_data, val_labels, mlb)
        results['DenseNet121'].append(densenet_fold_metrics)

    # Aggregate and print average per-class metrics across folds
    for model_name, folds in results.items():
        if not folds:
            continue
        # build per-class lists
        class_names = mlb.classes_
        agg = {c: {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'auc': []} for c in class_names}
        for fold in folds:
            for c in class_names:
                m = fold.get(c, {})
                agg[c]['precision'].append(m.get('precision', np.nan))
                agg[c]['recall'].append(m.get('recall', np.nan))
                agg[c]['f1'].append(m.get('f1', np.nan))
                agg[c]['accuracy'].append(m.get('accuracy', np.nan))
                agg[c]['auc'].append(m.get('auc', np.nan))

        print(f"\n{model_name} per-class metrics (mean ± std over folds):")
        for c in class_names:
            p_mean, p_std = np.nanmean(agg[c]['precision']), np.nanstd(agg[c]['precision'])
            r_mean, r_std = np.nanmean(agg[c]['recall']), np.nanstd(agg[c]['recall'])
            f_mean, f_std = np.nanmean(agg[c]['f1']), np.nanstd(agg[c]['f1'])
            a_mean, a_std = np.nanmean(agg[c]['accuracy']), np.nanstd(agg[c]['accuracy'])
            auc_vals = np.array(agg[c]['auc'], dtype=float)
            auc_mean, auc_std = np.nanmean(auc_vals), np.nanstd(auc_vals)
            print(f"{c}: precision={p_mean:.4f}±{p_std:.4f}, recall={r_mean:.4f}±{r_std:.4f}, f1={f_mean:.4f}±{f_std:.4f}, accuracy={a_mean:.4f}±{a_std:.4f}, auc={auc_mean:.4f}±{auc_std:.4f}")

if __name__ == "__main__":
    main()