import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import kagglehub

# 1. Config & Seeds
SEED = 42
tf.keras.utils.set_random_seed(SEED)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

print("=" * 60)
print("BRAIN TUMOR MRI MODEL RETRAINING PIPELINE")
print("Target: >96% Test Accuracy with Balanced Generalization")
print("=" * 60)

# Check Metal GPU
physical_devices = tf.config.list_physical_devices()
print("Physical devices detected:", physical_devices)

# Dataset path
download_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
TRAIN_DIR = os.path.join(download_path, "Training")
TEST_DIR = os.path.join(download_path, "Testing")

print(f"Training Directory: {TRAIN_DIR}")
print(f"Testing Directory:  {TEST_DIR}")

# 2. Data Loaders using image_dataset_from_directory
# Note: Keep pixel values in [0, 255] because EfficientNetB0 has its own built-in Rescaling layer!
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.15,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASSES
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASSES
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASSES,
    shuffle=False
)

# Compute Class Weights to balance rare categories
train_labels = []
for _, labels in raw_train_ds:
    train_labels.extend(np.argmax(labels.numpy(), axis=1))
train_labels = np.array(train_labels)

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)
class_weights = dict(enumerate(class_weights_arr))
print("Computed Class Weights:", class_weights)

# 3. Data Augmentation (Lightweight to prevent structural distortion of MRI scans)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.05, seed=SEED),
    layers.RandomZoom(0.05, seed=SEED),
], name="mri_augmentation")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = raw_train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(buffer_size=AUTOTUNE)

val_ds = raw_val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds_buffered = test_ds.prefetch(buffer_size=AUTOTUNE)

# 4. Model Architecture: EfficientNetB0 Backbone + Regularized Top
def create_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze for Phase 1

    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    # EfficientNetB0 expects raw inputs in [0, 255] and applies internal Rescaling + Normalization
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="top_bn")(x)
    x = layers.Dropout(0.35, name="top_dropout_1")(x)
    x = layers.Dense(
        256, 
        activation="relu", 
        kernel_regularizer=regularizers.l2(1e-4),
        name="top_dense"
    )(x)
    x = layers.Dropout(0.25, name="top_dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="BrainTumor_EfficientNetB0")
    return model, base_model

model, base_model = create_model()
model.summary()

# 5. Phase 1: Train Top Classifier Head
print("\n>>> STARTING PHASE 1: Training Classification Head (Base Model Frozen) <<<")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model_phase1.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

start_phase1 = time.time()
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks_phase1,
    verbose=1
)
print(f"Phase 1 completed in {(time.time() - start_phase1) / 60:.2f} mins.")

# 6. Phase 2: Fine-Tuning Top 50 Layers of EfficientNetB0
print("\n>>> STARTING PHASE 2: Fine-Tuning Top 50 Layers of EfficientNetB0 <<<")
base_model.trainable = True
# Freeze all except top 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# BatchNormalization layers in the base should remain in inference mode to preserve learned statistics
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = len([l for l in model.trainable_weights])
print(f"Trainable weight tensors in Phase 2: {trainable_count}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "brain_tumor_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

start_phase2 = time.time()
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks_phase2,
    verbose=1
)
print(f"Phase 2 completed in {(time.time() - start_phase2) / 60:.2f} mins.")

# 7. Final Comprehensive Evaluation on Test Set
print("\n>>> EVALUATING ON TEST SET <<<")
test_results = model.evaluate(test_ds_buffered, verbose=1)
test_metrics = dict(zip(model.metrics_names, test_results))
print("Final Test Metrics:", test_metrics)

# Generate Predictions & Confusion Matrix
y_true_all = []
y_pred_all = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true_all.extend(np.argmax(labels.numpy(), axis=1))
    y_pred_all.extend(np.argmax(preds, axis=1))

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

clf_report = classification_report(y_true_all, y_pred_all, target_names=CLASSES, output_dict=True)
print("\nClassification Report:\n", classification_report(y_true_all, y_pred_all, target_names=CLASSES))

# Save Confusion Matrix Plot
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Test Confusion Matrix (Test Accuracy: {test_metrics['accuracy']:.2%})")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig("test_confusion_matrix.png", dpi=300)
plt.close()

# Save Metrics JSON
eval_summary = {
    "test_accuracy": float(test_metrics["accuracy"]),
    "test_loss": float(test_metrics["loss"]),
    "test_auc": float(test_metrics["auc"]),
    "classification_report": clf_report
}
with open("training_evaluation_metrics.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print(f"\nAll models and metrics saved successfully!")
print(f"Final Model: brain_tumor_model.keras (Accuracy: {test_metrics['accuracy']:.2%})")
