"""Train a 7-class facial-expression CNN from the supplied FER folder dataset."""
from pathlib import Path
import argparse, json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
CLASS_NAMES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
FOLDER_TO_CLASS = {
    "angry": "angry", "disgust": "disgusted", "disgusted": "disgusted",
    "fear": "fearful", "fearful": "fearful", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "surprise": "surprised",
    "surprised": "surprised",
}


def collect_images(split_dir):
    paths, labels = [], []
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")
    for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if folder.name.lower() not in FOLDER_TO_CLASS:
            print(f"Skipping unknown folder: {folder.name}")
            continue
        label = CLASS_NAMES.index(FOLDER_TO_CLASS[folder.name.lower()])
        for path in folder.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                paths.append(str(path))
                labels.append(label)
    if not paths:
        raise ValueError(f"No supported images found in {split_dir}")
    return np.asarray(paths), np.asarray(labels, dtype=np.int32)


def decode_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 10000), reshuffle_each_iteration=True)
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
        ])
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="FER-56-images-per-class/facial_expression_source",
                        help="Folder containing train/ and test/ directories")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    root = Path(args.data)
    train_paths, train_labels = collect_images(root / "train")
    test_paths, test_labels = collect_images(root / "test")
    tr_p, val_p, tr_y, val_y = train_test_split(
        train_paths, train_labels, test_size=0.20, random_state=42, stratify=train_labels)
    train_ds = make_dataset(tr_p, tr_y, training=True)
    val_ds = make_dataset(val_p, val_y)
    test_ds = make_dataset(test_paths, test_labels)
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(tr_y), y=tr_y)
    class_weights = {int(k): float(v) for k, v in zip(np.unique(tr_y), weights)}
    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("facial_expression_model.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=class_weights, callbacks=callbacks)
    model.save("facial_expression_model.keras")
    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    results = model.evaluate(test_ds, return_dict=True, verbose=0)
    print("Test metrics:", results)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    print(classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES))))
    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

if __name__ == "__main__":
    main()
