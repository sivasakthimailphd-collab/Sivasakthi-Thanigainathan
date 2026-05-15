import random
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


@dataclass
class Config:
    data_root: str
    train_save_path: str
    test_save_path: str
    image_size: int = 224
    test_ratio: float = 0.2
    batch_size: int = 8   
    epochs: int = 5
    seed: int = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def augment(img):

    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    if random.random() < 0.5:
        angle = random.randint(-25, 25)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        img = cv2.resize(img, (224, 224))

    if random.random() < 0.3:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def scan_dataset(root):
    root = Path(root)
    samples = []

    for i, cls in enumerate(sorted([d for d in root.iterdir() if d.is_dir()])):
        for img in cls.glob("*"):
            samples.append((str(img), i))

    return samples


def split(samples, cfg):
    class_dict = defaultdict(list)

    for p, l in samples:
        class_dict[l].append(p)

    train, test = [], []

    for l, paths in class_dict.items():
        if len(paths) < 6:
            continue

        random.shuffle(paths)
        split_idx = int(len(paths) * (1 - cfg.test_ratio))

        train += [(p, l) for p in paths[:split_idx]]
        test += [(p, l) for p in paths[split_idx:]]

    return train, test


def save_image(img, label, path, root):
    class_dir = Path(root) / str(label)
    class_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(class_dir / Path(path).name), img)


def build_dataset(samples, cfg, save_root=None, is_train=True):

    paths = [p for p, _ in samples]
    labels = [l for _, l in samples]

    def load_fn(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (cfg.image_size, cfg.image_size))

        img_np = img.numpy().astype(np.uint8)

        if is_train:
            img_np = augment(img_np)

        if save_root:
            save_image(img_np, int(label.numpy()), path.numpy().decode(), save_root)

        img_np = preprocess_input(img_np.astype(np.float32))
        return img_np, label

    def tf_wrapper(path, label):
        img, lbl = tf.py_function(load_fn, [path, label], [tf.float32, tf.int32])
        img.set_shape((cfg.image_size, cfg.image_size, 3))
        lbl.set_shape(())
        return img, lbl

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(1000) if is_train else ds
    ds = ds.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


def build_model(num_classes):

    base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    output = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(base.input, output)

    model.compile(
        optimizer=Adam(0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    feature_model = models.Model(model.input, model.layers[-3].output)

    return model, feature_model


def extract_features(ds, model):

    feats, labels = [], []

    for batch_imgs, batch_lbls in ds:
        f = model.predict(batch_imgs, verbose=0)
        feats.append(f)
        labels.append(batch_lbls.numpy())

    return normalize(np.vstack(feats)), np.hstack(labels)


def dcg_score(rel):
    rel = np.asarray(rel)
    return np.sum(rel / np.log2(np.arange(2, rel.size + 2))) if rel.size else 0.0


def evaluate(query_f, query_l, db_f, db_l):

    recalls, precisions, maps, ndcgs, times = [], [], [], [], []

    for qf, ql in zip(query_f, query_l):

        start = time.perf_counter()

        sims = np.dot(db_f, qf)
        dists = 1 - sims

        total_relevant = np.sum(db_l == ql)
        k = max(1, min(int(total_relevant * 1.2), len(db_l)))

        idx = np.argsort(dists)[:k]

        times.append(time.perf_counter() - start)

        rel = (db_l[idx] == ql).astype(int)

        precision = np.mean(rel)
        recall = np.sum(rel) / (total_relevant + 1e-10)

        scores = sims[idx]
        ap = average_precision_score(rel, scores) if len(np.unique(rel)) > 1 else np.mean(rel)

        ideal = sorted(rel, reverse=True)
        ndcg = dcg_score(rel) / (dcg_score(ideal) + 1e-12)

        recalls.append(recall)
        precisions.append(precision)
        maps.append(ap)
        ndcgs.append(ndcg)

    return {
        "Recall@K(%)": 100 * np.mean(recalls),
        "Precision@K(%)": 100 * np.mean(precisions),
        "mAP@K": np.mean(maps),
        "nDCG@K": np.mean(ndcgs),
        "RetrievalTime(s)": np.mean(times),
    }


def train(cfg):

    set_seed(cfg.seed)

    shutil.rmtree(cfg.train_save_path, ignore_errors=True)
    shutil.rmtree(cfg.test_save_path, ignore_errors=True)

    samples = scan_dataset(cfg.data_root)
    train_s, test_s = split(samples, cfg)

    num_classes = len(set([l for _, l in train_s]))

    train_ds = build_dataset(train_s, cfg, cfg.train_save_path, True)
    test_ds = build_dataset(test_s, cfg, cfg.test_save_path, False)

    model, feature_model = build_model(num_classes)

    early = EarlyStopping(patience=5, restore_best_weights=True)
    lr = ReduceLROnPlateau(patience=2, factor=0.3)

    model.fit(train_ds, validation_data=test_ds,
              epochs=cfg.epochs, callbacks=[early, lr])

    
    for layer in model.layers[-40:]:
        layer.trainable = True

    model.compile(optimizer=Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_ds, validation_data=test_ds, epochs=5)

    
    train_f, train_l = extract_features(train_ds, feature_model)
    test_f, test_l = extract_features(test_ds, feature_model)

    metrics = evaluate(test_f, test_l, train_f, train_l)

    print("\n FINAL METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":

    cfg = Config(
        data_root=r"enter your dataset path",
        train_save_path=r"enter your trained output path",
        test_save_path=r"enter your test output path",
        epochs=5
    )

    train(cfg)