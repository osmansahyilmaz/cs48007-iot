import os
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

# Activity labels from dataset
UCI_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

def _read_txt_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path)

def _read_txt_vector(path: str) -> np.ndarray:
    return np.loadtxt(path).astype(int).ravel()

@dataclass
class FeatureSplit:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray

def load_feature_splits(data_root: str) -> Tuple[FeatureSplit, FeatureSplit]:
    """Loads engineered features (X_train/test.txt, y_train/test.txt)."""
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    X_train = _read_txt_matrix(os.path.join(train_dir, "X_train.txt"))
    y_train = _read_txt_vector(os.path.join(train_dir, "y_train.txt"))
    subj_train = _read_txt_vector(os.path.join(train_dir, "subject_train.txt"))

    X_test = _read_txt_matrix(os.path.join(test_dir, "X_test.txt"))
    y_test = _read_txt_vector(os.path.join(test_dir, "y_test.txt"))
    subj_test = _read_txt_vector(os.path.join(test_dir, "subject_test.txt"))

    return FeatureSplit(X_train, y_train, subj_train), FeatureSplit(X_test, y_test, subj_test)

@dataclass
class RawSplit:
    total_acc: np.ndarray
    body_acc: np.ndarray
    body_gyro: np.ndarray
    y: np.ndarray
    subjects: np.ndarray

def _read_signal_triplet(split_dir: str, prefix: str) -> List[np.ndarray]:
    paths = [
        os.path.join(split_dir, "Inertial Signals", f"{prefix}_{axis}_{os.path.basename(split_dir)}.txt")
        for axis in ("x", "y", "z")
    ]
    return [np.loadtxt(p) for p in paths]

def load_raw_windows(data_root: str) -> Tuple[RawSplit, RawSplit]:
    """Loads pre-windowed inertial signals from UCI-HAR."""
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    total_acc_train = np.stack(_read_signal_triplet(train_dir, "total_acc"), axis=-1)
    body_acc_train  = np.stack(_read_signal_triplet(train_dir, "body_acc"),  axis=-1)
    body_gyro_train = np.stack(_read_signal_triplet(train_dir, "body_gyro"), axis=-1)
    y_train = _read_txt_vector(os.path.join(train_dir, "y_train.txt"))
    subj_train = _read_txt_vector(os.path.join(train_dir, "subject_train.txt"))

    total_acc_test = np.stack(_read_signal_triplet(test_dir, "total_acc"), axis=-1)
    body_acc_test  = np.stack(_read_signal_triplet(test_dir, "body_acc"),  axis=-1)
    body_gyro_test = np.stack(_read_signal_triplet(test_dir, "body_gyro"), axis=-1)
    y_test = _read_txt_vector(os.path.join(test_dir, "y_test.txt"))
    subj_test = _read_txt_vector(os.path.join(test_dir, "subject_test.txt"))

    train = RawSplit(total_acc_train, body_acc_train, body_gyro_train, y_train, subj_train)
    test  = RawSplit(total_acc_test,  body_acc_test,  body_gyro_test,  y_test,  subj_test)
    return train, test

def stack_modalities(raw: RawSplit, use_modalities=("total_acc","body_acc","body_gyro")) -> np.ndarray:
    """Concatenates selected 3-axis modalities to [N,L,C]."""
    chans = []
    if "total_acc" in use_modalities: chans.append(raw.total_acc)
    if "body_acc"  in use_modalities: chans.append(raw.body_acc)
    if "body_gyro" in use_modalities: chans.append(raw.body_gyro)
    return np.concatenate(chans, axis=-1)
