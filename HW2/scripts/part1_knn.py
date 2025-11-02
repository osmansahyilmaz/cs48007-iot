import os, argparse, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from utils.data import load_feature_splits, UCI_LABELS

def evaluate(y_true, y_pred, label="aggregate", out_dir="outputs/part1"):
    os.makedirs(out_dir, exist_ok=True)
    report = classification_report(
        y_true, y_pred, digits=4, target_names=[UCI_LABELS[i] for i in sorted(UCI_LABELS)]
    )
    with open(os.path.join(out_dir, f"{label}_report.txt"), "w") as f:
        f.write(report)
    P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"[{label}] Macro Precision={P:.4f} Recall={R:.4f} F1={F1:.4f}")
    return P, R, F1

def train_eval_aggregate(data_root, k):
    train, test = load_feature_splits(data_root)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train.X)
    Xte = scaler.transform(test.X)
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(Xtr, train.y)
    yhat = clf.predict(Xte)
    return evaluate(test.y, yhat, "aggregate")

def train_eval_per_user_holdout(data_root, k, holdout):
    train, _ = load_feature_splits(data_root)
    unique_subjects = np.unique(train.subjects)
    metrics = {}
    for s in unique_subjects:
        mask = train.subjects == s
        if mask.sum() < 10:
            continue
        X_user = train.X[mask]
        y_user = train.y[mask]
        Xtr, Xte, ytr, yte = train_test_split(X_user, y_user, test_size=holdout, stratify=y_user, random_state=42)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        P, R, F1 = evaluate(yte, yhat, f"user_{s}")
        metrics[int(s)] = dict(precision=P, recall=R, f1=F1, n=len(yte))
    # weighted summary
    total = sum(v["n"] for v in metrics.values())
    if total > 0:
        P = sum(v["precision"] * v["n"] for v in metrics.values()) / total
        R = sum(v["recall"] * v["n"] for v in metrics.values()) / total
        F1 = sum(v["f1"] * v["n"] for v in metrics.values()) / total
        print(f"[per-user holdout weighted] Precision={P:.4f} Recall={R:.4f} F1={F1:.4f}")
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per-user-holdout", type=float, default=None,
                    help="Optional fraction (0–1) for per-user train/val split inside training data")
    args = ap.parse_args()

    print("=== Part 1: KNN on engineered features ===")
    train_eval_aggregate(args.data_root, args.k)

    if args.per_user_holdout is not None:
        print(f"\n=== Per-user holdout models (holdout={args.per_user_holdout}) ===")
        train_eval_per_user_holdout(args.data_root, args.k, args.per_user_holdout)

if __name__ == "__main__":
    main()
