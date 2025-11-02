import os, argparse, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
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

def train_eval_per_user(data_root, k):
    train, test = load_feature_splits(data_root)
    unique_subjects = np.unique(test.subjects)
    metrics = {}
    for s in unique_subjects:
        tr_mask = train.subjects == s
        te_mask = test.subjects == s
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train.X[tr_mask])
        Xte = scaler.transform(test.X[te_mask])
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        clf.fit(Xtr, train.y[tr_mask])
        yhat = clf.predict(Xte)
        P,R,F1 = evaluate(test.y[te_mask], yhat, f"user_{s}")
        metrics[int(s)] = dict(precision=P, recall=R, f1=F1, n=te_mask.sum())
    total = sum(v["n"] for v in metrics.values())
    if total > 0:
        P = sum(v["precision"]*v["n"] for v in metrics.values())/total
        R = sum(v["recall"]*v["n"] for v in metrics.values())/total
        F1= sum(v["f1"]*v["n"] for v in metrics.values())/total
        print(f"[per-user weighted] Precision={P:.4f} Recall={R:.4f} F1={F1:.4f}")
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    print("=== Part 1: KNN on engineered features ===")
    train_eval_aggregate(args.data_root, args.k)
    print("\n=== Per-user models ===")
    train_eval_per_user(args.data_root, args.k)

if __name__ == "__main__":
    main()
