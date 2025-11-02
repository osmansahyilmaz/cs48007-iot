import os, argparse, numpy as np, torch, random
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
from utils.data import load_raw_windows, stack_modalities, UCI_LABELS

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class CNN1D(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):
        x = x.transpose(1,2)
        h = self.net(x).squeeze(-1)
        return self.fc(h)

def make_loader(X, y, bs, shuffle=False):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y - 1, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle)

def train_epoch(model, loader, opt, device):
    model.train(); loss_fn = nn.CrossEntropyLoss()
    total, correct, losses = 0, 0, []
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward(); opt.step()
        losses.append(loss.item())
        pred = logits.argmax(1)
        total += y.size(0); correct += (pred==y).sum().item()
    return np.mean(losses), correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for X,y in loader:
        X = X.to(device)
        logits = model(X)
        y_true.append(y.numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_true1, y_pred1 = y_true + 1, y_pred + 1
    labels = [1,2,3,4,5,6]
    P,R,F1,_ = precision_recall_fscore_support(y_true1,y_pred1,average="macro",zero_division=0,labels=labels)
    return P,R,F1,y_true1,y_pred1

def train_and_eval(Xtr,ytr,Xte,yte,args,label,out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1D(in_ch=Xtr.shape[-1], n_classes=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_loader = make_loader(Xtr,ytr,args.batch_size,True)
    te_loader = make_loader(Xte,yte,args.batch_size)
    bestF1, best_state = -1, None
    for e in range(1,args.epochs+1):
        loss,acc = train_epoch(model,tr_loader,opt,device)
        P,R,F1,_,_ = evaluate(model,te_loader,device)
        print(f"[{label}] Epoch {e}/{args.epochs} loss={loss:.4f} acc={acc:.3f} P={P:.3f} R={R:.3f} F1={F1:.3f}")
        if F1>bestF1: bestF1,best_state = F1,{k:v.cpu() for k,v in model.state_dict().items()}
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir,f"{label}_best.pt"))
    model.load_state_dict(best_state)
    P,R,F1,y_true,y_pred = evaluate(model,te_loader,device)
    report = classification_report(y_true,y_pred,digits=4,target_names=[UCI_LABELS[i] for i in sorted(UCI_LABELS)])
    with open(os.path.join(out_dir,f"{label}_report.txt"),"w") as f: f.write(report)
    print(f"[{label}] BEST Macro P={P:.4f} R={R:.4f} F1={F1:.4f}")
    return P,R,F1

def standardize_train_test(Xtr,Xte):
    mu = Xtr.mean(axis=(0,1),keepdims=True)
    std= Xtr.std(axis=(0,1),keepdims=True)+1e-8
    return (Xtr-mu)/std, (Xte-mu)/std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--per-user", action="store_true")
    args = ap.parse_args()
    set_seed(42)

    print("=== Part 2: Deep Learning on raw inertial signals ===")
    train_raw, test_raw = load_raw_windows(args.data_root)
    Xtr, Xte = stack_modalities(train_raw), stack_modalities(test_raw)
    ytr, yte = train_raw.y, test_raw.y
    Xtr,Xte = standardize_train_test(Xtr,Xte)
    out_dir = "outputs/part2"
    if not args.per_user:
        train_and_eval(Xtr,ytr,Xte,yte,args,"aggregate",out_dir)
    else:
        subs = np.unique(test_raw.subjects)
        allP,allR,allF,Ns=[],[],[],[]
        for s in subs:
            tr_mask = train_raw.subjects==s
            te_mask = test_raw.subjects==s
            if tr_mask.sum()==0 or te_mask.sum()==0: continue
            P,R,F1 = train_and_eval(Xtr[tr_mask],ytr[tr_mask],Xte[te_mask],yte[te_mask],args,f"user_{int(s)}",out_dir)
            allP.append(P); allR.append(R); allF.append(F1); Ns.append(te_mask.sum())
        if Ns:
            w=np.array(Ns)/np.sum(Ns)
            print(f"[per-user weighted] P={np.sum(w*allP):.4f} R={np.sum(w*allR):.4f} F1={np.sum(w*allF):.4f}")

if __name__ == "__main__":
    main()
