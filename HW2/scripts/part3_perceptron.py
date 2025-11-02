import numpy as np

class Perceptron:
    def __init__(self, n_in, lr=0.1, epochs=50):
        self.w = np.zeros(n_in)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def f(self, x):
        return 1 if np.dot(self.w, x) + self.b >= 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                yhat = self.f(xi)
                err = yi - yhat
                self.w += self.lr * err * xi
                self.b += self.lr * err

    def predict(self, X):
        return np.array([self.f(xi) for xi in X])

def truth_table_2in(func_name):
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    if func_name == "OR": y = np.array([0,1,1,1])
    elif func_name == "AND": y = np.array([0,0,0,1])
    elif func_name == "NAND": y = np.array([1,1,1,0])
    else: raise ValueError("Unknown gate")
    return X, y

def demo_gate(name):
    X, y = truth_table_2in(name)
    p = Perceptron(n_in=2, lr=0.2, epochs=20)
    p.train(X, y)
    yhat = p.predict(X)
    print(f"{name} gate:")
    print("Weights:", p.w, "Bias:", p.b)
    print("Pred:", yhat, "==", y, "\n")
    return p

def xor_from_gates(p_or, p_and, p_nand, X):
    z1 = p_or.predict(X)
    z2 = p_nand.predict(X)
    Z = np.stack([z1, z2], axis=1).astype(float)
    return p_and.predict(Z)

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    por = demo_gate("OR")
    pand = demo_gate("AND")
    pnand = demo_gate("NAND")
    xor_pred = xor_from_gates(por, pand, pnand, X)
    print("XOR via (OR, NAND) -> AND:")
    print("X:", X.astype(int))
    print("Y:", np.array([0,1,1,0]))
    print("Pred:", xor_pred.astype(int))

if __name__ == "__main__":
    main()
