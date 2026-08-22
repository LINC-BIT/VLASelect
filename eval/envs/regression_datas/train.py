import numpy as np
import pandas as pd
import glob

def load_all_data(path):
    # files = sorted(glob.glob(glob_path))

    A_list, B_list, R_list, Y_list, D_List, P_List = [], [], [], [], [], []

    # for f in files:
    df = pd.read_csv(path)

    A_list.append(df["A"].values)
    B_list.append(df["B"].values)
    R_list.append(df["roll"].values)
    Y_list.append(df["yaw"].values)
    D_List.append(df["distal"].values)
    P_List.append(df["proximal"].values)

    A = np.concatenate(A_list)
    B = np.concatenate(B_list)
    roll = np.concatenate(R_list)
    yaw = np.concatenate(Y_list)
    distal = np.concatenate(D_List)
    proximal = np.concatenate(P_List)

    X = np.stack([A, B], axis=1)
    # Y = np.stack([roll, yaw, distal, proximal], axis=1)
    Y = np.stack([roll, yaw], axis=1)

    return X, Y

import torch
import torch.nn as nn

class F_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def encode(self, x):
        return torch.cat([
            x,
            torch.sin(x),
            torch.cos(x),
            torch.sin(2*x),
            torch.cos(2*x)
        ], dim=-1)

    def forward(self, x):
        x = self.encode(x)
        return self.net(x)
    
class G_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),   # ⭐ 改这里
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
    
def train_f(model, X, Y, epochs=10000, lr=1e-3, device="cuda"):

    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for i in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, Y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 200 == 0:
            print(f"[f] epoch {i}, loss = {loss.item():.6f}")

def build_g_input_output(X, Y):
    """
    X: (A,B)
    Y: (roll,yaw, distal)

    return:
        X_g_input: (roll,yaw, distal,A_{t-1},B_{t-1})
        Y_g_target: (A,B)
    """

    N = len(X)

    A_prev = np.zeros((N, 1))
    B_prev = np.zeros((N, 1))

    A_prev[1:] = X[:-1, 0:1]
    B_prev[1:] = X[:-1, 1:2]

    X_g = np.concatenate([
        Y,        # roll, yaw, distal
        A_prev,
        B_prev
    ], axis=1)

    return X_g, X

def train_g(model, X, Y, epochs=5000, lr=1e-3, device="cuda"):

    X_g, Y_g = build_g_input_output(X, Y)

    X_g = torch.tensor(X_g, dtype=torch.float32).to(device)
    Y_g = torch.tensor(Y_g, dtype=torch.float32).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for i in range(epochs):

        pred = model(X_g)
        loss = loss_fn(pred, Y_g)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 200 == 0:
            print(f"[g] epoch {i}, loss = {loss.item():.6f}")

def cycle_consistency_loss(f, g, X):
    Y = f(X)
    X_recon = g(Y)

    return torch.mean((X_recon - X) ** 2)

if __name__ == "__main__":
    name='finger4'
    X, Y = load_all_data(f"envs/regression_datas/{name}.csv")
    device = 'cuda'

    f_model = F_Model().to(device)
    g_model = G_Model().to(device)

    # -------- train f --------
    train_f(f_model, X, Y, device=device)

    # -------- train g --------
    train_g(g_model, X, Y, device=device)

    # -------- test cycle --------
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    A_prev = np.zeros((len(X), 1))
    B_prev = np.zeros((len(X), 1))

    A_prev[1:] = X[:-1, 0:1]
    B_prev[1:] = X[:-1, 1:2]

    Y_g_test = np.concatenate([Y, A_prev, B_prev], axis=1)

    Y_g_test = torch.tensor(Y_g_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        Y_pred = f_model(X_t)

        A_prev = torch.zeros_like(X_t[:, 0:1])
        B_prev = torch.zeros_like(X_t[:, 1:2])

        A_prev[1:] = X_t[:-1, 0:1]
        B_prev[1:] = X_t[:-1, 1:2]

        Y_pred_g = torch.cat([Y_pred, A_prev, B_prev], dim=1)
        
        X_recon = g_model(Y_pred_g)
        
        X_pred = g_model(Y_g_test)

        error = torch.mean((X_recon - X_t) ** 2)
    
    Y_err = torch.tensor(Y).to(device) - Y_pred

    err = np.abs(Y_err.cpu().numpy())  # (N,2)
    err = err.mean(axis=1)

    r = np.max(np.abs(X), axis=1)  # 距离边界的指标

    bins = [0, 0.5, 1.0, 1.5, 2.0]
    for i in range(len(bins)-1):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if mask.sum() > 0:
            print(bins[i], bins[i+1], err[mask].mean(), mask.sum())

    
    X_err = torch.tensor(X).to(device) - X_pred
    print(Y_err.abs().max())
    print(Y_err.abs().mean())
    print(X_err.abs().max())
    print("cycle reconstruction error:", error.item())

    torch.save(f_model.state_dict(), f"envs/regression_datas/{name}_f.pth")
    torch.save(g_model.state_dict(), f"envs/regression_datas/{name}_g.pth")