import numpy as np
import pandas as pd
import glob
from sklearn.decomposition import PCA


# =========================
# 1️⃣ 读取单个文件
# =========================
def load_single(file):
    df = pd.read_csv(file)
    print(f"\nLoaded {file}, shape = {df.shape}")
    return df.values


# =========================
# 2️⃣ 数据质量检查（单文件）
# =========================
def check_data_quality(X, name=""):
    print(f"\n================ {name} ================")

    N, D = X.shape
    print(f"样本数: {N}, 维度: {D}")

    # ---------- 覆盖性 ----------
    print("\n[1] 覆盖范围:")
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    for i in range(D):
        print(f"dim {i}: [{mins[i]:.3f}, {maxs[i]:.3f}]")

    # ---------- 方差 ----------
    print("\n[2] 方差:")
    var = np.var(X, axis=0)
    print(var)

    # ---------- 相关性 ----------
    print("\n[3] 最大相关性:")
    if D > 1:
        corr = np.corrcoef(X.T)
        off_diag = np.abs(corr - np.eye(D))
        max_corr = np.max(off_diag)
        print("max |corr| =", max_corr)
    else:
        print("skip (1D data)")

    # ---------- rank ----------
    print("\n[4] rank:")
    rank = np.linalg.matrix_rank(X)
    print(f"rank = {rank} / {D}")

    # ---------- 条件数 ----------
    print("\n[5] condition number:")
    cond = np.linalg.cond(X)
    print(cond)

    # ---------- 激励 ----------
    print("\n[6] mean |delta|:")
    diff = np.diff(X, axis=0)
    print(np.mean(np.abs(diff)))

    # ---------- PCA ----------
    print("\n[7] PCA:")
    pca = PCA()
    pca.fit(X)
    print(pca.explained_variance_ratio_)

    print("=====================================\n")


# =========================
# 3️⃣ 主流程：逐 finger 检查
# =========================
if __name__ == "__main__":

    files = sorted(glob.glob("envs/regression_datas/finger*.csv"))

    assert len(files) > 0, "未找到 finger*.csv"

    results = {}

    for f in files:
        X = load_single(f)

        # 如果第一列是 time，可以取消注释
        # X = X[:, 1:]

        check_data_quality(X, name=f)

        

        results[f] = X