import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks

# ==========================================
# 设置绘图格式
# ==========================================
config = {
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 12,
    "axes.labelsize": 14, "axes.titlesize": 15, "axes.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "legend.frameon": False, "figure.dpi": 300,
}
mpl.rcParams.update(config)


def solve_pure_exchange_battery_with_coords(N=6, coords=None, t_max=15.0):
    """
    coords: 长度为 N 的数组，代表每个原子第一个耦合点的绝对 X 坐标。
    """
    if coords is None:
        # 默认回退到均匀排布
        coords = np.linspace(0, (N - 1) * np.pi / N, N)

    # --- 基础物理参数 ---
    gamma_0 = 1.0
    gamma_nr = 0.001

    # --- 算符构建 ---
    id_a = [qt.qeye(2) for _ in range(N)]
    sm_list = [qt.tensor([qt.sigmam() if i == j else qt.qeye(2) for i in range(N)]) for j in range(N)]
    sp_list = [sm.dag() for sm in sm_list]

    # --- [关键修改]: 根据输入的真实坐标计算矩阵 ---
    Gamma_mat = np.zeros((N, N))
    G_mat = np.zeros((N, N))

    # 每个原子两个耦合点：x_j 和 x_j + pi (维持 DFS 条件)
    atom_points = [[coords[j], coords[j] + np.pi] for j in range(N)]

    for j in range(N):
        for k in range(N):
            for p1 in atom_points[j]:
                for p2 in atom_points[k]:
                    dist = abs(p1 - p2)
                    # 此时 dist 直接就是相位差
                    Gamma_mat[j, k] += gamma_0 * np.cos(dist)
                    if j != k:
                        G_mat[j, k] += (gamma_0 / 2.0) * np.sin(dist)

    print("=" * 50)
    print(f"输入坐标对应的 G 矩阵 (N={N}):")
    print(np.round(G_mat, 3))
    print("=" * 50)

    # --- 构建哈密顿量与耗散 ---
    H_exchange = sum(G_mat[j, k] * (sp_list[j] * sm_list[k] + sm_list[j] * sp_list[k])
                     for j in range(N) for k in range(j + 1, N))

    c_ops = [np.sqrt(gamma_nr) * sm for sm in sm_list]
    evals, evecs = np.linalg.eigh(Gamma_mat)
    for idx, lam in enumerate(evals):
        if lam > 1e-9:
            L_coll = sum(evecs[i, idx] * sm_list[i] for i in range(N))
            c_ops.append(np.sqrt(lam) * L_coll)

    # --- 初态与目标设定 ---
    psi0_list = [qt.basis(2, 0) if i % 2 == 0 else qt.basis(2, 1) for i in range(N)]
    psi0 = qt.tensor(psi0_list)
    targets = [i for i in range(N) if i % 2 != 0]
    N_tgt = len(targets)
    H_B_tgt = sum(qt.tensor([qt.sigmap() * qt.sigmam() if i == j else qt.qeye(2)
                             for i in range(N_tgt)]) for j in range(N_tgt))
    E_n_asc = np.sort(np.real(H_B_tgt.eigenenergies()))

    def calc_ergo(t, rho_or_psi):
        rho_B = rho_or_psi.ptrace(targets)
        r_desc = np.sort(np.real(rho_B.eigenenergies()))[::-1]
        E_pass = np.sum(r_desc * E_n_asc)
        E_act = np.real(qt.expect(H_B_tgt, rho_B))
        return E_act - E_pass

    # --- 演化 ---
    tlist = np.linspace(0, t_max, 1000)
    e_ops = [sp * sm for sp, sm in zip(sp_list, sm_list)] + [calc_ergo]

    res = qt.mesolve(H_exchange, psi0, tlist, c_ops, e_ops)
    expect_N = res.expect[:N]
    ergo_t = res.expect[N]
    E_targets = np.sum([expect_N[i] for i in targets], axis=0)

    # 峰值提取
    peaks, _ = find_peaks(E_targets)
    p_idx = peaks[0] if len(peaks) > 0 else np.argmax(E_targets)
    P_avg = (E_targets[p_idx] - E_targets[0]) / tlist[p_idx]

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(tlist, E_targets, 'b-', lw=3, label='Total Energy $E_{act}$')
    plt.plot(tlist, ergo_t, 'g-', lw=3, label=r'Ergotropy $\mathcal{E}$')
    plt.fill_between(tlist, ergo_t, E_targets, color='gray', alpha=0.2, label='Bound Energy (Waste)')

    plt.axvline(x=tlist[p_idx], color='k', ls='--')
    plt.title(f'Evolution of Best Topology: P_avg={P_avg:.3f}', pad=15)
    plt.xlabel('Time')
    plt.ylabel('Energy / Ergotropy')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# ⚡ 调用示例：将你得到的星型坐标放入
# ==========================================
best_x_star = [0.  ,   1.7199 ,1.7299 ,1.7399 ,1.7499 ,1.7599 ,1.7699]
solve_pure_exchange_battery_with_coords(N=7, coords=best_x_star)