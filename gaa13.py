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


def solve_nearest_neighbor_battery(N=6, t_max=20.0):
    # --- 基础物理参数 ---
    gamma_0 = 1.0
    gamma_nr = 0.001

    # [关键修改 1]: DFS 要求距离为 2 的点干涉相消，因此单步间距必须是 pi/2
    phi_ideal = np.pi / 2

    # --- 算符构建 ---
    id_a = [qt.qeye(2) for _ in range(N)]
    sm_list = [qt.tensor([qt.sigmam() if i == j else qt.qeye(2) for i in range(N)]) for j in range(N)]
    sp_list = [sm.dag() for sm in sm_list]

    # --- 波导矩阵计算 (Gamma & G) ---
    Gamma_mat = np.zeros((N, N))
    G_mat = np.zeros((N, N))

    # [关键修改 2]: 拓扑结构变为 13, 24, 35... 即 [j, j+2]
    atom_points = [[j, j + 2] for j in range(N)]

    for j in range(N):
        for k in range(N):
            for p1 in atom_points[j]:
                for p2 in atom_points[k]:
                    dist = abs(p1 - p2)
                    # 这里的 dist 是坐标索引差，需乘以 phi_ideal 转换为真实相位
                    phase_diff = dist * phi_ideal
                    Gamma_mat[j, k] += gamma_0 * np.cos(phase_diff)
                    if j != k:
                        G_mat[j, k] += (gamma_0 / 2.0) * np.sin(phase_diff)

    print("=" * 50)
    print(f"最近邻编织结构对应的 G 矩阵 (N={N}):")
    print(np.round(G_mat, 3))
    print("=" * 50)

    # --- 构建纯交换哈密顿量与耗散 ---
    H_exchange = sum(G_mat[j, k] * (sp_list[j] * sm_list[k] + sm_list[j] * sp_list[k])
                     for j in range(N) for k in range(j + 1, N))

    c_ops = [np.sqrt(gamma_nr) * sm for sm in sm_list]
    evals, evecs = np.linalg.eigh(Gamma_mat)
    for idx, lam in enumerate(evals):
        if lam > 1e-9:
            L_coll = sum(evecs[i, idx] * sm_list[i] for i in range(N))
            c_ops.append(np.sqrt(lam) * L_coll)

    # --- 初态与目标设定：101010 间隔分布 ---
    psi0_list = [qt.basis(2, 0) if i % 2 == 0 else qt.basis(2, 1) for i in range(N)]
    psi0 = qt.tensor(psi0_list)

    targets = [i for i in range(N) if i % 2 != 0]
    sources = [i for i in range(N) if i % 2 == 0]
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

    print("🚀 正在演化密度矩阵，请稍候...")
    res = qt.mesolve(H_exchange, psi0, tlist, c_ops, e_ops)

    expect_N = res.expect[:N]
    ergo_t = res.expect[N]
    E_targets = np.sum([expect_N[i] for i in targets], axis=0)

    peaks, _ = find_peaks(E_targets)
    p_idx = peaks[0] if len(peaks) > 0 else np.argmax(E_targets)
    P_avg = (E_targets[p_idx] - E_targets[0]) / tlist[p_idx]

    # ==========================================
    # 🔥 绘图
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # 上图：单体原子的演化
    colors_src = plt.cm.Reds(np.linspace(0.4, 0.9, len(sources)))
    colors_tgt = plt.cm.Blues(np.linspace(0.4, 0.9, len(targets)))

    src_idx, tgt_idx = 0, 0
    for i in range(N):
        if i in sources:
            ax1.plot(tlist, expect_N[i], color=colors_src[src_idx], lw=2, ls='--', label=f'Atom {i + 1} (Source)')
            src_idx += 1
        else:
            ax1.plot(tlist, expect_N[i], color=colors_tgt[tgt_idx], lw=2, label=f'Atom {i + 1} (Target)')
            tgt_idx += 1

    ax1.axvline(x=tlist[p_idx], color='k', ls=':', alpha=0.5)
    ax1.set_ylabel('Excitation Probability')
    ax1.set_title('Quantum Random Walk in 1D Nearest-Neighbor Chain', pad=10)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    # 下图：热力学宏观指标
    ax2.plot(tlist, E_targets, 'b-', lw=3, label='Total Battery Energy $E_{act}$')
    ax2.plot(tlist, ergo_t, 'g-', lw=3, label=r'Battery Ergotropy $\mathcal{E}$')
    ax2.fill_between(tlist, ergo_t, E_targets, color='gray', alpha=0.2, label='Bound Energy (Waste)')

    ax2.axvline(x=tlist[p_idx], color='k', ls='--', lw=2)
    ax2.set_title(f'Thermodynamics: Avg Power = {P_avg:.3f}', pad=10)
    ax2.set_xlabel('Time ($1/\gamma_0$)')
    ax2.set_ylabel('Energy / Ergotropy')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('Fig_Nearest_Neighbor_Chain.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# 执行代码
solve_nearest_neighbor_battery(N=8, t_max=20.0)