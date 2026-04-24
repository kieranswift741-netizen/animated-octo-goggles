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


def solve_pure_exchange_battery(N=8, t_max=20.0):
    # --- 基础物理参数 ---
    gamma_0 = 1.0  # 定义波导自发辐射率基准
    gamma_nr = 0.001  # 极小的非辐射损耗
    phi_ideal = np.pi / N  # 全联通网络实现 DFS 的完美暗态相位

    # --- 算符构建 (纯 Qubit 空间) ---
    id_a = [qt.qeye(2) for _ in range(N)]
    sm_list = [qt.tensor([qt.sigmam() if i == j else qt.qeye(2) for i in range(N)]) for j in range(N)]
    sp_list = [sm.dag() for sm in sm_list]

    # --- 波导矩阵计算 (Gamma & G) ---
    Gamma_mat = np.zeros((N, N))
    G_mat = np.zeros((N, N))
    atom_points = [[j, j + N] for j in range(N)]

    for j in range(N):
        for k in range(N):
            for n in atom_points[j]:
                for m in atom_points[k]:
                    d = abs(n - m)
                    Gamma_mat[j, k] += gamma_0 * np.cos(d * phi_ideal)
                    if j != k:
                        G_mat[j, k] += (gamma_0 / 2.0) * np.sin(d * phi_ideal)

    print("=" * 50)
    print(f"Exchange Interaction Matrix G (N={N})")
    print("=" * 50)
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

    # --- 初态设定：间隔激发 (1, 0, 1, 0, ...) ---
    psi0_list = [qt.basis(2, 0) if i % 2 == 0 else qt.basis(2, 1) for i in range(N)]
    psi0 = qt.tensor(psi0_list)

    # ==========================================
    # 🔥 新增功能 2: Ergotropy (\mathcal{E}) 的动态计算
    # ==========================================
    # 定义目标电池子系统 (Target Atoms: 奇数索引 1, 3, 5, 7)
    targets = [i for i in range(N) if i % 2 != 0]
    N_tgt = len(targets)

    # 在电池的局部子空间 (2^4 = 16维) 中构建哈密顿量 H_B
    H_B_tgt = sum(qt.tensor([qt.sigmap() * qt.sigmam() if i == j else qt.qeye(2)
                             for i in range(N_tgt)]) for j in range(N_tgt))

    # 预先计算并排序 H_B 的本征能量 (升序)
    E_n_asc = np.sort(np.real(H_B_tgt.eigenenergies()))

    # QuTiP 回调函数: 实时截取系统密度矩阵计算 Ergotropy
    def calc_ergo(t, rho_or_psi):
        # 偏迹操作 (Partial Trace)，提取目标电池的约化密度矩阵 \rho_B
        rho_B = rho_or_psi.ptrace(targets)
        # 计算 \rho_B 的本征值，并降序排列 (r_0 >= r_1 >= ...)
        r_n_desc = np.sort(np.real(rho_B.eigenenergies()))[::-1]

        # 计算被动状态的能量 (Passive Energy)
        E_pass = np.sum(r_n_desc * E_n_asc)
        # 电池此刻包含的实际总能量
        E_act = np.real(qt.expect(H_B_tgt, rho_B))

        # Ergotropy = 实际能量 - 被动能量
        return E_act - E_pass

    # --- 演化 ---
    tlist = np.linspace(0, t_max, 1000)
    e_ops = [sp * sm for sp, sm in zip(sp_list, sm_list)]
    e_ops.append(calc_ergo)  # 将 Ergotropy 回调函数挂载进观测列表

    print("🚀 正在演化密度矩阵并实时提取 Ergotropy (N=8，维度256，请稍候)...")
    res = qt.mesolve(H_exchange, psi0, tlist, c_ops, e_ops)

    # 提取演化数据
    expect_N = res.expect[:N]
    ergo_t = res.expect[N]

    # ==========================================
    # 🔥 新增功能 1: 寻找第一个峰值，并计算平均功率
    # ==========================================
    # 计算目标电池吸收的总能量
    E_targets = np.sum([expect_N[i] for i in targets], axis=0)

    # 依靠 scipy 精确捕获能量波峰
    peaks, _ = find_peaks(E_targets)
    peak_idx = peaks[0] if len(peaks) > 0 else np.argmax(E_targets)

    t_peak = tlist[peak_idx]
    E_peak = E_targets[peak_idx]
    P_avg = (E_peak - E_targets[0]) / t_peak if t_peak > 0 else 0

    print(f"\n⚡ 量子电池性能评估 (N={N}):")
    print(f"  第一个能量波峰时间 : t_peak = {t_peak:.2f}")
    print(f"  波峰时电池吸收能量 : E_max  = {E_peak:.3f}")
    print(f"  波峰时可提取功(\u2130) : E_ergo = {ergo_t[peak_idx]:.3f}")
    print(f"  平均充电功率(P_avg): {P_avg:.3f} (\u03b3_0)")

    # ==========================================
    # 绘图展示
    # ==========================================
    plt.figure(figsize=(10, 6))

    # 1. 绘制单体原子概率 (细线)
    colors_src = plt.cm.Reds(np.linspace(0.4, 0.9, N // 2))
    colors_tgt = plt.cm.Blues(np.linspace(0.4, 0.9, N // 2))
    for i in range(N):
        if i % 2 == 0:
            plt.plot(tlist, expect_N[i], color=colors_src[i // 2], lw=1.2, ls='--', alpha=0.5)
        else:
            plt.plot(tlist, expect_N[i], color=colors_tgt[i // 2], lw=1.2, alpha=0.5)

    # 2. 绘制电池总能量 vs Ergotropy (加粗实线对比)
    plt.plot(tlist, E_targets, 'b-', lw=3, label='Battery Total Energy ($E_{act}$)')
    plt.plot(tlist, ergo_t, 'g-', lw=3, label=r'Battery Ergotropy ($\mathcal{E}$)')

    # 3. 标注峰值信息
    plt.axvline(x=t_peak, color='k', ls=':', lw=2)
    plt.plot(t_peak, E_peak, 'k*', markersize=12)

    # 画一条虚线展示 Energy 与 Ergotropy 在峰值处的 GAP
    plt.plot([t_peak, t_peak], [ergo_t[peak_idx], E_peak], 'k-', lw=1.5, alpha=0.5)

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)
    plt.text(t_peak + 0.5, E_peak - 0.2,
             f'$P_{{avg}} = {P_avg:.3f}$\n$Gap = {E_peak - ergo_t[peak_idx]:.3f}$',
             fontsize=11, bbox=bbox_props)

    plt.title(f'Thermodynamics of Topological Battery (N={N})', pad=15)
    plt.xlabel(r'Time ($1/\gamma_0$)')
    plt.ylabel(r'Excitation / Energy Units')
    plt.ylim(0, N / 2 + 0.2)
    plt.legend(loc='upper right', ncol=1, fontsize=11)
    plt.tight_layout()
    #plt.savefig(f'Fig_Battery_Thermodynamics_N{N}.pdf', format='pdf')
    plt.show()


# 执行代码
solve_pure_exchange_battery(N=8, t_max=15.0)