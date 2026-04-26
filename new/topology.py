import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks

# 设置绘图格式
config = {
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 12,
    "axes.labelsize": 14, "axes.titlesize": 15, "axes.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "legend.frameon": False, "figure.dpi": 300,
}
mpl.rcParams.update(config)


def solve_pure_exchange_battery_with_coords(N=6, coords=None, t_max=15.0, excitation_mode='alternating'):
    """
    接受自定义坐标的量子电池模拟
    
    参数:
        N: 原子数量
        coords: 长度为 N 的数组，代表每个原子第一个耦合点的绝对 X 坐标
        t_max: 演化时间
        excitation_mode: 激发模式 ('alternating' 或 'block')
    """
    if coords is None:
        coords = np.linspace(0, (N - 1) * np.pi / N, N)

    # 基础物理参数
    gamma_0 = 1.0
    gamma_nr = 0.001

    # 构建量子算符
    id_a = [qt.qeye(2) for _ in range(N)]
    sm_list = [qt.tensor([qt.sigmam() if i == j else qt.qeye(2) for i in range(N)]) for j in range(N)]
    sp_list = [sm.dag() for sm in sm_list]

    # 根据输入的真实坐标计算矩阵
    Gamma_mat = np.zeros((N, N))
    G_mat = np.zeros((N, N))

    atom_points = [[coords[j], coords[j] + np.pi] for j in range(N)]

    for j in range(N):
        for k in range(N):
            for p1 in atom_points[j]:
                for p2 in atom_points[k]:
                    dist = abs(p1 - p2)
                    Gamma_mat[j, k] += gamma_0 * np.cos(dist)
                    if j != k:
                        G_mat[j, k] += (gamma_0 / 2.0) * np.sin(dist)

    print("=" * 50)
    print(f"输入坐标对应的 G 矩阵 (N={N}):")
    print(np.round(G_mat, 3))
    print("=" * 50)

    # 构建哈密顿量与耗散
    H_exchange = sum(G_mat[j, k] * (sp_list[j] * sm_list[k] + sm_list[j] * sp_list[k])
                     for j in range(N) for k in range(j + 1, N))

    c_ops = [np.sqrt(gamma_nr) * sm for sm in sm_list]
    evals, evecs = np.linalg.eigh(Gamma_mat)
    for idx, lam in enumerate(evals):
        if lam > 1e-9:
            L_coll = sum(evecs[i, idx] * sm_list[i] for i in range(N))
            c_ops.append(np.sqrt(lam) * L_coll)

    # 设置初始态和目标电池
    if excitation_mode == 'block':
        # 块状激发：前一半激发，后一半基态
        half_N = N // 2
        psi0_list = [qt.basis(2, 0)] * half_N + [qt.basis(2, 1)] * (N - half_N)
        targets = list(range(half_N, N))
        sources = list(range(half_N))
    else:
        # 交替激发：101010...
        psi0_list = [qt.basis(2, 0) if i % 2 == 0 else qt.basis(2, 1) for i in range(N)]
        targets = [i for i in range(N) if i % 2 != 0]
        sources = [i for i in range(N) if i % 2 == 0]

    psi0 = qt.tensor(psi0_list)
    N_tgt = len(targets)
    N_src = len(sources)

    # 构建目标电池子系统的哈密顿量
    H_B_tgt = sum(qt.tensor([qt.sigmap() * qt.sigmam() if i == j else qt.qeye(2)
                             for i in range(N_tgt)]) for j in range(N_tgt))
    E_n_asc = np.sort(np.real(H_B_tgt.eigenenergies()))

    # 定义可提取功计算函数
    def calc_ergo(t, rho_or_psi):
        rho_B = rho_or_psi.ptrace(targets)
        r_desc = np.sort(np.real(rho_B.eigenenergies()))[::-1]
        E_pass = np.sum(r_desc * E_n_asc)
        E_act = np.real(qt.expect(H_B_tgt, rho_B))
        return E_act - E_pass

    # 执行演化
    tlist = np.linspace(0, t_max, 1000)
    e_ops = [sp * sm for sp, sm in zip(sp_list, sm_list)] + [calc_ergo]

    res = qt.mesolve(H_exchange, psi0, tlist, c_ops, e_ops)
    expect_N = res.expect[:N]
    ergo_t = res.expect[N]
    E_targets = np.sum([expect_N[i] for i in targets], axis=0)

    # 寻找能量峰值
    peaks, _ = find_peaks(E_targets)
    p_idx = peaks[0] if len(peaks) > 0 else np.argmax(E_targets)
    P_avg = (E_targets[p_idx] - E_targets[0]) / tlist[p_idx]

    # 绘图：双层图表 (单体演化 + 热力学指标)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # 上图：单体原子的演化
    colors_src = plt.cm.Reds(np.linspace(0.4, 0.9, N_src))
    colors_tgt = plt.cm.Blues(np.linspace(0.4, 0.9, N_tgt))

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
    ax1.set_title('Evolution of Individual Giant Atoms', pad=10)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    # 下图：热力学宏观指标
    ax2.plot(tlist, E_targets, 'b-', lw=3, label='Total Battery Energy $E_{act}$')
    ax2.plot(tlist, ergo_t, 'g-', lw=3, label=r'Battery Ergotropy $\mathcal{E}$')
    ax2.fill_between(tlist, ergo_t, E_targets, color='gray', alpha=0.2, label='Bound Energy (Waste)')

    ax2.axvline(x=tlist[p_idx], color='k', ls='--', lw=2)
    ax2.set_title(f'Thermodynamics: Avg Power = {P_avg:.3f}', pad=10)
    ax2.set_xlabel(r'Time ($1/\gamma_0$)')
    ax2.set_ylabel('Energy / Ergotropy')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'Fig_Topology_Dynamics_N{N}.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

    return {
        'P_avg': P_avg,
        'E_peak': E_targets[p_idx],
        'Ergo_peak': ergo_t[p_idx],
        't_peak': tlist[p_idx]
    }


# 预定义的最优拓扑坐标（从参数扫描中获得）
# 这些坐标是通过拓扑扫描找到的具有最高功率或ergotropy的配置
best_topology_coords = {
    # N=7 的最优坐标
    7: {
        'max_power': [0.0, 0.4358, 0.4458, 0.4649, 1.7653, 1.7753, 1.7876],
        'max_ergotropy': [0.0, 1.7199, 1.7299, 1.7399, 1.7499, 1.7599, 1.7699]
    },
    # N=8 的最优坐标
    8: {
        'max_power': [0.0, 0.4358, 0.4458, 0.4649, 1.7653, 1.7753, 1.7876, 1.8686],
        'max_ergotropy': [0.0, 1.7199, 1.7299, 1.7399, 1.7499, 1.7599, 1.7699, 1.8686]
    }
}


def get_optimal_topology_coords(N, target='max_power'):
    """
    获取从参数扫描中得到的最优拓扑坐标
    
    参数:
        N: 原子数量
        target: 优化目标 ('max_power' 或 'max_ergotropy')
    
    返回:
        coords: 最优拓扑的坐标数组
    """
    if N in best_topology_coords:
        if target in best_topology_coords[N]:
            return np.array(best_topology_coords[N][target])
        else:
            raise ValueError(f"目标 '{target}' 不存在，可选值: 'max_power', 'max_ergotropy'")
    else:
        # 如果没有预定义的最优坐标，返回均匀分布
        return np.linspace(0, (N - 1) * np.pi / N, N)
