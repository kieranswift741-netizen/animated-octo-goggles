import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
import time

config = {
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 12,
    "axes.labelsize": 14, "axes.titlesize": 15, "axes.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "legend.frameon": False, "figure.dpi": 300,
}
mpl.rcParams.update(config)


def generate_stratified_configs(N, num_samples):
    pi = np.pi
    pool_configs = []
    pool_chis = []

    print("⏳ 正在进行海量预采样与拓扑空间探索...")
    # 1. 暴力生成 15000 个合法拓扑，使用混合极端分布
    for _ in range(15000):
        rand_type = np.random.rand()
        if rand_type < 0.2:
            # 策略A：均匀随机 (产生低 \chi)
            x_raw = np.random.uniform(0.05, pi - 0.05, N - 1)
        elif rand_type < 0.6:
            # 策略B：Beta 分布 (把原子强行往两端推，产生高 \chi)
            x_raw = np.random.beta(0.1, 0.1, N - 1) * pi
        else:
            # 策略C：随机簇状聚集 (人为制造聚类中心，产生极高 \chi)
            num_clusters = np.random.randint(1, 3)
            centers = np.random.uniform(0.1, pi - 0.1, num_clusters)
            x_raw = np.array([np.random.choice(centers) + np.random.normal(0, 0.05) for _ in range(N - 1)])

        x_raw = np.sort(x_raw)
        x_raw = np.clip(x_raw, 0.02, pi - 0.02)

        # 解决碰撞：允许极度拥挤，但保留最小物理间隔 0.01
        for i in range(1, N - 1):
            if x_raw[i] - x_raw[i - 1] < 0.01:
                x_raw[i] = x_raw[i - 1] + 0.01

        if x_raw[-1] < pi - 0.01:  # 确保没有越出右边界
            # 极速计算 \chi
            x_full = np.concatenate(([0.0], x_raw))
            g_vals = [np.sin(x_full[k] - x_full[j]) for j in range(N) for k in range(j + 1, N)]
            chi = np.std(g_vals) / np.mean(g_vals)
            pool_configs.append(x_raw)
            pool_chis.append(chi)

    pool_configs = np.array(pool_configs)
    pool_chis = np.array(pool_chis)

    min_chi, max_chi = np.min(pool_chis), np.max(pool_chis)
    print(f"✅ 成功构建拓扑池！捕获 \chi 极限范围: [{min_chi:.3f}, {max_chi:.3f}]")

    # 2. 将 \chi 范围均匀切分，进行分层抽样 (Stratified Sampling)
    selected_configs = []
    selected_chis = []

    # 强制加入绝对均匀分布的基准点
    uniform_x = np.linspace(pi / N, (N - 1) * pi / N, N - 1)
    selected_configs.append(uniform_x)

    bins = np.linspace(min_chi, max_chi, num_samples)
    for i in range(num_samples - 1):
        idx_in_bin = np.where((pool_chis >= bins[i]) & (pool_chis < bins[i + 1]))[0]
        if len(idx_in_bin) > 0:
            chosen_idx = np.random.choice(idx_in_bin)
            selected_configs.append(pool_configs[chosen_idx])
            selected_chis.append(pool_chis[chosen_idx])

    return selected_configs


def scan_topology_space(N=6, num_samples=25, t_max=15.0):
    print(f"🚀 开始拓扑相空间扫描 (N={N})，目标样本数 {num_samples} ...")

    # 获取经过完美分层的构型
    configs = generate_stratified_configs(N, num_samples)
    actual_samples = len(configs)

    gamma_0 = 1.0
    gamma_nr = 0.001

    id_a = [qt.qeye(2) for _ in range(N)]
    sm_list = [qt.tensor([qt.sigmam() if i == j else qt.qeye(2) for i in range(N)]) for j in range(N)]
    sp_list = [sm.dag() for sm in sm_list]
    c_ops = [np.sqrt(gamma_nr) * sm for sm in sm_list]

    psi0_list = [qt.basis(2, 0) if i % 2 == 0 else qt.basis(2, 1) for i in range(N)]
    psi0 = qt.tensor(psi0_list)

    targets = [i for i in range(N) if i % 2 != 0]
    N_tgt = len(targets)
    H_B_tgt = sum(qt.tensor([qt.sigmap() * qt.sigmam() if i == j else qt.qeye(2)
                             for i in range(N_tgt)]) for j in range(N_tgt))
    E_n_asc = np.sort(np.real(H_B_tgt.eigenenergies()))

    def calc_ergo(t, rho_or_psi):
        rho_B = rho_or_psi.ptrace(targets)
        r_n_desc = np.sort(np.real(rho_B.eigenenergies()))[::-1]
        E_pass = np.sum(r_n_desc * E_n_asc)
        E_act = np.real(qt.expect(H_B_tgt, rho_B))
        return E_act - E_pass

    results_chi = []
    results_P_avg = []
    results_Ergo = []
    results_E_max = []

    tlist = np.linspace(0, t_max, 500)

    start_time = time.time()
    for idx, x_inner in enumerate(configs):
        x_full = np.concatenate(([0.0], x_inner))

        G_mat = np.zeros((N, N))
        g_vals = []
        for j in range(N):
            for k in range(j + 1, N):
                g = gamma_0 * np.sin(x_full[k] - x_full[j])
                G_mat[j, k] = g
                g_vals.append(g)

        chi = np.std(g_vals) / np.mean(g_vals)
        H_exchange = sum(G_mat[j, k] * (sp_list[j] * sm_list[k] + sm_list[j] * sp_list[k])
                         for j in range(N) for k in range(j + 1, N))

        e_ops = [sp * sm for sp, sm in zip(sp_list, sm_list)]
        e_ops.append(calc_ergo)

        # 使用多线程安全且较快的算子
        res = qt.mesolve(H_exchange, psi0, tlist, c_ops, e_ops, options=qt.Options(nsteps=5000))

        expect_N = res.expect[:N]
        ergo_t = res.expect[N]
        E_targets = np.sum([expect_N[i] for i in targets], axis=0)

        peaks, _ = find_peaks(E_targets)
        peak_idx = peaks[0] if len(peaks) > 0 else np.argmax(E_targets)

        t_peak = tlist[peak_idx]
        E_peak = E_targets[peak_idx]
        P_avg = (E_peak - E_targets[0]) / t_peak if t_peak > 0 else 0
        Ergo_peak = ergo_t[peak_idx]

        results_chi.append(chi)
        results_P_avg.append(P_avg)
        results_Ergo.append(Ergo_peak)
        results_E_max.append(E_peak)

        print(f"[{idx + 1}/{actual_samples}] 演化完毕 | \u03c7={chi:.3f} | P_avg={P_avg:.3f} | Ergo={Ergo_peak:.3f}")

    print(f"✅ 全谱段扫描完成！总耗时: {time.time() - start_time:.1f} 秒")

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sc1 = ax1.scatter(results_chi, results_P_avg, c=results_E_max, cmap='plasma', s=120, edgecolor='k', alpha=0.8)
    ax1.set_xlabel(r'Coupling Dispersion Index ($\chi$)')
    ax1.set_ylabel(r'Average Charging Power ($P_{avg}$)')
    ax1.set_title('Charging Power Collapse at High Dispersion')
    ax1.grid(True, linestyle='--', alpha=0.5)

    sc2 = ax2.scatter(results_chi, results_Ergo, c=results_E_max, cmap='plasma', s=120, edgecolor='k', alpha=0.8)
    ax2.set_xlabel(r'Coupling Dispersion Index ($\chi$)')
    ax2.set_ylabel(r'Peak Ergotropy ($\mathcal{E}_{peak}$)')
    ax2.set_title('Thermodynamic Purity Loss at High Dispersion')
    ax2.grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], fraction=0.02, pad=0.04)
    cbar.set_label(r'Battery Capacity ($E_{max}$)')

    # 添加趋势线引导视觉
    z1 = np.polyfit(results_chi, results_P_avg, 2)
    p1 = np.poly1d(z1)
    x_line = np.linspace(min(results_chi), max(results_chi), 100)
    ax1.plot(x_line, p1(x_line), 'k--', alpha=0.6)

    z2 = np.polyfit(results_chi, results_Ergo, 2)
    p2 = np.poly1d(z2)
    ax2.plot(x_line, p2(x_line), 'k--', alpha=0.6)

    plt.suptitle(f'Full Spectrum Topological Scan of Quantum Battery (N={N})', fontsize=17, y=1.03)
    plt.savefig('Fig_Full_Topology_Scan.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# 运行代码
scan_topology_space(N=6, num_samples=1000)