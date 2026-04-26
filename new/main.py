#!/usr/bin/env python3
"""
量子电池拓扑结构研究主脚本

这个脚本整合了量子电池拓扑结构研究的核心功能，包括：
1. 拓扑相空间扫描
2. 特定拓扑结构模拟
3. 桥接星型结构生成与测试

使用方法：
- python main.py scan [N] [num_samples] [excitation_mode] - 执行拓扑扫描
- python main.py star [N] - 运行星型拓扑模拟
- python main.py bridge [N] [chi] - 运行桥接星型拓扑模拟
"""

import sys
import os
from core import scan_topology_space
from topology import solve_pure_exchange_battery_with_coords, get_optimal_topology_coords


def print_usage():
    print("Usage:")
    print("  python main.py scan [N] [num_samples] [excitation_mode] - 执行拓扑扫描")
    print("  python main.py max_power [N] [excitation_mode] - 运行最大功率拓扑模拟（使用参数扫描获得的最优坐标）")
    print("  python main.py max_ergotropy [N] [excitation_mode] - 运行最大可提取功拓扑模拟（使用参数扫描获得的最优坐标）")
    print("  python main.py custom [N] [excitation_mode] - 运行自定义坐标的拓扑模拟（手动输入之前扫描出的坐标）")
    print("  python main.py help - 显示此帮助信息")
    print("")
    print("参数说明:")
    print("  N: 原子数量 (默认: 8)")
    print("  num_samples: 拓扑抽样数量 (默认: 1000)")
    print("  excitation_mode: 激发模式 ('alternating' 或 'block', 默认: 'block')")


def main():
    # 解析命令和参数
    if len(sys.argv) < 2:
        # 默认执行custom命令
        command = 'custom'
        N = 8
        excitation_mode = 'block'
        print("默认执行自定义坐标拓扑模拟")
    else:
        # 检查第一个参数是否为有效命令
        valid_commands = ['scan', 'max_power', 'max_ergotropy', 'custom', 'help']
        if sys.argv[1] in valid_commands:
            command = sys.argv[1]
            # 解析命令的参数
            if command == 'help':
                print_usage()
                return
            elif command == 'scan':
                N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
                num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
                excitation_mode = sys.argv[4] if len(sys.argv) > 4 else 'block'
            else:  # max_power, max_ergotropy, custom
                N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
                excitation_mode = sys.argv[3] if len(sys.argv) > 3 else 'block'
        else:
            # 第一个参数不是有效命令，默认执行custom命令
            command = 'custom'
            # 解析参数
            if sys.argv[1].isdigit():
                N = int(sys.argv[1])
                excitation_mode = sys.argv[2] if len(sys.argv) > 2 else 'block'
            else:
                # 第一个参数是激发模式
                N = 8
                excitation_mode = sys.argv[1]
                if len(sys.argv) > 2 and sys.argv[2].isdigit():
                    N = int(sys.argv[2])
        
        # 验证激发模式
        if excitation_mode not in ['alternating', 'block']:
            print("错误: excitation_mode 必须是 'alternating' 或 'block'")
            return

    # 执行命令
    if command == 'scan':
        print(f"执行拓扑扫描: N={N}, num_samples={num_samples}, excitation_mode={excitation_mode}")
        scan_topology_space(N=N, num_samples=num_samples, excitation_mode=excitation_mode)
    
    elif command == 'max_power':
        print(f"运行最大功率拓扑模拟: N={N}, excitation_mode={excitation_mode}")
        # 获取从参数扫描中得到的最大功率拓扑坐标
        coords = get_optimal_topology_coords(N, target='max_power')
        print(f"最大功率拓扑坐标: {np.round(coords, 4)}")
        # 运行模拟
        solve_pure_exchange_battery_with_coords(N=N, coords=coords, excitation_mode=excitation_mode)
    
    elif command == 'max_ergotropy':
        print(f"运行最大可提取功拓扑模拟: N={N}, excitation_mode={excitation_mode}")
        # 获取从参数扫描中得到的最大可提取功拓扑坐标
        coords = get_optimal_topology_coords(N, target='max_ergotropy')
        print(f"最大可提取功拓扑坐标: {np.round(coords, 4)}")
        # 运行模拟
        solve_pure_exchange_battery_with_coords(N=N, coords=coords, excitation_mode=excitation_mode)
    
    elif command == 'custom':
        print(f"运行自定义坐标拓扑模拟: N={N}, excitation_mode={excitation_mode}")
        # 提示用户输入坐标
        print("请输入拓扑坐标（用空格分隔，共" + str(N) + "个数值）:")
        coords_input = input().strip()
        coords = np.array([float(x) for x in coords_input.split()])
        
        if len(coords) != N:
            print(f"错误: 坐标数量必须为 {N}，您输入了 {len(coords)} 个")
            return
        
        print(f"自定义拓扑坐标: {np.round(coords, 4)}")
        # 运行模拟
        solve_pure_exchange_battery_with_coords(N=N, coords=coords, excitation_mode=excitation_mode)


if __name__ == "__main__":
    import numpy as np  # 确保numpy在主脚本中可用
    main()
