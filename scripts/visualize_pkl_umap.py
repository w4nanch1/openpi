#!/usr/bin/env python3
"""
使用UMAP降维并可视化pkl文件中的激活值
支持选择denoise步数，只可视化指定步数的数据
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import argparse

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("警告: umap-learn未安装，请运行: pip install umap-learn")


def load_pkl_data(pkl_path):
    """
    加载pkl文件，提取值（numpy数组）
    
    Args:
        pkl_path: pkl文件路径
        
    Returns:
        numpy数组，形状为 (时间步, 1, 特征维度)
    """
    if not os.path.exists(pkl_path):
        print(f"错误: 文件不存在 - {pkl_path}")
        return None
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取值，不管键是什么
        if isinstance(data, dict):
            # 获取第一个值（通常是唯一的）
            values = list(data.values())
            if len(values) > 0:
                array_data = values[0]
                if isinstance(array_data, np.ndarray):
                    return array_data
                else:
                    print(f"错误: 值不是numpy数组，而是 {type(array_data)}")
                    return None
            else:
                print("错误: 字典为空")
                return None
        elif isinstance(data, np.ndarray):
            return data
        else:
            print(f"错误: 不支持的数据类型 {type(data)}")
            return None
    
    except Exception as e:
        print(f"错误: 无法加载pkl文件 - {e}")
        return None


def sample_denoise_steps(data, denoise_steps=10, step_index=-1):
    """
    从denoise步骤中采样数据
    
    Args:
        data: numpy数组，形状为 (时间步*denoise_steps, 1, 特征维度)
        denoise_steps: denoise的步数（默认10）
        step_index: 要选择的步数索引，-1表示最后一步，0表示第一步，等等
        
    Returns:
        采样后的numpy数组，形状为 (时间步, 特征维度)
    """
    if data is None:
        return None
    
    # 确保step_index在有效范围内
    if step_index < 0:
        step_index = denoise_steps + step_index  # -1 -> 9 (最后一步)
    
    if step_index < 0 or step_index >= denoise_steps:
        print(f"错误: step_index {step_index} 超出范围 [0, {denoise_steps-1}]")
        return None
    
    # 获取原始形状
    total_steps, batch_dim, feature_dim = data.shape
    
    # 计算实际的时间步数
    actual_time_steps = total_steps // denoise_steps
    
    if total_steps % denoise_steps != 0:
        print(f"警告: 总步数 {total_steps} 不能被 denoise_steps {denoise_steps} 整除")
        # 截断到最接近的整数倍
        actual_time_steps = total_steps // denoise_steps
        total_steps = actual_time_steps * denoise_steps
        data = data[:total_steps]
    
    # 重塑数据: (时间步, denoise_steps, 1, 特征维度)
    reshaped = data.reshape(actual_time_steps, denoise_steps, batch_dim, feature_dim)
    
    # 选择指定步数的数据: (时间步, 1, 特征维度)
    selected = reshaped[:, step_index, :, :]
    
    # 移除批次维度: (时间步, 特征维度)
    selected = selected.squeeze(1)
    
    print(f"原始数据形状: {data.shape}")
    print(f"采样后数据形状: {selected.shape}")
    print(f"选择了denoise步骤 {step_index} (共{denoise_steps}步)")
    
    return selected


def visualize_with_umap(data, output_path=None, n_neighbors=10, min_dist=0.1, n_components=2, random_state=42):
    """
    使用UMAP降维并可视化
    
    Args:
        data: numpy数组，形状为 (时间步, 特征维度)
        output_path: 输出图片路径
        n_neighbors: UMAP的邻居数
        min_dist: UMAP的最小距离
        n_components: 降维后的维度（2或3）
        random_state: 随机种子
    """
    if not UMAP_AVAILABLE:
        print("错误: umap-learn未安装")
        return None
    
    if data is None:
        print("错误: 数据为空")
        return None
    
    print(f"\n开始UMAP降维...")
    print(f"输入数据形状: {data.shape}")
    print(f"UMAP参数: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}")
    
    # 创建UMAP模型
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        verbose=True
    )
    
    # 降维
    embedding = reducer.fit_transform(data)
    
    print(f"降维后形状: {embedding.shape}")
    
    # 可视化
    if n_components == 2:
        # 2D可视化
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 使用时间步作为颜色
        colors = np.arange(len(embedding))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=colors, cmap='viridis', 
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # 标记起始点
        ax.scatter(embedding[0, 0], embedding[0, 1], 
                  c='green', s=200, marker='o', label='Start',
                  edgecolors='black', linewidth=2, zorder=5)
        
        # 标记结束点
        ax.scatter(embedding[-1, 0], embedding[-1, 1], 
                  c='red', s=200, marker='s', label='End',
                  edgecolors='black', linewidth=2, zorder=5)
        
        # 绘制轨迹线
        ax.plot(embedding[:, 0], embedding[:, 1], 
               'b-', alpha=0.3, linewidth=1, label='Trajectory')
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title('UMAP Visualization of Hidden States', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Time Step')
        
    elif n_components == 3:
        # 3D可视化
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 使用时间步作为颜色
        colors = np.arange(len(embedding))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           c=colors, cmap='viridis',
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # 标记起始点
        ax.scatter(embedding[0, 0], embedding[0, 1], embedding[0, 2],
                  c='green', s=200, marker='o', label='Start',
                  edgecolors='black', linewidth=2)
        
        # 标记结束点
        ax.scatter(embedding[-1, 0], embedding[-1, 1], embedding[-1, 2],
                  c='red', s=200, marker='s', label='End',
                  edgecolors='black', linewidth=2)
        
        # 绘制轨迹线
        ax.plot(embedding[:, 0], embedding[:, 1], embedding[:, 2],
               'b-', alpha=0.3, linewidth=1, label='Trajectory')
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_zlabel('UMAP Dimension 3', fontsize=12)
        ax.set_title('UMAP Visualization of Hidden States (3D)', fontsize=14, fontweight='bold')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Time Step')
    
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return embedding


def main():
    parser = argparse.ArgumentParser(description='使用UMAP降维并可视化pkl文件')
    parser.add_argument('--pkl_path', type=str, help='pkl文件路径', default="/root/openpi/data/libero/activations/task_0/episode_0.pkl")
    parser.add_argument('--denoise-steps', type=int, default=10,
                       help='denoise的步数（默认: 10）')
    parser.add_argument('--step-index', type=int, default=-1,
                       help='要可视化的步数索引，-1表示最后一步，0表示第一步（默认: -1）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（默认: 自动生成）')
    parser.add_argument('--n-neighbors', type=int, default=15,
                       help='UMAP的邻居数（默认: 15）')
    parser.add_argument('--min-dist', type=float, default=0.1,
                       help='UMAP的最小距离（默认: 0.1）')
    parser.add_argument('--n-components', type=int, default=2, choices=[2, 3],
                       help='降维后的维度，2或3（默认: 2）')
    parser.add_argument('--random-state', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    args = parser.parse_args()
    
    # 检查UMAP是否可用
    if not UMAP_AVAILABLE:
        print("错误: 请先安装umap-learn: pip install umap-learn")
        return
    
    # 加载数据
    print(f"加载pkl文件: {args.pkl_path}")
    data = load_pkl_data(args.pkl_path)
    
    if data is None:
        return
    
    # 采样denoise步骤
    print(f"\n采样denoise步骤...")
    sampled_data = sample_denoise_steps(data, 
                                       denoise_steps=args.denoise_steps,
                                       step_index=args.step_index)
    
    if sampled_data is None:
        return
    
    # 生成输出路径
    if args.output is None:
        pkl_name = Path(args.pkl_path).stem
        step_suffix = f"_step{args.step_index}" if args.step_index >= 0 else "_laststep"
        output_dir = Path(args.pkl_path).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"{pkl_name}_umap{step_suffix}.png")
    
    # 可视化
    embedding = visualize_with_umap(sampled_data,
                                    output_path=args.output,
                                    n_neighbors=args.n_neighbors,
                                    min_dist=args.min_dist,
                                    n_components=args.n_components,
                                    random_state=args.random_state)
    
    if embedding is not None:
        print(f"\n可视化完成！")
        print(f"降维后的数据形状: {embedding.shape}")
        print(f"输出图片: {args.output}")


if __name__ == "__main__":
    main()

