#!/usr/bin/env python3
"""
使用UMAP降维并可视化两个pkl文件中的激活值
读取两个pkl文件，统一维度后拼接，使用UMAP拟合，然后分别transform生成两张图
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: torch未安装，但可能不需要")


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


def load_last_layer_states_np(pkl_path):
    """
    从pkl文件中加载最后一层的hidden_states
    
    Args:
        pkl_path: pkl文件路径
        
    Returns:
        numpy数组，形状为 (时间步, 特征维度)
    """
    if not os.path.exists(pkl_path):
        print(f"错误: 文件不存在 - {pkl_path}")
        return None
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'hidden_states' not in data:
            print("错误: 未找到hidden_states键")
            return None
        
        hidden_states = data['hidden_states']
        
        if hasattr(hidden_states, 'shape') and len(hidden_states.shape) == 3:
            last_layer_states = hidden_states[:, -1, :]
        elif hasattr(hidden_states, 'shape') and len(hidden_states.shape) == 4:
            last_layer_states = hidden_states[:, -1, -1, :]
        else:
            print(f"错误: 不支持的hidden_states形状: {getattr(hidden_states, 'shape', None)}")
            return None
        
        if TORCH_AVAILABLE and isinstance(last_layer_states, torch.Tensor):
            try:
                return last_layer_states.detach().cpu().numpy()
            except TypeError:
                return last_layer_states.detach().cpu().float().numpy()
        
        return np.asarray(last_layer_states)
    
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


def reduce_dimension_linear(data, target_dim, method='random_projection'):
    """
    将高维数据降维到目标维度
    
    Args:
        data: numpy数组，形状为 (样本数, 特征维度)
        target_dim: 目标维度
        method: 降维方法，'random_projection' 或 'truncated_svd'
        
    Returns:
        降维后的numpy数组，形状为 (样本数, target_dim)
    """
    if data is None:
        return None
    
    current_dim = data.shape[1]
    
    if current_dim == target_dim:
        print(f"维度已匹配，无需降维: {current_dim}")
        return data
    
    if current_dim < target_dim:
        print(f"警告: 当前维度 {current_dim} 小于目标维度 {target_dim}，无法降维")
        return data
    
    print(f"将数据从 {current_dim} 维降维到 {target_dim} 维，使用方法: {method}")
    
    if method == 'random_projection':
        # 使用随机投影（Gaussian Random Projection）
        try:
            from sklearn.random_projection import GaussianRandomProjection
            transformer = GaussianRandomProjection(n_components=target_dim, random_state=42)
            reduced_data = transformer.fit_transform(data)
            print(f"随机投影完成: {data.shape} -> {reduced_data.shape}")
            return reduced_data
        except ImportError:
            print("警告: sklearn未安装，使用简单的随机矩阵投影")
            # 简单的随机矩阵投影
            np.random.seed(42)
            projection_matrix = np.random.randn(current_dim, target_dim) / np.sqrt(current_dim)
            reduced_data = np.dot(data, projection_matrix)
            print(f"随机矩阵投影完成: {data.shape} -> {reduced_data.shape}")
            return reduced_data
    
    elif method == 'truncated_svd':
        # 使用TruncatedSVD（可以处理样本数少于特征数的情况）
        try:
            from sklearn.decomposition import TruncatedSVD
            n_components = min(target_dim, data.shape[0] - 1)  # 确保不超过样本数
            if n_components < target_dim:
                print(f"警告: 样本数 {data.shape[0]} 较少，只能降到 {n_components} 维")
            transformer = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_data = transformer.fit_transform(data)
            # 如果还需要进一步降维，使用随机投影
            if n_components < target_dim:
                return reduce_dimension_linear(reduced_data, target_dim, method='random_projection')
            print(f"TruncatedSVD完成: {data.shape} -> {reduced_data.shape}")
            return reduced_data
        except ImportError:
            print("警告: sklearn未安装，使用随机投影")
            return reduce_dimension_linear(data, target_dim, method='random_projection')
    
    else:
        print(f"错误: 未知的降维方法 {method}")
        return data


def visualize_with_umap_transform(embedding, output_path=None, title="UMAP Visualization", 
                                   n_components=2, data_label="Data"):
    """
    使用UMAP transform后的结果进行可视化
    
    Args:
        embedding: numpy数组，形状为 (时间步, n_components)
        output_path: 输出图片路径
        title: 图片标题
        n_components: 降维后的维度（2或3）
        data_label: 数据标签（用于图例）
    """
    if embedding is None:
        print("错误: embedding为空")
        return
    
    print(f"\n可视化embedding...")
    print(f"Embedding形状: {embedding.shape}")
    
    # 可视化
    if n_components == 2:
        # 2D可视化
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 使用时间步作为颜色
        colors = np.arange(len(embedding))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=colors, cmap='viridis', 
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5,
                           label=data_label)
        
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
        ax.set_title(title, fontsize=14, fontweight='bold')
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
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5,
                           label=data_label)
        
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
        ax.set_title(title, fontsize=14, fontweight='bold')
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


def main():
    parser = argparse.ArgumentParser(description='使用UMAP降维并可视化两个pkl文件')
    parser.add_argument('--pkl_path', type=str, 
                       help='原始pkl文件路径', 
                       default="/root/openpi/data/libero/activations/task_0/episode_0.pkl")
    parser.add_argument('--pkl_path2', type=str,
                       help='第二个pkl文件路径',
                       default="/root/openpi/scripts/task0--ep1--succ1.pkl")
    parser.add_argument('--denoise-steps', type=int, default=10,
                       help='denoise的步数（默认: 10）')
    parser.add_argument('--step-index', type=int, default=-1,
                       help='要可视化的步数索引，-1表示最后一步，0表示第一步（默认: -1）')
    parser.add_argument('--output1', type=str, default=None,
                       help='第一个数据集的输出图片路径（默认: 自动生成）')
    parser.add_argument('--output2', type=str, default=None,
                       help='第二个数据集的输出图片路径（默认: 自动生成）')
    parser.add_argument('--n-neighbors', type=int, default=15,
                       help='UMAP的邻居数（默认: 15）')
    parser.add_argument('--min-dist', type=float, default=0.1,
                       help='UMAP的最小距离（默认: 0.1）')
    parser.add_argument('--n-components', type=int, default=2, choices=[2, 3],
                       help='降维后的维度，2或3（默认: 2）')
    parser.add_argument('--random-state', type=int, default=42,
                       help='随机种子（默认: 42）')
    parser.add_argument('--reduce-method', type=str, default='random_projection',
                       choices=['random_projection', 'truncated_svd'],
                       help='降维方法（默认: random_projection）')
    
    args = parser.parse_args()
    
    # 检查UMAP是否可用
    if not UMAP_AVAILABLE:
        print("错误: 请先安装umap-learn: pip install umap-learn")
        return
    
    # 加载第一个pkl文件（原始文件）
    print(f"\n{'='*60}")
    print(f"加载第一个pkl文件: {args.pkl_path}")
    print(f"{'='*60}")
    data1 = load_pkl_data(args.pkl_path)
    
    if data1 is None:
        return
    
    # 采样denoise步骤
    print(f"\n采样denoise步骤...")
    sampled_data1 = sample_denoise_steps(data1, 
                                        denoise_steps=args.denoise_steps,
                                        step_index=args.step_index)
    
    if sampled_data1 is None:
        return
    
    print(f"第一个数据集形状: {sampled_data1.shape}")
    
    # 加载第二个pkl文件
    print(f"\n{'='*60}")
    print(f"加载第二个pkl文件: {args.pkl_path2}")
    print(f"{'='*60}")
    data2 = load_last_layer_states_np(args.pkl_path2)
    
    if data2 is None:
        return
    
    print(f"第二个数据集形状: {data2.shape}")
    
    # 统一维度
    dim1 = sampled_data1.shape[1]
    dim2 = data2.shape[1]
    
    print(f"\n{'='*60}")
    print(f"统一维度...")
    print(f"第一个数据集维度: {dim1}")
    print(f"第二个数据集维度: {dim2}")
    print(f"{'='*60}")
    
    if dim1 != dim2:
        target_dim = min(dim1, dim2)
        print(f"目标维度: {target_dim}")
        
        if dim1 > target_dim:
            print(f"将第一个数据集从 {dim1} 维降到 {target_dim} 维")
            sampled_data1 = reduce_dimension_linear(sampled_data1, target_dim, args.reduce_method)
        
        if dim2 > target_dim:
            print(f"将第二个数据集从 {dim2} 维降到 {target_dim} 维")
            data2 = reduce_dimension_linear(data2, target_dim, args.reduce_method)
    
    print(f"\n统一维度后:")
    print(f"第一个数据集形状: {sampled_data1.shape}")
    print(f"第二个数据集形状: {data2.shape}")
    
    # 拼接数据
    print(f"\n{'='*60}")
    print(f"拼接数据...")
    print(f"{'='*60}")
    combined_data = np.vstack([sampled_data1, data2])
    print(f"拼接后数据形状: {combined_data.shape}")
    
    # 使用UMAP拟合
    print(f"\n{'='*60}")
    print(f"使用UMAP拟合...")
    print(f"UMAP参数: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, n_components={args.n_components}")
    print(f"{'='*60}")
    
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        random_state=args.random_state,
        verbose=True
    )
    
    # 在拼接数据上拟合
    reducer.fit(combined_data)
    print("UMAP拟合完成")
    
    # 分别transform两个数据集
    print(f"\n{'='*60}")
    print(f"分别transform两个数据集...")
    print(f"{'='*60}")
    
    embedding1 = reducer.transform(sampled_data1)
    print(f"第一个数据集transform完成: {sampled_data1.shape} -> {embedding1.shape}")
    
    embedding2 = reducer.transform(data2)
    print(f"第二个数据集transform完成: {data2.shape} -> {embedding2.shape}")
    
    # 生成输出路径
    if args.output1 is None:
        pkl_name1 = Path(args.pkl_path).stem
        step_suffix = f"_step{args.step_index}" if args.step_index >= 0 else "_laststep"
        output_dir = Path(args.pkl_path).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)
        args.output1 = str(output_dir / f"{pkl_name1}_umap_combined{step_suffix}.png")
    
    if args.output2 is None:
        pkl_name2 = Path(args.pkl_path2).stem
        output_dir2 = Path(args.pkl_path2).parent / "visualizations"
        output_dir2.mkdir(exist_ok=True)
        args.output2 = str(output_dir2 / f"{pkl_name2}_umap_combined.png")
    
    # 可视化第一个数据集
    print(f"\n{'='*60}")
    print(f"可视化第一个数据集...")
    print(f"{'='*60}")
    visualize_with_umap_transform(
        embedding1,
        output_path=args.output1,
        title=f'UMAP Visualization - Dataset 1 (Combined Fit)',
        n_components=args.n_components,
        data_label="Dataset 1"
    )
    
    # 可视化第二个数据集
    print(f"\n{'='*60}")
    print(f"可视化第二个数据集...")
    print(f"{'='*60}")
    visualize_with_umap_transform(
        embedding2,
        output_path=args.output2,
        title=f'UMAP Visualization - Dataset 2 (Combined Fit)',
        n_components=args.n_components,
        data_label="Dataset 2"
    )
    
    print(f"\n{'='*60}")
    print(f"可视化完成！")
    print(f"第一个数据集图片: {args.output1}")
    print(f"第二个数据集图片: {args.output2}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

