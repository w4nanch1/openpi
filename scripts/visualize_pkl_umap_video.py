#!/usr/bin/env python3
"""
生成包含原始视频和UMAP降维可视化的组合视频
视频每5帧对应1个activation，降维图中同步显示对应的点
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys
from pathlib import Path
import argparse
import cv2

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


def extract_frames_from_video(video_path):
    """
    从视频中提取所有帧
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        帧列表，每个帧是numpy数组
    """
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 - {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    print(f"从视频中提取了 {len(frames)} 帧")
    return frames


def gif_to_video(gif_path, output_video_path, fps=10):
    """
    将GIF文件转换为MP4视频文件
    
    Args:
        gif_path: GIF文件路径
        output_video_path: 输出视频文件路径
        fps: 视频帧率
        
    Returns:
        成功返回True，失败返回False
    """
    if not os.path.exists(gif_path):
        print(f"错误: GIF文件不存在 - {gif_path}")
        return False
    
    print(f"\n将GIF转换为视频...")
    print(f"GIF文件: {gif_path}")
    print(f"输出视频: {output_video_path}")
    
    try:
        # 使用PIL读取GIF帧
        from PIL import Image
        
        # 打开GIF文件
        gif = Image.open(gif_path)
        
        # 获取第一帧的尺寸
        width, height = gif.size
        print(f"GIF尺寸: {width}x{height}, 帧率: {fps}")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while True:
                # 转换为RGB（如果GIF是调色板模式）
                frame_rgb = gif.convert('RGB')
                # 转换为numpy数组
                frame_array = np.array(frame_rgb)
                # 转换RGB到BGR（cv2使用BGR）
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                frame_count += 1
                
                # 移动到下一帧
                gif.seek(gif.tell() + 1)
        except EOFError:
            # GIF结束
            pass
        
        out.release()
        print(f"转换完成: {frame_count} 帧")
        print(f"视频已保存到: {output_video_path}")
        return True
        
    except ImportError:
        print("错误: 需要PIL/Pillow库来读取GIF")
        return False
    except Exception as e:
        print(f"转换GIF到视频时出错: {e}")
        return False


def create_combined_video(
    pkl_path,
    video_path,
    output_path=None,
    denoise_steps=10,
    step_index=-1,
    frames_per_activation=5,
    n_neighbors=5,
    min_dist=0.1,
    random_state=42,
    fps=10,
    generate_video=False,
    ref_pkl_path=None
):
    """
    创建包含原始视频和UMAP降维可视化的组合视频
    
    Args:
        pkl_path: pkl文件路径
        video_path: 视频文件路径
        output_path: 输出视频路径
        denoise_steps: denoise的步数
        step_index: 要选择的denoise步数索引
        frames_per_activation: 每个activation对应的视频帧数（默认5）
        n_neighbors: UMAP的邻居数
        min_dist: UMAP的最小距离
        random_state: 随机种子
        fps: 输出视频的帧率
        generate_video: 是否生成MP4视频文件（默认False，只生成GIF）
        ref_pkl_path: 参考pkl文件路径（可选），如果提供，将在拼接数据上fit UMAP
    """
    if not UMAP_AVAILABLE:
        print("错误: 请先安装umap-learn: pip install umap-learn")
        return None
    
    print(f"=== 开始处理 ===")
    print(f"PKL文件: {pkl_path}")
    print(f"视频文件: {video_path}")
    print(f"每 {frames_per_activation} 帧对应1个activation")
    
    # 保存原始的frames_per_activation值，用于生成文件名
    original_frames_per_activation = frames_per_activation
    
    # 加载pkl数据
    print(f"\n加载pkl数据...")
    data = load_pkl_data(pkl_path)
    if data is None:
        return None
    
    # 采样denoise步骤
    print(f"\n采样denoise步骤...")
    sampled_data = sample_denoise_steps(data, denoise_steps=denoise_steps, step_index=step_index)
    if sampled_data is None:
        return None
    
    # UMAP降维
    print(f"\n进行UMAP降维...")
    print(f"输入数据形状: {sampled_data.shape}")
    
    # 如果提供了ref_pkl_path，加载并拼接数据（使用相同的方式读取）
    if ref_pkl_path is not None:
        print(f"\n加载参考pkl文件: {ref_pkl_path}")
        ref_data_raw = load_pkl_data(ref_pkl_path)
        if ref_data_raw is None:
            raise ValueError(f"无法加载参考pkl文件: {ref_pkl_path}")
        else:
            # 使用相同的方式处理ref数据
            print(f"\n采样参考pkl的denoise步骤...")
            ref_data = sample_denoise_steps(ref_data_raw, denoise_steps=denoise_steps, step_index=step_index)
            if ref_data is None:
                print("警告: 无法处理参考pkl数据，将只在原始数据上fit UMAP")
                ref_data = None
            else:
                print(f"参考数据形状: {ref_data.shape}")
                # 检查维度是否匹配
                if sampled_data.shape[1] != ref_data.shape[1]:
                    print(f"警告: 维度不匹配 - 原始数据: {sampled_data.shape[1]}, 参考数据: {ref_data.shape[1]}")
                    print("将只在原始数据上fit UMAP")
                    ref_data = None
                else:
                    # 拼接数据
                    combined_data = np.vstack([sampled_data, ref_data])
                    print(f"拼接后数据形状: {combined_data.shape}")
    else:
        ref_data = None
        combined_data = sampled_data
    
    # 创建UMAP模型
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=True
    )
    
    # 在拼接数据上fit（如果有ref数据）或只在原始数据上fit
    if ref_data is not None:
        print(f"\n在拼接数据上fit UMAP...")
        reducer.fit(combined_data)
        print("UMAP fit完成")
        # 只在原始数据上transform
        print(f"在原始数据上transform...")
        embedding = reducer.transform(sampled_data)
    else:
        # 只在原始数据上fit和transform
        embedding = reducer.fit_transform(sampled_data)
    
    print(f"降维后形状: {embedding.shape}")
    
    # 提取视频帧
    print(f"\n提取视频帧...")
    video_frames = extract_frames_from_video(video_path)
    if video_frames is None:
        return None
    
    # 计算activation数量
    num_activations = len(embedding)
    num_video_frames = len(video_frames)
    
    print(f"\n数据统计:")
    print(f"  Activation数量: {num_activations}")
    print(f"  原始视频帧数: {num_video_frames}")
    print(f"  每 {frames_per_activation} 帧对应1个activation")
    
    # 如果frames_per_activation > 1，对视频帧进行采样
    if frames_per_activation > 1:
        print(f"\n对视频帧进行采样 (frames_per_activation={frames_per_activation})...")
        # 从每个activation对应的帧组中采样中间帧
        sampled_video_frames = []
        for i in range(num_activations):
            start_idx = i * frames_per_activation
            end_idx = min(start_idx + frames_per_activation, num_video_frames)
            if start_idx < num_video_frames:
                # 取该帧组的中间帧（或最后一帧如果不足）
                frame_group = video_frames[start_idx:end_idx]
                if len(frame_group) > 0:
                    # 取中间帧
                    mid_idx = len(frame_group) // 2
                    sampled_video_frames.append(frame_group[mid_idx])
        
        video_frames = sampled_video_frames
        num_video_frames = len(video_frames)
        print(f"采样后视频帧数: {num_video_frames}")
        
        # 如果采样后的帧数少于activation数量，调整activation数量
        if num_video_frames < num_activations:
            print(f"警告: 采样后视频帧数 ({num_video_frames}) 少于activation数量 ({num_activations})")
            print(f"将调整activation数量以匹配采样后的视频帧数")
            num_activations = num_video_frames
            embedding = embedding[:num_activations]
            print(f"调整后的activation数量: {num_activations}")
        
        # 更新frames_per_activation为1，因为已经采样过了
        frames_per_activation = 1
    else:
        # frames_per_activation == 1，不需要采样
        # 计算实际需要的视频帧数（每个activation对应1帧）
        expected_video_frames = num_activations
        
        if num_video_frames < expected_video_frames:
            print(f"警告: 视频帧数 ({num_video_frames}) 少于预期 ({expected_video_frames})")
            print(f"将使用所有视频帧，最后一个activation可能对应较少的帧")
            # 调整activation数量以匹配视频帧数
            num_activations = num_video_frames
            embedding = embedding[:num_activations]
            print(f"调整后的activation数量: {num_activations}")
        elif num_video_frames > expected_video_frames:
            print(f"警告: 视频帧数 ({num_video_frames}) 多于预期 ({expected_video_frames})")
            print(f"将截断视频帧以匹配activation数量")
            video_frames = video_frames[:expected_video_frames]
            num_video_frames = len(video_frames)
    
    # 创建matplotlib图形（降低分辨率以避免编码问题）
    fig = plt.figure(figsize=(16, 8))
    
    # 左图：原始视频
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Original Video', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 右图：UMAP降维可视化
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('UMAP Visualization', fontsize=16, fontweight='bold')
    ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 设置UMAP图的坐标轴范围
    x_margin = (embedding[:, 0].max() - embedding[:, 0].min()) * 0.1
    y_margin = (embedding[:, 1].max() - embedding[:, 1].min()) * 0.1
    ax2.set_xlim(embedding[:, 0].min() - x_margin, embedding[:, 0].max() + x_margin)
    ax2.set_ylim(embedding[:, 1].min() - y_margin, embedding[:, 1].max() + y_margin)
    
    # 初始化显示对象
    video_display = ax1.imshow(video_frames[0])
    ax1.set_title(f'Frame 0/{num_video_frames-1}', fontsize=14)
    
    # 存储已显示的activation点
    shown_activations = []
    trajectory_line = None
    
    def animate(frame_idx):
        # 更新视频帧
        if frame_idx < len(video_frames):
            video_display.set_array(video_frames[frame_idx])
            ax1.set_title(f'Frame {frame_idx}/{num_video_frames-1}', fontsize=14)
        
        # 计算当前应该显示的activation索引
        # 采样后，每个视频帧对应1个activation（frames_per_activation已设为1）
        activation_idx = frame_idx // frames_per_activation
        
        # 更新UMAP可视化
        ax2.clear()
        ax2.set_title('UMAP Visualization', fontsize=16, fontweight='bold')
        ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(embedding[:, 0].min() - x_margin, embedding[:, 0].max() + x_margin)
        ax2.set_ylim(embedding[:, 1].min() - y_margin, embedding[:, 1].max() + y_margin)
        
        # 显示到当前activation的所有点
        if activation_idx < len(embedding):
            # 绘制轨迹线（到当前activation）
            if activation_idx > 0:
                ax2.plot(embedding[:activation_idx+1, 0], embedding[:activation_idx+1, 1],
                        'b-', alpha=0.3, linewidth=1, label='Trajectory')
            
            # 绘制已显示的所有点（使用时间颜色映射）
            if activation_idx >= 0:
                colors = np.linspace(0.3, 1.0, activation_idx + 1)
                ax2.scatter(embedding[:activation_idx+1, 0], embedding[:activation_idx+1, 1],
                           c=colors, s=50, alpha=0.7, cmap=plt.colormaps['viridis'],
                           edgecolors='black', linewidth=0.5)
            
            # 标记起始点
            if activation_idx >= 0:
                ax2.scatter(embedding[0, 0], embedding[0, 1],
                           c='darkgreen', s=200, marker='o', label='Start',
                           edgecolors='black', linewidth=2, zorder=5)
            
            # 标记当前点（高亮显示）
            if activation_idx >= 0:
                ax2.scatter(embedding[activation_idx, 0], embedding[activation_idx, 1],
                           c='red', s=200, marker='*', label='Current',
                           edgecolors='black', linewidth=2, zorder=6)
            
            # 标记结束点（如果到达最后）
            if activation_idx == len(embedding) - 1:
                ax2.scatter(embedding[-1, 0], embedding[-1, 1],
                           c='darkred', s=200, marker='s', label='End',
                           edgecolors='black', linewidth=2, zorder=5)
        
        ax2.legend()
        
        return video_display,
    
    # 创建动画
    print(f"\n生成动画...")
    print(f"总帧数: {num_video_frames}")
    anim = FuncAnimation(fig, animate, frames=num_video_frames, interval=1000/fps, blit=False)
    
    # 生成输出路径
    if output_path is None:
        # 从pkl_path提取episode编号
        pkl_name = Path(pkl_path).stem
        episode_num = None
        if pkl_name.startswith("episode_"):
            try:
                episode_num = int(pkl_name.split("_")[1])
            except:
                pass
        
        # 从ref_pkl_path提取episode编号（如果有）
        ref_episode_num = None
        if ref_pkl_path is not None:
            ref_pkl_name = Path(ref_pkl_path).stem
            if ref_pkl_name.startswith("episode_"):
                try:
                    ref_episode_num = int(ref_pkl_name.split("_")[1])
                except:
                    pass
        
        # 生成文件名
        step_suffix = "last" if step_index < 0 else f"step{step_index}"
        if episode_num is not None:
            base_name = f"ep{episode_num}_{original_frames_per_activation}frame_{step_suffix}"
        else:
            # 如果无法提取episode编号，使用原始名称
            base_name = f"{pkl_name}_{original_frames_per_activation}frame_{step_suffix}"
        
        # 如果有ref，添加ref后缀
        if ref_episode_num is not None:
            base_name = f"{base_name}_ref_ep{ref_episode_num}"
        elif ref_pkl_path is not None:
            # 如果提供了ref_pkl_path但无法提取编号，使用ref文件名
            ref_name = Path(ref_pkl_path).stem
            base_name = f"{base_name}_ref_{ref_name}"
        
        output_dir = Path(pkl_path).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)
        if generate_video:
            output_path = str(output_dir / f"{base_name}.mp4")
        else:
            output_path = str(output_dir / f"{base_name}.gif")
    
    # 先保存为GIF
    gif_path = output_path
    if generate_video:
        # 如果最终要生成视频，先保存为临时GIF
        gif_path = output_path.replace('.mp4', '.gif')
    
    print(f"保存GIF到: {gif_path}")
    try:
        writer_gif = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer_gif, dpi=150)
        print(f"GIF已保存到: {gif_path}")
    except Exception as e:
        print(f"保存GIF失败: {e}")
        plt.close()
        return None
    
    plt.close()
    
    # 如果需要生成视频，将GIF转换为MP4
    if generate_video:
        if gif_to_video(gif_path, output_path, fps=fps):
            # 可选：删除临时GIF文件（如果用户想要的话，可以保留）
            # os.remove(gif_path)
            return output_path
        else:
            print(f"警告: 视频转换失败，返回GIF路径: {gif_path}")
            return gif_path
    
    return gif_path


def main():
    parser = argparse.ArgumentParser(description='生成包含原始视频和UMAP降维可视化的组合视频')
    parser.add_argument('--pkl_path', type=str, help='pkl文件路径', default="/root/openpi/data/libero/activations/task_0_5action/episode_9.pkl")
    parser.add_argument('--ref-pkl-path', type=str, default="/root/openpi/data/libero/activations/task_0_5action/episode_10.pkl",
                       help='参考pkl文件路径（可选），如果提供，将在拼接数据上fit UMAP，只在原始数据上transform）')
    parser.add_argument('--video-path', type=str, default=None,
                       help='视频文件路径（默认: 根据pkl路径自动查找）')
    parser.add_argument('--denoise-steps', type=int, default=10,
                       help='denoise的步数（默认: 10）')
    parser.add_argument('--step-index', type=int, default=-1,
                       help='要可视化的denoise步数索引，-1表示最后一步（默认: -1）')
    parser.add_argument('--frames-per-activation', type=int, default=5,
                       help='每个activation对应的视频帧数（默认: 5）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（默认: 自动生成）')
    parser.add_argument('--n-neighbors', type=int, default=10,
                       help='UMAP的邻居数（默认: 10）')
    parser.add_argument('--min-dist', type=float, default=0.1,
                       help='UMAP的最小距离（默认: 0.1）')
    parser.add_argument('--random-state', type=int, default=42,
                       help='随机种子（默认: 42）')
    parser.add_argument('--fps', type=int, default=10,
                       help='输出视频的帧率（默认: 10）')
    parser.add_argument('--generate-video', action='store_true',
                       help='是否生成MP4视频文件（默认: False，只生成GIF）')
    
    
    args = parser.parse_args()
    args.fps = int(10 / args.frames_per_activation)
    # 自动查找视频文件
    if args.video_path is None:
        pkl_path_obj = Path(args.pkl_path)
        pkl_dir = pkl_path_obj.parent
        pkl_name = pkl_path_obj.stem
        
        # 查找对应的视频文件（rollout_*.mp4格式）
        video_files = list(pkl_dir.glob("rollout_*.mp4"))
        if video_files:
            # 尝试找到与episode编号匹配的视频
            episode_num = None
            if pkl_name.startswith("episode_"):
                try:
                    episode_num = int(pkl_name.split("_")[1])
                except:
                    pass
            
            if episode_num is not None:
                # 查找包含epoch{episode_num}的视频
                matching_videos = [v for v in video_files if f"epoch{episode_num}" in v.name]
                if matching_videos:
                    args.video_path = str(matching_videos[0])
                else:
                    args.video_path = str(video_files[0])
            else:
                args.video_path = str(video_files[0])
            
            print(f"自动找到视频文件: {args.video_path}")
        else:
            print("错误: 未找到对应的视频文件，请使用--video-path指定")
            return
    
    # 创建组合视频
    output_path = create_combined_video(
        pkl_path=args.pkl_path,
        video_path=args.video_path,
        output_path=args.output,
        denoise_steps=args.denoise_steps,
        step_index=args.step_index,
        frames_per_activation=args.frames_per_activation,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        fps=args.fps,
        generate_video=args.generate_video,
        ref_pkl_path=args.ref_pkl_path
    )
    
    if output_path:
        print(f"\n✓ 完成！输出视频: {output_path}")


if __name__ == "__main__":
    main()

