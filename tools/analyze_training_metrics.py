#!/usr/bin/env python3
"""
分析训练metrics，可视化性能变化趋势
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze training metrics')
    parser.add_argument('--metrics-file', type=str, 
                        default='output_potsdam/metrics.json',
                        help='Path to metrics.json file')
    parser.add_argument('--output-dir', type=str,
                        default='analysis',
                        help='Output directory for plots')
    return parser.parse_args()


def load_metrics(metrics_file):
    """加载metrics.json文件"""
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except:
                pass
    return metrics


def analyze_metrics(metrics):
    """分析并分类metrics"""
    # 分离训练和验证metrics
    train_metrics = []
    val_metrics = []
    
    for m in metrics:
        if 'sem_seg/mIoU' in m:  # 验证指标
            val_metrics.append(m)
        elif 'loss' in m:  # 训练指标
            train_metrics.append(m)
    
    # 按iteration排序
    train_metrics = sorted(train_metrics, key=lambda x: x.get('iteration', 0))
    val_metrics = sorted(val_metrics, key=lambda x: x.get('iteration', 0))
    
    return train_metrics, val_metrics


def print_metrics_summary(train_metrics, val_metrics):
    """打印metrics摘要"""
    print("=" * 80)
    print("训练Metrics摘要")
    print("=" * 80)
    
    if train_metrics:
        print(f"训练记录数: {len(train_metrics)}")
        print(f"Iteration范围: {train_metrics[0]['iteration']} - {train_metrics[-1]['iteration']}")
    
    print("\n" + "=" * 80)
    print("验证Metrics摘要")
    print("=" * 80)
    
    if val_metrics:
        print(f"验证记录数: {len(val_metrics)}")
        print(f"Iteration范围: {val_metrics[0]['iteration']} - {val_metrics[-1]['iteration']}")
        print(f"\n详细验证结果:")
        print(f"{'Iter':>7s} {'mIoU(%)':>10s} {'mACC(%)':>10s} {'pACC(%)':>10s}")
        print("-" * 40)
        
        for m in val_metrics:
            iter_num = m['iteration']
            miou = m.get('sem_seg/mIoU', 0) * 100
            macc = m.get('sem_seg/mACC', 0) * 100
            pacc = m.get('sem_seg/pACC', 0) * 100
            print(f"{iter_num:7d} {miou:10.4f} {macc:10.4f} {pacc:10.4f}")
        
        # 找到最佳性能
        best_miou_idx = max(range(len(val_metrics)), 
                           key=lambda i: val_metrics[i].get('sem_seg/mIoU', 0))
        best_m = val_metrics[best_miou_idx]
        print(f"\n最佳mIoU: {best_m.get('sem_seg/mIoU', 0)*100:.4f}% @ iteration {best_m['iteration']}")
        
        # 分析性能趋势
        if len(val_metrics) >= 2:
            first = val_metrics[0]
            last = val_metrics[-1]
            miou_change = (last.get('sem_seg/mIoU', 0) - first.get('sem_seg/mIoU', 0)) * 100
            macc_change = (last.get('sem_seg/mACC', 0) - first.get('sem_seg/mACC', 0)) * 100
            
            print(f"\n性能变化 (iter {first['iteration']} -> {last['iteration']}):")
            print(f"  mIoU变化: {miou_change:+.4f}%")
            print(f"  mACC变化: {macc_change:+.4f}%")
            
            if miou_change < -0.1:
                print(f"\n⚠️  mIoU下降 {abs(miou_change):.2f}%，可能存在过拟合!")


def plot_training_curves(train_metrics, val_metrics, output_dir):
    """绘制训练曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 训练loss和学习率曲线
    if train_metrics:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        iters = [m['iteration'] for m in train_metrics]
        losses = [m.get('loss', 0) for m in train_metrics]
        lrs = [m.get('lr', 0) for m in train_metrics]
        
        # Loss曲线
        ax1.plot(iters, losses, 'b-', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Training Loss Curve')
        ax1.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax2.plot(iters, lrs, 'r-', linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'train_loss_lr_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n保存训练曲线: {save_path}")
        plt.close()
    
    # 2. 验证指标曲线
    if val_metrics:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        iters = [m['iteration'] for m in val_metrics]
        mious = [m.get('sem_seg/mIoU', 0) * 100 for m in val_metrics]
        maccs = [m.get('sem_seg/mACC', 0) * 100 for m in val_metrics]
        paccs = [m.get('sem_seg/pACC', 0) * 100 for m in val_metrics]
        
        ax.plot(iters, mious, 'o-', label='mIoU', linewidth=2, markersize=8)
        ax.plot(iters, maccs, 's-', label='mACC', linewidth=2, markersize=8)
        ax.plot(iters, paccs, '^-', label='pACC', linewidth=2, markersize=8)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Validation Metrics Over Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标注最佳点
        best_idx = mious.index(max(mious))
        ax.annotate(f'Best: {mious[best_idx]:.2f}%\n@iter {iters[best_idx]}',
                   xy=(iters[best_idx], mious[best_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'val_metrics_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存验证曲线: {save_path}")
        plt.close()
    
    # 3. 对比iter 4999和69999的性能
    if len(val_metrics) >= 2:
        # 找到这两个checkpoint
        checkpoints = [4999, 69999]
        checkpoint_data = {}
        
        for cp in checkpoints:
            for m in val_metrics:
                if m['iteration'] == cp:
                    checkpoint_data[cp] = m
                    break
        
        if len(checkpoint_data) == 2:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            metrics_names = ['mIoU', 'mACC', 'pACC']
            metric_keys = ['sem_seg/mIoU', 'sem_seg/mACC', 'sem_seg/pACC']
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            values_4999 = [checkpoint_data[4999].get(k, 0) * 100 for k in metric_keys]
            values_69999 = [checkpoint_data[69999].get(k, 0) * 100 for k in metric_keys]
            
            bars1 = ax.bar(x - width/2, values_4999, width, label='Iter 4999', alpha=0.8)
            bars2 = ax.bar(x + width/2, values_69999, width, label='Iter 69999', alpha=0.8)
            
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title('Performance Comparison: Iter 4999 vs 69999', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'checkpoint_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"保存checkpoint对比: {save_path}")
            plt.close()


def main():
    args = parse_args()
    
    print("分析训练Metrics")
    print(f"Metrics文件: {args.metrics_file}")
    print(f"输出目录: {args.output_dir}\n")
    
    # 加载metrics
    metrics = load_metrics(args.metrics_file)
    print(f"加载了 {len(metrics)} 条metrics记录\n")
    
    # 分析metrics
    train_metrics, val_metrics = analyze_metrics(metrics)
    
    # 打印摘要
    print_metrics_summary(train_metrics, val_metrics)
    
    # 绘制曲线
    plot_training_curves(train_metrics, val_metrics, args.output_dir)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
