#!/usr/bin/env python3
"""
综合诊断脚本：整合所有分析结果
"""

import json
import os
import sys


def analyze_metrics_data():
    """分析metrics.json数据"""
    print("="* 80)
    print("METRICS数据分析")
    print("=" * 80)
    
    metrics_file = 'output_potsdam/metrics.json'
    if not os.path.exists(metrics_file):
        print(f"错误: 找不到 {metrics_file}")
        return
    
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except:
                pass
    
    # 分离训练和验证metrics
    train_metrics = sorted([m for m in metrics if 'loss' in m and 'sem_seg/mIoU' not in m], 
                          key=lambda x: x.get('iteration', 0))
    val_metrics = sorted([m for m in metrics if 'sem_seg/mIoU' in m], 
                        key=lambda x: x.get('iteration', 0))
    
    print(f"\n1. 数据概况:")
    print(f"   - 训练记录数: {len(train_metrics)}")
    print(f"   - 验证记录数: {len(val_metrics)}")
    
    # 验证性能趋势
    print(f"\n2. 验证性能变化:")
    print(f"   {'Iter':>7s} {'mIoU(%)':>10s} {'mACC(%)':>10s} {'pACC(%)':>10s}")
    print("   " + "-" * 42)
    
    for m in val_metrics:
        print(f"   {m['iteration']:7d} {m['sem_seg/mIoU']*100:10.4f} "
              f"{m['sem_seg/mACC']*100:10.4f} {m['sem_seg/pACC']*100:10.4f}")
    
    if len(val_metrics) >= 2:
        best_idx = max(range(len(val_metrics)), key=lambda i: val_metrics[i]['sem_seg/mIoU'])
        best = val_metrics[best_idx]
        first = val_metrics[0]
        last = val_metrics[-1]
        
        print(f"\n3. 关键发现:")
        print(f"   - 最佳mIoU: {best['sem_seg/mIoU']*100:.4f}% @ iteration {best['iteration']}")
        print(f"   - 首次评估 (iter {first['iteration']}): mIoU={first['sem_seg/mIoU']*100:.4f}%")
        print(f"   - 最后评估 (iter {last['iteration']}):  mIoU={last['sem_seg/mIoU']*100:.4f}%")
        
        miou_change = (last['sem_seg/mIoU'] - first['sem_seg/mIoU']) * 100
        macc_change = (last['sem_seg/mACC'] - first['sem_seg/mACC']) * 100
        
        print(f"\n   性能变化:")
        print(f"   - mIoU: {miou_change:+.4f}%")
        print(f"   - mACC: {macc_change:+.4f}%")
        
        if best['iteration'] < last['iteration']:
            print(f"\n   ⚠️  最佳性能出现在iter {best['iteration']}, 早于最后的iter {last['iteration']}")
            print(f"       这表明可能存在过拟合！")  
    
    # 学习率分析
    if train_metrics:
        print(f"\n4. 学习率调度:")
        checkpoints = [999, 4999, 19999, 39999, 59999, 79999]
        print(f"   {'Iter':>7s} {'LR':>15s} {'Loss':>10s}")
        print("   " + "-" * 35)
        
        for cp in checkpoints:
            matches = [m for m in train_metrics if m.get('iteration') == cp]
            if matches:
                m = matches[0]
                print(f"   {m['iteration']:7d} {m.get('lr', 0):15.8f} {m.get('loss', 0):10.6f}")
        
        # 检查学习率是否过小
        last_train = train_metrics[-1]
        if last_train.get('lr', 1) < 1e-5:
            print(f"\n   ⚠️  后期学习率非常小 ({last_train.get('lr', 0):.2e})")
            print(f"       这可能导致模型过拟合到训练集!")


def analyze_config():
    """分析配置文件"""
    print("\n" + "=" * 80)
    print("配置分析")
    print("=" * 80)
    
    config_file = 'output_potsdam/config.yaml'
    if not os.path.exists(config_file):
        config_file = 'configs/uavid/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml'
    
    if not os.path.exists(config_file):
        print("无法找到配置文件")
        return
    
    # 读取关键配置
    import yaml
    with open(config_file) as f:
        if config_file.endswith('.yaml'):
            try:
                cfg = yaml.safe_load(f)
            except:
                print("无法解析YAML配置")
                return
    
    print("\n关键配置参数:")
    
    # 检查dropout
    if 'MODEL' in cfg and 'MASK_FORMER' in cfg['MODEL']:
        dropout = cfg['MODEL']['MASK_FORMER'].get('DROPOUT', 'N/A')
        print(f"  - DROPOUT: {dropout}")
        if dropout == 0.0:
            print(f"    ⚠️  DROPOUT=0，缺少dropout正则化，容易过拟合!")
    
    # 检查weight decay
    if 'SOLVER' in cfg:
        wd = cfg['SOLVER'].get('WEIGHT_DECAY', 'N/A')
        print(f"  - WEIGHT_DECAY: {wd}")
        
        lr = cfg['SOLVER'].get('BASE_LR', 'N/A')
        lr_end = cfg['SOLVER'].get('BASE_LR_END', 'N/A')
        print(f"  - BASE_LR: {lr}")
        print(f"  - BASE_LR_END: {lr_end}")
        
        if lr_end and lr_end < 1e-5:
            print(f"    ⚠️  最终学习率过小 ({lr_end:.2e})，可能导致过拟合!")
    
    # 检查BN
    if 'MODEL' in cfg and 'RESNETS' in cfg['MODEL']:
        norm = cfg['MODEL']['RESNETS'].get('NORM', 'N/A')
        print(f"  - Backbone NORM: {norm}")
        if 'Frozen' in str(norm):
            print(f"    ⚠️  使用FrozenBN，BN统计量冻结，可能影响泛化能力!")


def print_recommendations():
    """打印建议"""
    print("\n" + "=" * 80)
    print("诊断结论与建议")
    print("=" * 80)
    
    print("\n根据分析，模型性能下降的可能原因:")
    print("\n1. **过拟合问题** (最可能)")
    print("   - DROPOUT=0.0，缺少dropout正则化")
    print("   - 后期学习率过小(衰减到1e-6)，导致过拟合训练集")
    print("   - 最佳性能出现在早期iteration")
    
    print("\n2. **Backbone配置问题**")
    print("   - 使用FrozenBatchNorm，训练时BN统计量不更新")
    print("   - 可能导致train/val分布不匹配")
    
    print("\n3. **学习率调度问题**")
    print("   - Cosine衰减在后期学习率过小")
    print("   - 建议调整BASE_LR_END或使用其他scheduler")
    
    print("\n" + "-" * 80)
    print("建议的改进方案:")
    print("-" * 80)
    
    print("\n方案1: 添加Dropout正则化")
    print("  修改配置:")
    print("    MODEL:")
    print("      MASK_FORMER:")
    print("        DROPOUT: 0.1  # 从0.0改为0.1")
    
    print("\n方案2: 调整学习率调度")
    print("  修改配置:")
    print("    SOLVER:")
    print("      BASE_LR_END: 1e-5  # 从1e-6改为1e-5")
    
    print("\n方案3: 使用SyncBN代替FrozenBN")
    print("  修改配置:")
    print("    MODEL:")
    print("      RESNETS:")
    print("        NORM: SyncBN  # 从FrozenBN改为SyncBN")
    
    print("\n方案4: Early Stopping")
    print("  - 基于验证集性能，在iter 4999左右停止训练")
    print("  - 或实施model selection，选择验证集最佳的checkpoint")
    
    print("\n方案5: 增强数据增强")
    print("  - 添加更多数据增强策略(rotation, cutout等)")
    print("  - 增加weight_decay (0.05 -> 0.1)")
    
    print("\n" + "-" * 80)
    print("立即可用的解决方案:")
    print("-" * 80)
    print("\n如果不想重新训练，建议:")
    print("  ✓ 使用model_0004999.pth作为最终模型")
    print("  ✓ 该checkpoint在验证集上性能更好")
    print("  ✓ mIoU=76.94%, mACC=84.77%, 优于model_0069999.pth")


def main():
    print("\n" + "=" * 80)
    print("Potsdam训练性能下降综合诊断报告")
    print("=" * 80)
    
    # 分析metrics
    analyze_metrics_data()
    
    # 分析配置
    analyze_config()
    
    # 打印建议
    print_recommendations()
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
    
    print("\n后续步骤:")
    print("  1. 运行 tools/run_per_class_eval.sh 获取每个类别的详细性能")
    print("  2. 根据建议调整配置并重新训练")
    print("  3. 或直接使用iter 4999的checkpoint")


if __name__ == '__main__':
    main()
