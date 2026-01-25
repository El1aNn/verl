#!/usr/bin/env python3
"""
合并版本：使用entropy_math重新打分并计算pass@k指标
支持多进程并行处理
"""

import json
import sys
import csv
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加路径
sys.path.insert(0, '/Users/elianxu/RL/verl')

# 导入宽松的评分函数
from recipe.entropy.reward_score.entropy_math import compute_score


def rescore_single_sample(args):
    """打分单个样本（用于并行处理）"""
    line, use_fast = args
    data = json.loads(line.strip())
    
    # 提取信息
    output = data['output']
    ground_truth = data['gts']
    old_score = data.get('score', 0.0)
    
    # 使用宽松的评分函数
    try:
        result = compute_score(output, ground_truth, fast=use_fast)
        
        # 更新数据
        if isinstance(result, dict):
            data['score'] = result['score']
            data['acc'] = result['acc']
            data['format_score'] = result.get('format_score', 1.0)
            data['extracted_gt'] = result.get('extracted_gt', ground_truth)
            new_score = result['score']
        else:
            data['score'] = float(result)
            new_score = float(result)
        
        # 判断变化
        if new_score > old_score:
            change = 'improved'
        elif new_score == old_score:
            change = 'same'
        else:
            change = 'worsened'
            
    except Exception as e:
        # 保持原分数
        data['score'] = old_score
        change = 'same'
    
    return data, change


def rescore_file(input_path, output_path, n_workers=None, use_fast=False):
    """重新打分一个jsonl文件（支持多进程）"""
    with open(input_path, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    score_changes = {'improved': 0, 'same': 0, 'worsened': 0}
    
    if n_workers and n_workers > 1:
        # 多进程处理
        with Pool(processes=n_workers) as pool:
            args = [(line, use_fast) for line in lines]
            results = list(tqdm(
                pool.imap(rescore_single_sample, args),
                total=len(lines),
                desc=f"  打分 {input_path.name}",
                unit="样本",
                leave=False
            ))
        
        rescored_data = []
        for data, change in results:
            rescored_data.append(data)
            score_changes[change] += 1
    else:
        # 单进程处理
        rescored_data = []
        for line in tqdm(lines, desc=f"  打分 {input_path.name}", unit="样本", leave=False):
            data, change = rescore_single_sample((line, use_fast))
            rescored_data.append(data)
            score_changes[change] += 1
    
    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for data in rescored_data:
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    return score_changes


def calculate_pass_at_k(scores, k):
    """
    计算pass@k指标
    scores: 每个问题的得分列表
    k: 取前k个样本
    """
    if len(scores) == 0:
        return 0.0
    
    # pass@k: 至少有一个正确的概率
    correct = any(score > 0 for score in scores[:k])
    return 1.0 if correct else 0.0


def evaluate_file(file_path, n_samples=8):
    """
    评估一个jsonl文件
    假设每个问题有n_samples个样本(连续排列)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
    if len(data) == 0:
        return None
    
    # 提取步数(从文件名)
    step = int(Path(file_path).stem)
    
    # 按问题分组 (假设每n_samples个样本是同一个问题)
    n_problems = len(data) // n_samples
    
    pass_1_scores = []
    pass_8_scores = []
    
    for i in range(n_problems):
        # 获取这个问题的所有样本
        samples = data[i * n_samples: (i + 1) * n_samples]
        
        # 提取分数
        scores = [sample.get('score', 0.0) for sample in samples]
        
        # 计算pass@1 (第一个样本是否正确)
        pass_1 = 1.0 if scores[0] > 0 else 0.0
        pass_1_scores.append(pass_1)
        
        # 计算pass@8 (8个样本中至少有1个正确)
        pass_8 = calculate_pass_at_k(scores, min(8, len(scores)))
        pass_8_scores.append(pass_8)
    
    # 计算平均值
    avg_pass_1 = sum(pass_1_scores) / len(pass_1_scores) if pass_1_scores else 0.0
    avg_pass_8 = sum(pass_8_scores) / len(pass_8_scores) if pass_8_scores else 0.0
    
    return {
        'step': step,
        'n_problems': n_problems,
        'pass@1': avg_pass_1,
        'pass@8': avg_pass_8
    }


def main():
    import argparse
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='重新打分并评估验证数据（支持多进程）')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'并行进程数（默认: CPU核心数-1 = {max(1, cpu_count()-1)}）')
    parser.add_argument('--fast', action='store_true',
                        help='使用快速模式（精度略低但速度更快）')
    parser.add_argument('--input-dir', type=str, 
                        default='/Users/elianxu/RL/verl/val_data/test_data_dir',
                        help='输入目录')
    parser.add_argument('--output-dir', type=str,
                        default='/Users/elianxu/RL/verl/val_data/test_data_dir_rescored',
                        help='输出目录')
    args = parser.parse_args()
    
    # 自动选择进程数
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)  # 保留一个核心给系统
    
    # 配置路径
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_csv = Path('/Users/elianxu/RL/verl/val_data/pass_metrics_entropy.csv')
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有jsonl文件
    jsonl_files = sorted(input_dir.glob('*.jsonl'), key=lambda x: int(x.stem))
    
    print("="*70)
    print("重新打分并评估验证数据（多进程加速版）")
    print("="*70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"评分函数: entropy_math (宽松，支持数学等价性)")
    print(f"CPU核心数: {cpu_count()}")
    print(f"并行进程: {args.workers}")
    print(f"快速模式: {'是' if args.fast else '否'}")
    print(f"找到 {len(jsonl_files)} 个文件\n")
    
    # 阶段1: 重新打分
    print("阶段 1/2: 重新打分")
    print("-" * 70)
    total_changes = {'improved': 0, 'same': 0, 'worsened': 0}
    
    for input_path in tqdm(jsonl_files, desc="总体进度", unit="文件"):
        output_path = output_dir / input_path.name
        
        changes = rescore_file(input_path, output_path, n_workers=args.workers, use_fast=args.fast)
        
        # 累计统计
        for key in total_changes:
            total_changes[key] += changes[key]
        
        tqdm.write(f"  ✓ {input_path.name:<10} 提高:{changes['improved']:>4} 不变:{changes['same']:>4} 降低:{changes['worsened']:>4}")
    
    # 打印重新打分总结
    total = sum(total_changes.values())
    print("\n" + "="*70)
    print("重新打分完成!")
    print("="*70)
    print(f"总样本数: {total:,}")
    print(f"分数提高: {total_changes['improved']:,} ({100*total_changes['improved']/total:.2f}%)")
    print(f"分数不变: {total_changes['same']:,} ({100*total_changes['same']/total:.2f}%)")
    print(f"分数降低: {total_changes['worsened']:,} ({100*total_changes['worsened']/total:.2f}%)")
    
    # 阶段2: 计算指标
    print("\n阶段 2/2: 计算pass@k指标")
    print("-" * 70)
    
    results = []
    
    for file_path in tqdm(output_dir.glob('*.jsonl'), desc="计算指标", unit="文件"):
        result = evaluate_file(file_path)
        
        if result:
            results.append(result)
            tqdm.write(f"  ✓ 步数 {result['step']:<6} pass@1={result['pass@1']:.4f} ({result['pass@1']*100:>5.2f}%)  pass@8={result['pass@8']:.4f} ({result['pass@8']*100:>5.2f}%)")
    
    # 按步数排序
    results.sort(key=lambda x: x['step'])
    
    # 写入CSV
    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'n_problems', 'pass@1', 'pass@8'])
            writer.writeheader()
            writer.writerows(results)
        
        # 打印摘要
        print("\n" + "="*70)
        print("指标计算完成!")
        print("="*70)
        print(f"结果已保存到: {output_csv}")
        print(f"共 {len(results)} 个检查点\n")
        
        # 最佳结果
        best_pass1 = max(results, key=lambda x: x['pass@1'])
        best_pass8 = max(results, key=lambda x: x['pass@8'])
        
        print("=== 最佳结果 ===")
        print(f"最佳 pass@1: {best_pass1['pass@1']*100:.2f}% (步数 {best_pass1['step']})")
        print(f"最佳 pass@8: {best_pass8['pass@8']*100:.2f}% (步数 {best_pass8['step']})")
        
        # 打印完整表格
        print("\n=== 完整指标表 ===")
        print(f"{'步数':<10} {'问题数':<10} {'pass@1':<15} {'pass@8':<15}")
        print("-" * 70)
        for r in results[::3]:  # 每隔3个显示，避免太长
            print(f"{r['step']:<10} {r['n_problems']:<10} {r['pass@1']*100:>6.2f}%{'':<8} {r['pass@8']*100:>6.2f}%")
        
        print("\n提示: 查看完整数据请打开CSV文件")
        print(f"  cat {output_csv}")
    else:
        print("\n✗ 没有有效数据")


if __name__ == '__main__':
    main()
