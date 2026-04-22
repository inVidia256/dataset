#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1-Parser 批量PDF解析脚本
从 docs 文件夹读取所有PDF文件，为每个PDF生成单独的输出文件夹

用法：python batch_parse.py [options]

示例：
  python batch_parse.py                           # 使用默认路径
  python batch_parse.py -i ./my_pdfs -o ./results # 自定义输入输出路径
"""

import os
import sys
import argparse
import time
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到Python路径
script_path = Path(__file__).resolve()
project_root = script_path.parent
sys.path.append(str(project_root))

from magic_pdf.model.taichu_custom_model import TaichuOCR
from parse.code.self_task import parse_pdf


def initialize_model(config_path=None):
    """
    初始化TaichuOCR模型
    
    Args:
        config_path: 配置文件路径，如果为None则使用项目根目录的model_configs.yaml
        
    Returns:
        初始化的TaichuOCR模型实例
    """
    if config_path is None:
        default_config_path = str(project_root / "model_configs.yaml")
        config_path = default_config_path
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    print(f"📋 加载配置文件: {config_path}")
    print("🚀 初始化S1-Parser模型...")
    start_time = time.time()
    
    model_parser = TaichuOCR(config_path)
    
    init_time = time.time() - start_time
    print(f"✅ 模型初始化完成，耗时: {init_time:.2f}s")
    
    return model_parser


def parse_single_pdf(pdf_path, output_dir, model_parser):
    """
    解析单个PDF文件（用于多进程）
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        model_parser: 预初始化的模型实例
        
    Returns:
        tuple: (pdf文件名, 是否成功, 结果目录或错误信息)
    """
    pdf_name = os.path.basename(pdf_path)
    try:
        result_dir = parse_pdf(pdf_path, output_dir, model_parser)
        return (pdf_name, True, result_dir)
    except Exception as e:
        return (pdf_name, False, str(e))


def get_pdf_files(input_dir):
    """
    获取输入目录下所有PDF文件
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        list: PDF文件路径列表
    """
    pdf_pattern = os.path.join(input_dir, "**", "*.pdf")
    pdf_files = glob.glob(pdf_pattern, recursive=True)
    
    # 也检查大写的 .PDF
    pdf_pattern_upper = os.path.join(input_dir, "**", "*.PDF")
    pdf_files_upper = glob.glob(pdf_pattern_upper, recursive=True)
    
    # 合并并去重
    all_pdfs = list(set(pdf_files + pdf_files_upper))
    
    # 按文件名排序
    all_pdfs.sort()
    
    return all_pdfs


def batch_parse_pdfs(input_dir, output_dir, config_path=None, workers=1):
    """
    批量解析PDF文件
    
    Args:
        input_dir: 输入目录（包含PDF文件）
        output_dir: 输出目录
        config_path: 配置文件路径
        workers: 并行工作进程数（默认1，即串行处理）
    """
    # 检查输入目录
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有PDF文件
    pdf_files = get_pdf_files(input_dir)
    
    if not pdf_files:
        print(f"⚠️ 在 {input_dir} 目录下未找到PDF文件")
        return
    
    print(f"📚 共找到 {len(pdf_files)} 个PDF文件")
    print("=" * 60)
    
    # 初始化模型（只初始化一次）
    model_parser = initialize_model(config_path)
    
    # 处理统计
    total = len(pdf_files)
    success_count = 0
    failed_count = 0
    failed_files = []
    
    start_time = time.time()
    
    if workers > 1:
        # 多进程处理（注意：模型可能不支持真正的并行，这里主要是I/O并行）
        print(f"⚙️ 使用 {workers} 个进程并行处理...")
        # 实际上，由于模型通常是GPU绑定的，多进程可能不会有太大收益
        # 但保留这个选项以备不时之需
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(parse_single_pdf, pdf_path, output_dir, model_parser): pdf_path
                for pdf_path in pdf_files
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                pdf_path = futures[future]
                pdf_name, success, result = future.result()
                
                if success:
                    success_count += 1
                    print(f"[{i}/{total}] ✅ 成功: {pdf_name}")
                else:
                    failed_count += 1
                    failed_files.append((pdf_name, result))
                    print(f"[{i}/{total}] ❌ 失败: {pdf_name}")
                    print(f"    错误: {result}")
    else:
        # 串行处理（推荐，因为模型初始化开销大，且GPU通常只能处理一个任务）
        print("⚙️ 串行处理模式...")
        for i, pdf_path in enumerate(pdf_files, 1):
            pdf_name = os.path.basename(pdf_path)
            print(f"\n[{i}/{total}] 📄 正在处理: {pdf_name}")
            print("-" * 60)
            
            try:
                result_dir = parse_pdf(pdf_path, output_dir, model_parser)
                success_count += 1
                print(f"✅ 完成: {pdf_name}")
                print(f"📂 结果目录: {result_dir}")
            except Exception as e:
                failed_count += 1
                failed_files.append((pdf_name, str(e)))
                print(f"❌ 失败: {pdf_name}")
                print(f"   错误: {str(e)}")
    
    total_time = time.time() - start_time
    
    # 打印总结
    print("\n" + "=" * 60)
    print("📊 批量处理完成")
    print("=" * 60)
    print(f"总文件数: {total}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"总耗时: {total_time:.2f}s")
    if total > 0:
        print(f"平均每个文件: {total_time/total:.2f}s")
    print("=" * 60)
    
    if failed_files:
        print("\n❌ 失败的文件:")
        for name, error in failed_files:
            print(f"  - {name}: {error}")
    
    return success_count, failed_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='S1-Parser 批量PDF解析脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_parse.py
  python batch_parse.py -i ./docs -o ./results
  python batch_parse.py --input ./my_pdfs --output ./output --config ./custom_config.yaml
        """
    )
    
    # 输入输出路径
    parser.add_argument(
        '-i', '--input',
        default=str(project_root / "docs"),
        help='输入目录路径，包含要处理的PDF文件 (默认: ./docs)'
    )
    parser.add_argument(
        '-o', '--output',
        default=str(project_root / "results"),
        help='输出目录路径 (默认: ./results)'
    )
    parser.add_argument(
        '-c', '--config',
        default=None,
        help='配置文件路径 (默认: model_configs.yaml)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=1,
        help='并行工作进程数 (默认: 1，建议保持1因为模型通常是GPU绑定的)'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("S1-Parser 批量PDF解析")
        print("=" * 60)
        print(f"📁 输入目录: {args.input}")
        print(f"📁 输出目录: {args.output}")
        print("=" * 60)
        
        batch_parse_pdfs(args.input, args.output, args.config, args.workers)
        
        print("\n" + "=" * 60)
        print("🎉 全部完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
