# test.py
"""
S1-Parser 直接推理脚本
无需Web服务，直接读取PDF进行推理
用法：python test.py [pdf_path] [output_dir]

示例：python test.py test_docs/sample.pdf ./results
"""

import os
import sys
import argparse
import time
from pathlib import Path
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
        # 获取脚本路径
        script_path = Path(__file__).resolve()
        # 计算默认配置路径（项目根目录的 model_configs.yaml）
        project_root = script_path.parent
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


def parse_single_pdf(pdf_path, output_dir, config_path=None):
    """
    解析单个PDF文件
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        config_path: 配置文件路径
        
    Returns:
        解析结果目录
    """
    # 检查输入文件
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"只支持PDF文件，当前文件: {pdf_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    model_parser = initialize_model(config_path)
    
    print(f"📄 开始解析PDF文件: {pdf_path}")
    print(f"📁 输出目录: {output_dir}")
    
    # 执行解析
    start_time = time.time()
    
    try:
        result_dir = parse_pdf(pdf_path, output_dir, model_parser)
        
        parse_time = time.time() - start_time
        print(f"✅ PDF解析完成，耗时: {parse_time:.2f}s")
        print(f"📂 结果保存在: {result_dir}")
        
        # 显示生成的文件
        if os.path.exists(result_dir):
            print("\n📋 生成的文件列表:")
            for root, dirs, files in os.walk(result_dir):
                level = root.replace(result_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}└─ {os.path.basename(root) or result_dir}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}└─ {file}")
        
        return result_dir
        
    except Exception as e:
        print(f"❌ 解析失败: {str(e)}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='S1-Parser 直接推理脚本')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('--config', help='配置文件路径 (默认: model_configs.yaml)')
    
    args = parser.parse_args()
    
    try:
        # 添加项目根目录到Python路径
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        print("=" * 60)
        print("S1-Parser 直接推理脚本")
        print("=" * 60)
        
        parse_single_pdf(args.pdf_path, args.output_dir, args.config)
        
        print("\n" + "=" * 60)
        print("🎉 推理完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()