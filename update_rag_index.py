#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
import sys
from pathlib import Path

# 尝试导入自定义模块
try:
    from rag_engine import RAGEngine
    from langchain_rag import LangchainRAG
    from config import DB_PATH
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    print("请确保在正确的目录下运行此脚本")
    sys.exit(1)

def rebuild_rag_index(database_dir="database", verbose=False):
    """
    重建RAG系统的FAISS索引
    
    参数:
        database_dir: 数据库目录路径，默认为"database"
        verbose: 是否输出详细信息，默认为False
    
    返回:
        成功返回True，失败返回False
    """
    try:
        # 确保数据库目录存在
        os.makedirs(database_dir, exist_ok=True)
        
        # 定义相关文件路径
        faiss_index_path = os.path.join(database_dir, "faiss_index")
        examples_file = os.path.join(database_dir, "examples.json")
        
        # 确认examples.json存在
        if not os.path.exists(examples_file):
            print(f"错误: 示例文件不存在 ({examples_file})")
            return False
        
        if verbose:
            print(f"正在检查示例文件: {examples_file}")
            # 显示示例数量
            with open(examples_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
                print(f"找到 {len(examples)} 个示例")
        
        # 1. 重建RAGEngine的索引
        try:
            if verbose:
                print("正在重建RAGEngine索引...")
            rag_engine = RAGEngine()  # RAGEngine构造函数不接受database_dir参数
            if verbose:
                print("RAGEngine索引重建成功")
        except Exception as e:
            print(f"重建RAGEngine索引失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        # 2. 重建LangchainRAG的索引
        try:
            # 如果FAISS索引目录存在，先删除它
            if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
                if verbose:
                    print(f"正在删除现有FAISS索引目录: {faiss_index_path}")
                shutil.rmtree(faiss_index_path)
                if verbose:
                    print("索引目录已删除")
            
            # 重新初始化会创建新的索引
            if verbose:
                print("正在重建LangchainRAG索引...")
            langchain_rag = LangchainRAG(db_path=database_dir)  # 使用db_path参数
            if verbose:
                print("LangchainRAG索引重建成功")
                
            # 验证索引是否已创建
            if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
                if verbose:
                    print(f"已确认索引目录存在: {faiss_index_path}")
                    # 列出索引目录中的文件
                    files = os.listdir(faiss_index_path)
                    print(f"索引目录包含以下文件: {files}")
            else:
                print(f"警告: 无法确认索引目录是否已创建: {faiss_index_path}")
                
        except Exception as e:
            print(f"重建LangchainRAG索引失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        if verbose:
            print("FAISS索引已成功重建")
        
        return True
    
    except Exception as e:
        print(f"重建索引时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数，解析命令行参数并执行重建索引操作
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="重建RAG系统的FAISS索引")
    parser.add_argument("-d", "--database", default=DB_PATH, help=f"数据库目录路径 (默认: {DB_PATH})")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细信息")
    
    # 解析参数
    args = parser.parse_args()
    
    # 重建索引
    success = rebuild_rag_index(args.database, args.verbose)
    
    # 输出结果
    if success:
        print("索引重建成功")
        return 0
    else:
        print("索引重建失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 