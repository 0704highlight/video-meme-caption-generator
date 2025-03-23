import os
import json
import numpy as np
import faiss
from typing import Dict, List, Any

class RAGEngine:
    """检索增强生成引擎"""
    
    def __init__(self):
        """初始化RAG引擎"""
        self.database_dir = "database"
        self.examples_file = os.path.join(self.database_dir, "examples.json")
        self.embeddings_file = os.path.join(self.database_dir, "embeddings.npy")
        
        # 加载示例和嵌入
        self.examples = self._load_examples()
        self.embeddings = self._load_embeddings()
        
        # 初始化FAISS索引
        if len(self.embeddings) > 0:
            self.dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.embeddings)
        else:
            print("警告：没有找到嵌入向量，RAG检索将返回空结果")
            self.index = None
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """
        加载示例数据
        
        Returns:
            示例数据列表
        """
        if not os.path.exists(self.examples_file):
            print(f"警告：示例文件 {self.examples_file} 不存在")
            return []
        
        try:
            with open(self.examples_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载示例文件时出错: {e}")
            return []
    
    def _load_embeddings(self) -> np.ndarray:
        """
        加载嵌入向量
        
        Returns:
            嵌入向量数组
        """
        if not os.path.exists(self.embeddings_file):
            print(f"警告：嵌入向量文件 {self.embeddings_file} 不存在")
            return np.array([])
        
        try:
            return np.load(self.embeddings_file)
        except Exception as e:
            print(f"加载嵌入向量文件时出错: {e}")
            return np.array([])
    
    def retrieve(self, description: Dict[str, str], k: int = 3) -> List[Dict[str, Any]]:
        """
        根据描述检索相似的例子
        
        Args:
            description: 视频描述
            k: 返回的结果数量
            
        Returns:
            相似例子的列表
        """
        # 如果没有索引或没有示例，返回空列表
        if self.index is None or len(self.examples) == 0:
            # 模拟一些结果
            return self._get_mock_results(description)
        
        # 生成查询向量
        query_vector = self._generate_query_vector(description)
        
        # 使用FAISS检索相似向量
        k = min(k, len(self.examples))
        distances, indices = self.index.search(np.array([query_vector]), k)
        
        # 返回检索结果
        results = []
        for i in indices[0]:
            results.append(self.examples[i])
        
        return results
    
    def _generate_query_vector(self, description: Dict[str, str]) -> np.ndarray:
        """
        生成查询向量
        
        实际应用中，应该使用与生成embeddings.npy相同的模型生成查询向量
        这里简单模拟一个随机向量
        
        Args:
            description: 视频描述
            
        Returns:
            查询向量
        """
        # 模拟生成查询向量
        # 在实际应用中，应该使用与训练嵌入相同的模型
        if self.index is not None:
            # 随机选择一个已有的嵌入向量
            return self.embeddings[np.random.randint(0, len(self.embeddings))]
        else:
            # 如果没有索引，生成一个随机向量
            return np.random.random(128)
    
    def _get_mock_results(self, description: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        生成模拟的检索结果
        
        Args:
            description: 视频描述
            
        Returns:
            模拟的检索结果
        """
        # 如果有示例，返回前3个
        if self.examples:
            return self.examples[:min(3, len(self.examples))]
        
        # 否则返回空列表
        return []
    
    def add_example(self, example_data: Dict[str, Any]) -> None:
        """
        添加新的配文案例到数据库
        
        Args:
            example_data: 示例数据，包含描述、风格、配文和情感标签
        """
        # 确保目录存在
        os.makedirs(self.database_dir, exist_ok=True)
        
        # 添加到示例列表
        self.examples.append(example_data)
        
        # 保存到文件
        self._save_examples()
        
        print(f"示例已添加到RAGEngine，现有{len(self.examples)}个示例")
        
    def _save_examples(self) -> None:
        """保存示例数据到文件"""
        try:
            with open(self.examples_file, 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, ensure_ascii=False, indent=2)
            print(f"示例数据已保存到{self.examples_file}")
        except Exception as e:
            print(f"保存示例数据时出错: {e}") 