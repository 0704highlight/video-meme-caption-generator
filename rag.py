import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import faiss
from config import DB_PATH

class RAGSystem:
    """基于检索增强生成的系统，用于检索相似的表情包配文案例"""
    
    def __init__(self, db_path: str = DB_PATH):
        """
        初始化RAG系统
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.examples = []
        self.index = None
        self.embeddings = None
        
        # 创建数据库目录（如果不存在）
        os.makedirs(db_path, exist_ok=True)
        
        # 加载数据库
        self._load_database()
    
    def _load_database(self) -> None:
        """加载数据库并创建索引"""
        db_file = os.path.join(self.db_path, "examples.json")
        embeddings_file = os.path.join(self.db_path, "embeddings.npy")
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_file) or not os.path.exists(embeddings_file):
            print(f"数据库文件不存在，将创建新的数据库: {db_file}")
            self._create_empty_database()
            return
        
        try:
            # 加载示例数据
            with open(db_file, 'r', encoding='utf-8') as f:
                self.examples = json.load(f)
            
            # 加载嵌入向量
            self.embeddings = np.load(embeddings_file)
            
            # 创建FAISS索引
            self._create_index()
            
            print(f"已加载数据库，共{len(self.examples)}个示例")
        except Exception as e:
            print(f"加载数据库时出错: {e}")
            self._create_empty_database()
    
    def _create_empty_database(self) -> None:
        """创建空数据库"""
        self.examples = []
        self.embeddings = np.zeros((0, 128))  # 假设嵌入维度为128
        self._create_index()
    
    def _create_index(self) -> None:
        """创建FAISS索引"""
        if len(self.examples) == 0:
            # 创建空索引
            dimension = 128  # 假设嵌入维度为128
            self.index = faiss.IndexFlatL2(dimension)
            return
        
        # 创建FAISS索引
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype(np.float32))
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """
        计算文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # 注意：这是一个简化实现，实际应该调用嵌入模型API
        # 这里只是随机生成一个向量来模拟嵌入
        np.random.seed(hash(text) % 2**32)
        return np.random.random(128).astype(np.float32)
    
    def search(self, description: Dict[str, str], style: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        检索相似的表情包配文案例
        
        Args:
            description: 视频描述
            style: 配文风格
            k: 返回的结果数量
            
        Returns:
            相似案例列表
        """
        if len(self.examples) == 0:
            print("数据库为空，无法执行检索")
            return []
        
        # 准备查询文本
        query_text = " ".join([f"{key}：{value}" for key, value in description.items()])
        query_text += f" 风格：{style}"
        
        # 计算查询嵌入
        query_embedding = self._compute_embedding(query_text)
        
        # 执行检索
        k = min(k, len(self.examples))
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # 返回检索结果
        results = []
        for i in range(k):
            idx = indices[0][i]
            results.append(self.examples[idx])
        
        return results
    
    def add_example(self, 
                   description: Dict[str, str], 
                   style: str, 
                   caption: str,
                   emotion_tags: List[str] = None) -> None:
        """
        添加新的配文案例到数据库
        
        Args:
            description: 视频描述
            style: 配文风格
            caption: 生成的配文
            emotion_tags: 情感标签列表
        """
        # 准备示例数据
        example = {
            "description": " ".join([f"{key}：{value}" for key, value in description.items()]),
            "style": style,
            "caption": caption,
            "emotion_tags": emotion_tags or []
        }
        
        # 计算嵌入
        text = example["description"] + f" 风格：{style}"
        embedding = self._compute_embedding(text)
        
        # 更新数据库
        self.examples.append(example)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        # 更新索引
        if self.index is None:
            self._create_index()
        else:
            self.index.add(embedding.reshape(1, -1))
        
        # 保存数据库
        self._save_database()
    
    def _save_database(self) -> None:
        """保存数据库到文件"""
        try:
            # 保存示例数据
            db_file = os.path.join(self.db_path, "examples.json")
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, ensure_ascii=False, indent=2)
            
            # 保存嵌入向量
            embeddings_file = os.path.join(self.db_path, "embeddings.npy")
            np.save(embeddings_file, self.embeddings)
            
            print(f"数据库已保存，共{len(self.examples)}个示例")
        except Exception as e:
            print(f"保存数据库时出错: {e}")
    
    def add_example_file(self, file_path: str) -> int:
        """
        从文件批量添加示例
        
        Args:
            file_path: 示例文件路径（JSON格式）
            
        Returns:
            添加的示例数量
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            
            count = 0
            for example in examples:
                if "description" in example and "style" in example and "caption" in example:
                    # 解析描述
                    description_parts = example["description"].split()
                    description = {}
                    for part in description_parts:
                        if "：" in part:
                            key, value = part.split("：", 1)
                            description[key] = value
                    
                    # 添加示例
                    self.add_example(
                        description=description,
                        style=example["style"],
                        caption=example["caption"],
                        emotion_tags=example.get("emotion_tags", [])
                    )
                    count += 1
            
            return count
        except Exception as e:
            print(f"从文件添加示例时出错: {e}")
            return 0 