import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import faiss
import sys

# 打印Python路径信息用于调试
print("Python路径:", sys.executable)
print("sys.path:", sys.path)

# 添加venv的site-packages目录到路径
site_packages_path = os.path.join('D:', 'Ex', 'venv', 'Lib', 'site-packages')
if os.path.exists(site_packages_path) and site_packages_path not in sys.path:
    sys.path.append(site_packages_path)
    print(f"已添加路径: {site_packages_path}")

try:
    # langchain相关导入
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings  # 使用HuggingFaceEmbeddings替代FakeEmbeddings
    from langchain_community.embeddings import FakeEmbeddings  # 作为备用
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
    print("langchain模块导入成功")
except ImportError as e:
    print(f"导入错误: {e}")
    # 尝试使用替代导入
    try:
        import langchain
        print("langchain版本:", langchain.__version__)
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings  # 使用HuggingFaceEmbeddings替代FakeEmbeddings
        from langchain.embeddings import FakeEmbeddings  # 作为备用
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        from langchain.schema import Document
        print("使用原始langchain导入成功")
    except ImportError as e2:
        print(f"替代导入也失败: {e2}")
        raise

# 导入项目配置
from config import DB_PATH, LLM_API_KEY, LLM_API_ENDPOINT

class LangchainRAG:
    """基于Langchain的检索增强生成系统"""
    
    def __init__(self, db_path: str = DB_PATH):
        """
        初始化Langchain RAG系统
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        
        # 创建数据库目录（如果不存在）
        os.makedirs(db_path, exist_ok=True)
        
        # 尝试初始化HuggingFaceEmbeddings
        try:
            # 使用本地下载的模型
            model_path = os.path.join(os.getcwd(), "m3e-base")
            if os.path.exists(model_path):
                print(f"使用本地HuggingFaceEmbeddings模型: {model_path}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={'device': 'cpu'}
                )
            else:
                print(f"本地模型路径不存在: {model_path}，使用FakeEmbeddings作为后备")
                self.embeddings = FakeEmbeddings(size=1536)  # 修改为1536维，更接近真实模型
                print("使用FakeEmbeddings嵌入模型")
        except Exception as e:
            print(f"初始化HuggingFaceEmbeddings失败: {e}，使用FakeEmbeddings作为后备")
            self.embeddings = FakeEmbeddings(size=1536)  # 修改为1536维，更接近真实模型
            print("使用FakeEmbeddings嵌入模型")
        
        # 初始化向量存储
        self.vector_store = self._init_vector_store()
        
        # 初始化LLM模型（使用项目已有API配置）
        self.llm = self._init_llm()
        
        # 加载示例数据（如果存在）
        self.examples = self._load_examples()
        
    def _init_vector_store(self) -> VectorStore:
        """初始化向量存储"""
        faiss_index_path = os.path.join(self.db_path, "faiss_index")
        examples_path = os.path.join(self.db_path, "examples.json")
        
        # 检查是否已有向量存储
        if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
            try:
                # 从已有索引加载
                return FAISS.load_local(faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"加载FAISS索引出错: {e}")
        
        # 如果没有现有索引，但有示例文件，则创建新索引
        if os.path.exists(examples_path):
            try:
                with open(examples_path, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                
                # 将示例转换为文档格式
                docs = []
                for example in examples:
                    text = f"描述: {example.get('description', '')}\n风格: {example.get('style', '')}"
                    metadata = {
                        "caption": example.get("caption", ""),
                        "style": example.get("style", ""),
                        "emotion_tags": example.get("emotion_tags", [])
                    }
                    docs.append(Document(page_content=text, metadata=metadata))
                
                # 创建新的向量存储
                if docs:
                    vector_store = FAISS.from_documents(docs, self.embeddings)
                    # 保存索引
                    vector_store.save_local(faiss_index_path)
                    return vector_store
            except Exception as e:
                print(f"从示例创建FAISS索引出错: {e}")
        
        # 如果没有现有索引和示例，创建空的向量存储
        return FAISS.from_texts(["空示例"], self.embeddings)
    
    def _init_llm(self):
        """初始化LLM模型"""
        try:
            # 使用OpenAI客户端直接连接API
            from openai import OpenAI
            
            # 确保API端点格式正确，移除末尾的/chat/completions（如果存在）
            base_url = LLM_API_ENDPOINT
            if base_url.endswith('/chat/completions'):
                base_url = base_url[:-16]  # 移除末尾的'/chat/completions'
            
            print(f"使用OpenAI客户端连接LLM API")
            print(f"  - API密钥: {LLM_API_KEY[:5]}...{LLM_API_KEY[-5:] if LLM_API_KEY else None}")
            print(f"  - API端点: {base_url}")
            
            from langchain_openai import ChatOpenAI
            
            return ChatOpenAI(
                api_key=LLM_API_KEY,
                base_url=base_url,
                model_name="doubao-1-5-pro-32k-250115",
                temperature=0.7,
                streaming=False,
                verbose=True
            )
        except Exception as e:
            print(f"初始化LLM出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回一个简单的模拟LLM，仅用于测试
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import AIMessage, BaseMessage
            from typing import List, Optional, Any
            
            class MockChatModel(BaseChatModel):
                """模拟聊天模型，返回预设的有趣配文"""
                
                def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> AIMessage:
                    return AIMessage(content="这是一个模拟的回复，因为无法连接到实际的LLM API。")
                
                async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> AIMessage:
                    return AIMessage(content="这是一个模拟的回复，因为无法连接到实际的LLM API。")
                
                @property
                def _llm_type(self) -> str:
                    return "mock-chat-model"
            
            print("使用模拟LLM代替真实API")
            return MockChatModel()
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """加载示例数据"""
        examples_path = os.path.join(self.db_path, "examples.json")
        if os.path.exists(examples_path):
            try:
                with open(examples_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载示例文件出错: {e}")
                return []
        return []
    
    def add_example(self, description: Dict[str, str], style: str, caption: str, emotion_tags: List[str] = None) -> None:
        """
        添加示例到RAG系统
        
        Args:
            description: 视频描述
            style: 配文风格
            caption: 配文内容
            emotion_tags: 情感标签
        """
        # 准备示例数据
        if isinstance(description, dict):
            desc_text = " ".join([f"{key}：{value}" for key, value in description.items()])
        else:
            desc_text = str(description)
            
        example = {
            "description": desc_text,
            "style": style,
            "caption": caption,
            "emotion_tags": emotion_tags or []
        }
        
        # 更新示例列表
        self.examples.append(example)
        
        # 更新向量存储
        text = f"描述: {desc_text}\n风格: {style}"
        metadata = {
            "caption": caption,
            "style": style,
            "emotion_tags": emotion_tags or []
        }
        
        self.vector_store.add_documents([Document(page_content=text, metadata=metadata)])
        
        # 保存示例数据和向量存储
        self._save_examples()
        self._save_vector_store()
    
    def _save_examples(self) -> None:
        """保存示例数据"""
        examples_path = os.path.join(self.db_path, "examples.json")
        try:
            with open(examples_path, 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存示例数据出错: {e}")
    
    def _save_vector_store(self) -> None:
        """保存向量存储"""
        faiss_index_path = os.path.join(self.db_path, "faiss_index")
        try:
            self.vector_store.save_local(faiss_index_path)
        except Exception as e:
            print(f"保存向量存储出错: {e}")
    
    def search(self, description: Dict[str, str], style: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索相似的表情包配文
        
        Args:
            description: 视频描述
            style: 配文风格
            k: 返回结果数量
            
        Returns:
            相似示例列表
        """
        if isinstance(description, dict):
            query_text = " ".join([f"{key}：{value}" for key, value in description.items()])
        else:
            query_text = str(description)
            
        query = f"描述: {query_text}\n风格: {style}"
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                results.append({
                    "description": doc.page_content,
                    "caption": doc.metadata.get("caption", ""),
                    "style": doc.metadata.get("style", ""),
                    "emotion_tags": doc.metadata.get("emotion_tags", [])
                })
            
            return results
        except Exception as e:
            print(f"搜索相似表情包配文出错: {e}")
            return []
            
    def generate_caption_with_rag(self, description: Dict[str, str], style: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        使用RAG生成配文
        
        Args:
            description: 视频描述
            style: 配文风格
            params: 额外的配文参数，包含风格和其他设置
            
        Returns:
            Dict[str, Any]: 生成结果，包含:
                - caption: 主配文
                - alt_captions: 可选配文列表
        """
        try:
            # 准备查询
            if isinstance(description, dict):
                query_text = "\n".join([f"{key}：{value}" for key, value in description.items()])
            else:
                query_text = str(description)
                
            # 检索相似示例
            retrieved_docs = self.vector_store.similarity_search(
                f"描述: {query_text}\n风格: {style}", 
                k=3
            )
            
            # 准备相似示例文本
            examples_text = ""
            for i, doc in enumerate(retrieved_docs):
                caption = doc.metadata.get("caption", "")
                if caption:
                    examples_text += f"示例{i+1}：{caption}\n"
            
            # 构建提示模板 - 根据风格和参数调整
            style_guidance = ""
            if params:
                if style == "anime" or style == "ANIME":
                    # 添加动漫风格的特殊参数
                    anime_type = params.get('anime_type', '日漫')
                    character_trait = params.get('character_trait', '元气')
                    use_emoticons = params.get('use_emoticons', True)
                    
                    style_guidance = f"""
配文风格: 动漫风格
动漫类型: {anime_type}
角色特征: {character_trait}
使用颜文字: {'是' if use_emoticons else '否'}
"""
                else:
                    style_guidance = f"配文风格: {style}"
            else:
                style_guidance = f"配文风格: {style}"
            
            template = """你是一个专业的视频配文助手，擅长根据视频内容生成有趣、简短、吸引人的配文。

视频内容描述:
{description}

{style_guidance}

{examples}

请根据以上信息，生成一个适合表情包的配文。配文应该简洁有力，能够引起共鸣和传播。
同时提供3个替代配文选项。

输出格式:
1. 主配文
2. 替代配文1
3. 替代配文2
4. 替代配文3
"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["description", "style_guidance", "examples"]
            )
            
            # 直接使用LLM和提示模板生成配文，避免使用RetrievalQA链
            # 这样可以避免StuffDocumentsChain的验证错误
            input_values = {
                "description": query_text,
                "style_guidance": style_guidance,
                "examples": f"参考相似案例:\n{examples_text}" if examples_text else ""
            }
            
            # 生成配文
            result = self.llm.invoke(prompt.format(**input_values))
            
            # 解析结果
            content = result.content.strip()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # 提取主配文和备选配文
            captions = []
            for line in lines:
                # 移除行号和点
                cleaned_line = line
                if line and line[0].isdigit() and '.' in line[:3]:
                    cleaned_line = line.split('.', 1)[1].strip()
                captions.append(cleaned_line)
            
            # 确保至少有一个配文
            if not captions:
                captions = ["生成配文失败，请重试"]
            
            return {
                "caption": captions[0],
                "alt_captions": captions[1:] if len(captions) > 1 else []
            }
            
        except Exception as e:
            print(f"使用Langchain RAG生成配文出错: {e}")
            return {
                "caption": "生成配文出错，请重试",
                "alt_captions": []
            } 