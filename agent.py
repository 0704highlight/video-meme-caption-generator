from typing import Dict, List, Any
from config import CaptionStyle
from langchain_rag import LangchainRAG
import json

class Agent:
    """表情包配文生成助手"""
    
    def __init__(self, api_client, rag_engine=None):
        """初始化表情包配文生成助手
        
        Args:
            api_client: API客户端实例
            rag_engine: RAG检索引擎实例（可选）
        """
        self.api_client = api_client
        self.rag_engine = rag_engine
        
        # 初始化LangchainRAG
        self.langchain_rag = LangchainRAG()
    
    def generate_caption(self, description: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据视频描述生成配文
        
        Args:
            description: 视频分析结果，包含各个维度的描述
            params: 配文参数
                - style: 配文风格
                - rag_enabled: 是否启用RAG
                - anime_* (如果有): 动漫相关参数
                
        Returns:
            Dict[str, Any]: 配文结果
        """
        # 获取配文风格
        style = params.get('style', CaptionStyle.FUNNY)
        if isinstance(style, CaptionStyle):
            style = style.value
        
        # 将布局信息添加到请求中，帮助模型生成更适合该布局的配文
        layout_info = {}
        if '布局类型' in description:
            layout_type = description['布局类型']
            if layout_type == 'grid':
                layout_info['layout_name'] = '网格布局'
                layout_info['frame_count'] = min(4, int(description.get('帧数量', 4)))
            elif layout_type == 'list':
                layout_info['layout_name'] = '时间序列布局'
                layout_info['frame_count'] = min(6, int(description.get('帧数量', 6)))
            elif layout_type == 'grid_in_list':
                layout_info['layout_name'] = '网格+序列布局'
                layout_info['frame_count'] = min(8, int(description.get('帧数量', 8)))
        
        # 用于存储生成结果
        caption_result = None
        
        # 使用LangchainRAG进行生成（如果启用了RAG）
        if params.get('rag_enabled', False):
            try:
                # 使用Langchain RAG生成配文
                caption_result = self.langchain_rag.generate_caption_with_rag(description, style)
                
                # 将生成的配文保存到RAG系统中（用于未来的检索）
                if caption_result and caption_result.get('caption'):
                    # 提取情感标签
                    emotion_tags = self._extract_emotion_tags(description)
                    
                    # 添加示例到RAG系统
                    self.langchain_rag.add_example(
                        description=description,
                        style=style,
                        caption=caption_result['caption'],
                        emotion_tags=emotion_tags
                    )
            except Exception as e:
                print(f"使用Langchain RAG生成配文出错: {e}")
                caption_result = None
        
        # 如果RAG未启用或生成失败，使用普通API生成
        if caption_result is None:
            # 检索相似案例（仍然使用原有RAG引擎，但仅用于参考）
            rag_examples = []
            if params.get('rag_enabled', False) and self.rag_engine:
                try:
                    # 构建查询向量
                    query = description
                    # 检索相似案例
                    rag_examples = self.rag_engine.retrieve(query)
                except Exception as e:
                    print(f"使用原有RAG引擎检索出错: {e}")
                
            # 生成配文
            caption_result = self.api_client.generate_caption(description, style, rag_examples)
        
        return caption_result
    
    def _prepare_description_for_rag(self, description: Dict[str, str]) -> str:
        """准备用于RAG检索的描述文本
        
        Args:
            description: 视频描述信息
            
        Returns:
            str: 用于RAG检索的描述文本
        """
        # 提取最重要的描述维度用于检索
        important_keys = ['人物表情', '人物动作', '情感表现', '角色特征']
        description_parts = []
        
        for key in important_keys:
            if key in description and description[key]:
                description_parts.append(f"{key}: {description[key]}")
        
        # 如果没有找到任何重要维度，使用所有可用的描述
        if not description_parts:
            description_parts = [f"{k}: {v}" for k, v in description.items() if v]
        
        return " ".join(description_parts)
    
    def _get_style_string(self, style) -> str:
        """转换风格枚举为字符串形式
        
        Args:
            style: 风格枚举
            
        Returns:
            str: 风格字符串
        """
        if style == CaptionStyle.FUNNY:
            return "funny"
        elif style == CaptionStyle.SARCASTIC:
            return "sarcastic"
        elif style == CaptionStyle.CUTE:
            return "cute"
        elif style == CaptionStyle.ANIME:
            return "anime"
        else:
            return "funny"  # 默认使用搞笑风格
    
    def _enrich_description_with_anime_params(self, description: Dict[str, str], params: Dict[str, Any], style) -> Dict[str, str]:
        """根据用户参数丰富描述信息，特别是动漫相关参数
        
        Args:
            description: 原始视频描述
            params: 用户参数
            style: 配文风格
            
        Returns:
            Dict[str, str]: 丰富后的描述
        """
        # 创建描述的副本以避免修改原始数据
        enriched_description = description.copy()
        
        # 只有当风格为动漫且原描述中没有相关信息时才添加
        if style == CaptionStyle.ANIME:
            # 添加动漫类型（如果未包含）
            if '动漫类型' not in enriched_description and 'anime_type' in params:
                enriched_description['动漫类型'] = params.get('anime_type', '日漫')
            
            # 添加角色特征（如果未包含）
            if '角色特征' not in enriched_description and 'character_trait' in params:
                enriched_description['角色特征'] = params.get('character_trait', '元气')
            
            # 添加是否使用颜文字
            enriched_description['使用颜文字'] = params.get('use_emoticons', True)
        
        return enriched_description
    
    def _get_query_from_description(self, description: Dict[str, str]) -> Dict[str, str]:
        """从描述中提取查询信息
        
        Args:
            description: 视频描述信息
            
        Returns:
            Dict[str, str]: 用于RAG检索的查询信息
        """
        # 提取最重要的维度
        query = {}
        important_keys = ['人物表情', '人物动作', '情感表现', '角色特征', '场景', '主要颜色']
        
        for key in important_keys:
            if key in description and description[key]:
                query[key] = description[key]
        
        return query
    
    def add_example(self, description, caption, output_path=None, params=None, style=None):
        """添加示例到数据库"""
        if not description:
            return False
        
        try:
            # 提取情感标签
            emotion_tags = self._extract_emotion_tags(description)
            
            # 构建示例数据
            example = {
                'description': description,
                'caption': caption,
                'emotion_tags': emotion_tags
            }
            
            if style:
                example['style'] = style
                
            if output_path:
                example['output_path'] = output_path
                
            # 保存到数据库
            examples = self.load_examples()
            examples.append(example)
            
            with open(self.examples_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
                
            # 更新索引
            if self.rag_engine == "naive":
                self.naive_rag.update_index(examples)
            elif self.rag_engine == "langchain" and self.langchain_rag:
                self.langchain_rag.update_index(examples)
                
            print(f"已添加新示例, 情感标签: {emotion_tags}")
            return True
            
        except Exception as e:
            print(f"添加示例时出错: {e}")
            return False
    
    def _extract_emotion_tags(self, description):
        """
        从描述中提取情感标签
        """
        # 首先检查是否有用户自定义的情感标签
        if '用户情感标签' in description and description['用户情感标签']:
            return description['用户情感标签']
            
        emotion_tags = []
        emotion_fields = [
            '情感表现', '人物表情', '情绪状态', '表情描述', '情感描述', 
            '表情', '情绪', '情感'
        ]
        
        # 常见情感词列表
        common_emotions = [
            '开心', '快乐', '愉悦', '欢喜', '高兴', 
            '悲伤', '难过', '痛苦', '忧伤', '伤心',
            '愤怒', '生气', '恼怒', '暴怒', '气愤',
            '惊讶', '震惊', '惊奇', '意外', '诧异',
            '害怕', '恐惧', '惊恐', '畏惧', '担忧',
            '厌恶', '反感', '恶心', '讨厌', '嫌弃',
            '无聊', '乏味', '无趣', '枯燥', '疲倦',
            '困惑', '疑惑', '不解', '迷惑', '迷茫',
            '尴尬', '羞愧', '害羞', '羞涩', '羞耻',
            '平静', '冷静', '安静', '淡定', '从容',
            '激动', '兴奋', '热情', '亢奋', '喜悦',
            '可爱', '萌', '娇憨', '俏皮', '调皮',
            '皮', '无语', '呆', '摆烂', '挖苦',
            '严肃', '正经', '认真', '庄重', '肃穆',
            '满足', '满意', '欣慰', '欣喜', '喜悦',
            '自信', '自豪', '得意', '骄傲', '自恋',
            
        ]
        
        # 1. 检查描述中指定情感字段
        for field in emotion_fields:
            if field in description and description[field]:
                value = description[field]
                # 检查是否为长句，尝试提取关键词
                if len(value) > 10 and ('，' in value or '、' in value or '。' in value):
                    print(f"发现长句标签: '{value}', 尝试提取关键词")
                    # 分割长句
                    parts = value.replace('，', '、').replace('。', '、').split('、')
                    for part in parts:
                        part = part.strip()
                        # 查找情感词
                        for emotion in common_emotions:
                            if emotion in part:
                                print(f"  - 提取到关键词: '{emotion}'")
                                if emotion not in emotion_tags:
                                    emotion_tags.append(emotion)
                else:
                    # 直接添加简短描述
                    if value not in emotion_tags and len(value) < 10:
                        emotion_tags.append(value)
        
        # 2. 在描述文本中查找常见情感词
        for field, value in description.items():
            if isinstance(value, str):
                for emotion in common_emotions:
                    if emotion in value and emotion not in emotion_tags:
                        emotion_tags.append(emotion)
        
        # 3. 根据配文风格设置默认情感标签（如果没有找到）
        if not emotion_tags and 'style' in self.current_params:
            style = self.current_params['style']
            if style == CaptionStyle.FUNNY:
                emotion_tags = ['搞笑', '幽默']
            elif style == CaptionStyle.SARCASTIC:
                emotion_tags = ['讽刺', '嘲讽']
            elif style == CaptionStyle.CUTE:
                emotion_tags = ['可爱', '萌']
            elif style == CaptionStyle.ANIME:
                # 获取角色特征作为情感标签
                character_trait = self.current_params.get('character_trait', '元气')
                emotion_tags = [character_trait]
        
        return emotion_tags 