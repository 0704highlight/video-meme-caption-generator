import os
import requests
import base64
import cv2
import numpy as np
import json
import re
from typing import List, Dict, Any
from config import VLM_API_KEY, VLM_API_ENDPOINT, LLM_API_KEY, LLM_API_ENDPOINT, VIDEO_DESCRIPTION_DIMENSIONS, ANIME_DESCRIPTION_DIMENSIONS

class APIClient:
    """API客户端类，处理与VLM和LLM的API通信"""
    
    def __init__(self):
        self.vlm_api_key = os.environ.get('VLM_API_KEY', VLM_API_KEY)
        self.vlm_endpoint = os.environ.get('VLM_API_ENDPOINT', VLM_API_ENDPOINT)
        self.llm_api_key = os.environ.get('LLM_API_KEY', LLM_API_KEY)
        self.llm_endpoint = os.environ.get('LLM_API_ENDPOINT', LLM_API_ENDPOINT)
        
        # 验证API密钥设置
        if not self.vlm_api_key:
            print("警告：未设置VLM API密钥，将使用模拟数据")
        if not self.llm_api_key:
            print("警告：未设置LLM API密钥，将使用模拟数据")
    
    def analyze_video_frames(self, frames: List[np.ndarray]) -> Dict[str, str]:
        """
        使用VLM分析视频帧内容
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict[str, str]: 视频分析结果，包含各个维度的描述
        """
        # 如果没有配置API，返回模拟数据
        if not self.vlm_api_key or not self.vlm_endpoint:
            print("未设置VLM API密钥或端点，使用模拟数据")
            return self._generate_mock_analysis()
        
        try:
            # 将帧转换为base64编码
            encoded_frames = [self._encode_image(frame) for frame in frames]
            
            # 构建API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.vlm_api_key}"
            }
            
            # 构建消息内容，包含文本和图像
            # 添加系统消息以帮助模型理解这是视频帧序列
            system_content = """你是一个专业的视频分析助手，擅长分析视频帧内容。
            接下来你将收到一组按时间顺序排列的视频帧图像。这些帧来自同一个视频，代表了视频的关键时刻。
            请分析这个视频内容，注意人物表情、动作、场景变化等关键细节。
            理解这是一个时间序列，不同帧之间可能展示了一个动作或情节的发展过程。"""
            
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请分析这些视频帧，描述以下内容：1. 人物表情，2. 人物动作，3. 场景，4. 人物着装，5. 主要颜色，6. 画面构图。如果是动漫图像，还请额外分析：1. 动漫类型，2. 角色特征，3. 常见梗。以JSON格式返回结果。请特别注意这些帧是视频的时间序列，描述中应包含对动作或场景变化的理解。"
                        }
                    ]
                }
            ]
            
            # 将图像添加到消息内容中
            for encoded_frame in encoded_frames:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_frame}"
                    }
                })
            
            # 构建完整请求体
            payload = {
                "model": "doubao-1-5-vision-pro-32k-250115",  # 修正模型名称
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # 发送API请求
            response = requests.post(
                self.vlm_endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # 检查响应状态
            if response.status_code != 200:
                print(f"VLM API请求失败，状态码：{response.status_code}")
                print(f"错误信息：{response.text}")
                return self._generate_mock_analysis()
            
            # 解析响应
            result = response.json()
            
            # 尝试从返回的文本中提取JSON内容
            try:
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                # 提取JSON部分
                import re
                import json
                
                # 尝试直接解析整个内容
                try:
                    analysis = json.loads(content)
                except:
                    # 如果失败，尝试从文本中提取JSON部分
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        analysis = json.loads(json_str)
                    else:
                        # 如果仍然失败，尝试用正则表达式匹配键值对
                        analysis = {}
                        for dim in VIDEO_DESCRIPTION_DIMENSIONS + ANIME_DESCRIPTION_DIMENSIONS:
                            match = re.search(rf'{dim}[：:]\s*(.*?)(?:\n|$)', content)
                            if match:
                                analysis[dim] = match.group(1).strip()
                
                return analysis
            except Exception as e:
                print(f"解析VLM响应内容时出错：{str(e)}")
                return self._generate_mock_analysis()
            
        except Exception as e:
            print(f"调用VLM API时出错：{str(e)}")
            return self._generate_mock_analysis()
    
    def generate_caption(self, description: Dict[str, str], style: str, rag_examples: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        根据视频描述生成配文
        
        Args:
            description: 视频描述信息
            style: 配文风格
            rag_examples: 可选的RAG检索结果
            
        Returns:
            Dict[str, Any]: 生成结果，包含:
                - caption: 主配文
                - alt_captions: 可选配文列表
        """
        # 如果没有配置API，返回模拟数据
        if not self.llm_api_key or not self.llm_endpoint:
            print("未设置LLM API密钥或端点，使用模拟数据")
            return self._generate_mock_caption(description)
        
        try:
            # 构建提示词
            prompt = self._build_caption_prompt(description, style, rag_examples)
            
            # 构建API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            # 构建消息数组
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的视频配文助手，擅长根据视频内容生成有趣、简短、吸引人的配文。配文应该能够引起共鸣和传播。请直接输出配文文本，不要添加任何格式标记如【】，不要添加标签如'备选配文1：'，不要添加emoji。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # 构建完整请求体
            payload = {
                "model": "doubao-1-5-pro-32k-250115",  # 使用提供的模型
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # 发送API请求
            response = requests.post(
                self.llm_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # 检查响应状态
            if response.status_code != 200:
                print(f"LLM API请求失败，状态码：{response.status_code}")
                print(f"错误信息：{response.text}")
                return self._generate_mock_caption(description)
            
            # 解析响应
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # 提取主配文和替代配文
            caption = ""
            alt_captions = []
            
            # 处理返回内容
            lines = content.split('\n')
            filtered_lines = [self._clean_caption(line.strip()) for line in lines if line.strip() and not line.startswith('#') and not line.startswith('*')]
            
            if filtered_lines:
                caption = filtered_lines[0]
                # 剩余行作为替代配文
                alt_captions = filtered_lines[1:4] if len(filtered_lines) > 1 else []
            
            # 如果没有提取到有效配文，使用整个内容作为配文
            if not caption and content:
                caption = self._clean_caption(content)
            
            # 如果仍然没有配文，使用模拟数据
            if not caption:
                return self._generate_mock_caption(description)
            
            # 如果没有替代配文，生成一些
            if not alt_captions:
                alt_captions = self._generate_alt_captions(caption, style)
            
            return {
                "caption": caption,
                "alt_captions": alt_captions
            }
            
        except Exception as e:
            print(f"调用LLM API时出错：{str(e)}")
            return self._generate_mock_caption(description)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """将OpenCV图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _build_caption_prompt(self, description: Dict[str, str], style: str, rag_examples: List[Dict[str, Any]] = None) -> str:
        """构建生成配文的提示词"""
        prompt = "请根据以下视频内容描述，生成一个适合表情包的配文。\n\n"
        prompt += "视频内容描述：\n"
        
        # 添加视频描述
        for key, value in description.items():
            # 跳过布局类型和帧数量等技术细节
            if key not in ['布局类型', '帧数量', '视频时长']:
                prompt += f"- {key}：{value}\n"
        
        # 添加视频的技术信息
        if '视频时长' in description:
            prompt += f"\n视频时长：{description.get('视频时长')}\n"
            
        # 添加配文风格
        prompt += f"\n配文风格：{style}\n"
        
        # 添加RAG示例（如果有）
        if rag_examples and len(rag_examples) > 0:
            prompt += "\n参考相似案例：\n"
            for i, example in enumerate(rag_examples[:3]):  # 最多使用前3个示例
                prompt += f"示例{i+1}：{example.get('caption', '')}（{example.get('description', '')}）\n"
        
        # 添加特殊要求
        if style.lower() == 'anime' or '动漫' in description.get('动漫类型', ''):
            prompt += "\n特殊要求：\n"
            prompt += f"- 角色特征：{description.get('角色特征', '普通')}\n"
            
            if description.get('使用颜文字', False):
                prompt += "- 请在配文中适当添加表情，但不要使用emoji或颜文字\n"
        
        # 添加输出格式要求
        prompt += "\n请直接输出配文，不要有任何前缀或解释、格式标记如【】，不要添加标签。配文应该简洁有力，能够引起共鸣和传播。\n"
        prompt += "请同时提供3个替代配文选项，但不要添加标签如'备选配文1：'，直接每行输出一个配文即可。"
        
        return prompt
    
    def _generate_alt_captions(self, main_caption: str, style: str) -> List[str]:
        """生成替代配文选项"""
        # 在实际场景中，这里应该调用LLM API生成替代选项
        # 为了简化，这里返回模拟数据
        if style.lower() == 'funny':
            return [
                f"{main_caption}！笑死我了！",
                f"哈哈哈！{main_caption[:-1] if main_caption.endswith(('！', '!', '？', '?')) else main_caption}",
                f"这也太真实了：{main_caption}"
            ]
        elif style.lower() == 'sarcastic':
            return [
                f"{main_caption}...真是太棒了呢",
                f"不愧是你，{main_caption}",
                f"我只想说：{main_caption}"
            ]
        elif style.lower() == 'anime':
            return [
                f"{main_caption} (≧▽≦)",
                f"{main_caption.replace('！', '')}！！ (*/ω＼*)",
                f"呜呜呜...{main_caption} (╥﹏╥)"
            ]
        else:
            return [
                f"{main_caption}~",
                f"啊这...{main_caption}",
                f"就是这个感觉：{main_caption}"
            ]
    
    def _generate_mock_analysis(self) -> Dict[str, str]:
        """生成模拟的视频分析结果"""
        return {
            '人物表情': '惊讶',
            '人物动作': '睁大眼睛，张开嘴',
            '场景': '室内，办公环境',
            '人物着装': '正式，白色衬衫',
            '主要颜色': '白色，蓝色',
            '画面构图': '特写镜头，人物居中'
        }
    
    def _generate_mock_caption(self, description: Dict[str, str]) -> Dict[str, Any]:
        """生成模拟的配文结果"""
        captions = {
            'funny': "我看到预算时的表情！",
            'sarcastic': "周一见到老板的那一刻...",
            'cute': "啊！原来是这样的啊！好惊讶！",
            'anime': "这这这这是在开玩笑吧！？"
        }
        
        alt_captions = {
            'funny': [
                "领导说今天要加班的瞬间！",
                "当我发现昨天的工作全白做了...",
                "看到同事工资比我高时的表情"
            ],
            'sarcastic': [
                "哦，又要改需求了呢，真令人愉悦...",
                "这就是所谓的'小需求'吗？真有趣...",
                "让我猜猜，这是'紧急'任务？"
            ],
            'cute': [
                "哇塞！这也太厉害啦！",
                "咦？真的吗？好神奇呀！",
                "呀！原来是这样的呀！学习了！"
            ],
            'anime': [
                "等等...这不可能...这绝对不可能！！",
                "什...什么？！这种事情...怎么会！？",
                "不不不不可能！我不相信！"
            ]
        }
        
        return {
            "caption": captions.get(description.get('动漫类型', 'funny'), captions['funny']),
            "alt_captions": alt_captions.get(description.get('动漫类型', 'funny'), alt_captions['funny'])
        } 
    
    def _clean_caption(self, caption: str) -> str:
        """清理配文中的格式符号"""
        # 移除【】及内部文本
        caption = re.sub(r'【.*?】', '', caption)
        # 移除"备选配文X："格式
        caption = re.sub(r'备选配文\d+[:：]', '', caption)
        # 移除"替代配文X："格式
        caption = re.sub(r'替代配文\d+[:：]', '', caption)
        # 移除前缀的数字序号
        caption = re.sub(r'^\d+[:：]', '', caption)
        # 清理空白
        caption = caption.strip()
        return caption