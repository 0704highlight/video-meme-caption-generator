import os
from dotenv import load_dotenv
from enum import Enum, auto

# 加载环境变量
load_dotenv()

# 应用配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

# API配置
VLM_API_KEY = os.environ.get('VLM_API_KEY', '')
VLM_API_ENDPOINT = os.environ.get('VLM_API_ENDPOINT', 'https://ark.cn-beijing.volces.com/api/v3')
LLM_API_KEY = os.environ.get('LLM_API_KEY', 'edc32f02e43f9a0034e6b9ec32df01e77')
LLM_API_ENDPOINT = os.environ.get('LLM_API_ENDPOINT', 'https://ark.cn-beijing.volces.com/api/v3')

# 数据库配置
DB_PATH = os.getenv("DB_PATH", "./database")

# 关键帧提取
NUM_FRAMES = 4  # 提取的关键帧数量

# 视频描述维度
VIDEO_DESCRIPTION_DIMENSIONS = [
    '人物表情',
    '人物动作',
    '场景',
    '人物着装',
    '主要颜色',
    '画面构图'
]

# 动漫描述维度
ANIME_DESCRIPTION_DIMENSIONS = [
    '动漫类型',
    '角色特征',
    '常见梗'
]

# 布局类型
class LayoutType(Enum):
    GRID = 'grid'
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'
    SINGLE = 'single'
    LIST = 'list'  # 新增：带有时间序列指示的列表布局
    GRID_IN_LIST = 'grid_in_list'  # 新增：网格视图嵌入列表布局

# 配文风格
class CaptionStyle(Enum):
    FUNNY = 'funny'
    SARCASTIC = 'sarcastic'
    CUTE = 'cute'
    ANIME = 'anime'

# 动漫角色特征
ANIME_CHARACTER_TRAITS = [
    '傲娇',
    '元气',
    '腹黑',
    '天然呆',
    '中二',
    '温柔',
    '强势',
    '胆小',
    '认真',
    '懒散'
]

# 动漫类型
ANIME_TYPES = [
    '日漫',
    '国漫',
    '欧美动画',
    '少女漫画',
    '少年漫画',
    '科幻动漫',
    '奇幻动漫',
    '校园动漫',
    '恋爱动漫',
    '治愈系'
] 