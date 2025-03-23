# 视频表情包配文生成器

这是一个基于Flask的Web应用，可以从视频中提取关键帧，并利用视觉语言大模型(VLM)和语言大模型(LLM)自动生成适合的表情包配文。应用还集成了RAG（检索增强生成）技术，通过历史数据增强配文生成效果。

## 功能特点

- **关键帧提取**：从用户上传的视频中提取多个关键帧
- **多种布局**：支持网格布局、水平布局、垂直布局和单帧布局
- **多种风格**：支持搞笑、讽刺、可爱和动漫等多种配文风格
- **动漫特化**：针对动漫内容提供特定的配文风格与表达方式
- **检索增强生成**：利用RAG技术参考历史数据库中的相似案例
- **替代配文选择**：为每个生成结果提供多个备选配文

## 系统架构

该应用采用模块化设计，主要包括以下组件：

- **视频处理模块**：负责视频关键帧提取与布局处理
- **图像生成模块**：负责将关键帧和配文合成为最终图像
- **API客户端**：负责与VLM和LLM API通信
- **RAG引擎**：负责检索相似案例
- **智能代理**：协调各模块工作并整合用户偏好

## 开始使用

### 前提条件

- Python 3.8+
- Flask 2.0+
- OpenCV
- NumPy
- FAISS (用于向量检索)
- 其他依赖项详见`requirements.txt`

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/0704highlight/video-meme-caption-generator.git
cd video-meme-caption-generator
```

2. 激活虚拟环境
```bash
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 设置API密钥（可选）
```bash
# 创建.env文件（或设置环境变量）
VLM_API_KEY=your_vlm_api_key
LLM_API_KEY=your_llm_api_key
```

5. 生成嵌入向量（初次使用）
```bash
python dummy_embeddings_generator.py
```

6. 运行应用
```bash
python app.py
```

浏览器访问 `http://127.0.0.1:5000/` 即可使用

## 使用方法

1. 从首页上传视频文件（支持MP4, AVI, MOV等格式）
2. 选择布局方式（网格、水平、垂直或单帧）
3. 选择配文风格（搞笑、讽刺、可爱或动漫）
4. 如选择动漫风格，可进一步选择动漫类型和角色特征
5. 开启/关闭RAG检索增强
6. 点击"生成表情包配文"按钮
7. 在结果页面查看生成的表情包，可选择备选配文
8. 下载生成的表情包图片

## 无API模式

如果没有配置API密钥，应用将使用模拟数据进行演示。

## 贡献指南

欢迎通过Issue和Pull Request来改进这个项目。具体贡献方法请参阅`CONTRIBUTING.md`。

## 许可证

本项目使用MIT许可证 - 详情请查看`LICENSE`文件。
