import os
import time
import uuid
import json
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont

# 导入自定义模块
from config import (
    UPLOAD_FOLDER, OUTPUT_FOLDER, ALLOWED_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS,
    LayoutType, CaptionStyle, 
    ANIME_CHARACTER_TRAITS, ANIME_TYPES,
    VIDEO_DESCRIPTION_DIMENSIONS, ANIME_DESCRIPTION_DIMENSIONS
)
from video_processor import VideoProcessor
from image_generator import ImageGenerator
from api_client import APIClient
from agent import Agent
from rag_engine import RAGEngine
from langchain_rag import LangchainRAG

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'development_key')

# 确保上传和输出目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 初始化组件
video_processor = VideoProcessor()
image_generator = ImageGenerator()
api_client = APIClient()
rag_engine = RAGEngine()
# 初始化Langchain RAG系统
langchain_rag = LangchainRAG()
agent = Agent(api_client, rag_engine)

def allowed_file(filename):
    """检查文件是否有允许的扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_image_file(filename):
    """检查图像文件是否有允许的扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

@app.route('/')
def index():
    """渲染首页"""
    # 清除之前的会话数据
    session.clear()
    
    # 构建布局和风格选项
    layouts = [
        {"id": "grid", "name": "网格"},
        {"id": "horizontal", "name": "水平"},
        {"id": "vertical", "name": "垂直"},
        {"id": "single", "name": "单帧"},
        {"id": "list", "name": "列表式"},
        {"id": "grid_in_list", "name": "网格嵌入列表"}
    ]
    
    styles = [
        {"id": "funny", "name": "搞笑风格"},
        {"id": "sarcastic", "name": "讽刺风格"},
        {"id": "cute", "name": "可爱风格"},
        {"id": "anime", "name": "动漫风格"}
    ]
    
    return render_template(
        'index.html', 
        layouts=layouts, 
        styles=styles,
        anime_traits=ANIME_CHARACTER_TRAITS,
        anime_types=ANIME_TYPES
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传请求"""
    # 重置会话，确保不会混用上一次的结果
    for key in list(session.keys()):
        if key != '_flashes':  # 保留flash消息
            session.pop(key)
    
    # 检查是否上传了视频或图像
    if 'video' not in request.files and 'image' not in request.files:
        flash('没有选择文件', 'error')
        return redirect(url_for('index'))
    
    # 获取选项
    layout_type = request.form.get('layout', 'grid')
    caption_style = request.form.get('style', 'funny')
    rag_enabled = 'rag_enabled' in request.form
    num_frames = 8  # 默认提取8帧，以支持复杂布局
    
    # 检查是否使用自定义配文
    use_custom_caption = 'use_custom_caption' in request.form
    custom_caption = request.form.get('custom_caption', '')
    
    # 检查是否使用自定义风格
    use_custom_style = 'use_custom_style' in request.form
    custom_style = request.form.get('custom_style', '')
    if use_custom_style and custom_style:
        caption_style = 'custom_' + custom_style.lower().replace(' ', '_')
    
    # 获取情感标签
    emotion_tags = request.form.getlist('emotion_tags')
    
    # 根据布局类型调整要提取的帧数
    if layout_type == 'single':
        num_frames = 1
    elif layout_type in ['grid', 'horizontal', 'vertical']:
        num_frames = 4
    elif layout_type == 'list':
        num_frames = 6
    elif layout_type == 'grid_in_list':
        num_frames = 8
    
    # 获取动漫相关参数（如果有）
    anime_params = {}
    if caption_style == 'anime':
        anime_params = {
            'anime_type': request.form.get('anime_type', '日漫'),
            'character_trait': request.form.get('character_trait', '元气'),
            'use_emoticons': 'use_emoticons' in request.form
        }
    
    # 处理视频上传
    if 'video' in request.files and request.files['video'].filename != '':
        file = request.files['video']
        if allowed_file(file.filename):
            return process_video(file, layout_type, caption_style, rag_enabled, num_frames, anime_params, 
                                use_custom_caption, custom_caption, emotion_tags)
        else:
            flash('不支持的视频文件类型', 'error')
            return redirect(url_for('index'))
    
    # 处理图像上传
    elif 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        if allowed_image_file(file.filename):
            return process_image(file, caption_style, rag_enabled, anime_params, 
                                use_custom_caption, custom_caption, emotion_tags)
        else:
            flash('不支持的图像文件类型', 'error')
            return redirect(url_for('index'))
    
    flash('没有选择文件', 'error')
    return redirect(url_for('index'))

def process_video(file, layout_type, caption_style, rag_enabled, num_frames, anime_params, 
                 use_custom_caption=False, custom_caption='', emotion_tags=None):
    """处理上传的视频文件"""
    # 安全地处理文件名并保存
    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # 处理视频并生成表情包
    try:
        # 1. 提取视频关键帧
        frames = video_processor.extract_keyframes(filepath, num_frames)
        
        # 2. 根据布局组织帧
        layout_frames = video_processor.organize_frames(
            frames, 
            getattr(LayoutType, layout_type.upper())
        )
        
        # 3. 获取视频信息
        video_info = video_processor.get_video_info(filepath)
        
        # 4. 生成唯一输出文件名
        output_filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # 5. 分析视频内容获取描述
        description = {}
        try:
            # 尝试通过API获取视频描述
            description = api_client.analyze_video_frames(layout_frames)
            # 将布局类型添加到描述中，以便VLM了解这是帧序列而非单一图像
            description['布局类型'] = layout_type
            description['帧数量'] = len(layout_frames)
            description['视频时长'] = f"{video_info['duration']:.2f}秒"
        except Exception as e:
            # 如果API失败，使用模拟数据
            print(f"API分析失败: {str(e)}")
            description = generate_mock_description(caption_style, anime_params)
            description['布局类型'] = layout_type
            description['帧数量'] = len(layout_frames)
            description['视频时长'] = f"{video_info['duration']:.2f}秒"
        
        # 6. 如果用户提供了情感标签，添加到描述中
        if emotion_tags:
            description['用户情感标签'] = emotion_tags
        
        # 7. 根据描述生成配文或使用用户自定义配文
        caption_result = {}
        if use_custom_caption and custom_caption:
            caption_result = {'caption': custom_caption, 'alt_captions': []}
        else:
            caption_params = {
                'style': getattr(CaptionStyle, caption_style.upper(), CaptionStyle.FUNNY),
                'rag_enabled': rag_enabled
            }
            
            # 合并动漫参数（如果是动漫风格）
            if caption_style == 'anime':
                caption_params.update(anime_params)
            
            # 生成配文
            caption_result = agent.generate_caption(description, caption_params)
        
        # 8. 生成最终图片
        image_generator.generate_image(
            layout_frames,
            caption_result['caption'],
            getattr(LayoutType, layout_type.upper()),
            output_path
        )
        
        # 9. 准备结果
        result = {
            'caption': caption_result['caption'],
            'alt_captions': caption_result.get('alt_captions', []),
            'image_path': output_filename,
            'analysis': description,
            'video_path': unique_filename,
            'is_video': True,
            'layout_type': layout_type,
            'is_user_caption': use_custom_caption and bool(custom_caption),
            'emotion_tags': emotion_tags if emotion_tags else []
        }
        
        # 10. 如果是用户自定义配文，则将其保存到数据库
        if use_custom_caption and custom_caption:
            save_to_database(description, custom_caption, caption_style, emotion_tags)
        
        # 存储结果到会话中，以便后续使用
        session['last_result'] = result
        session['is_video_result'] = True
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        flash(f'处理视频时出错：{str(e)}', 'error')
        return redirect(url_for('index'))

def process_image(file, caption_style, rag_enabled, anime_params, 
                 use_custom_caption=False, custom_caption='', emotion_tags=None):
    """处理上传的图像文件"""
    # 安全地处理文件名并保存
    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    try:
        print(f"开始处理图像: {filepath}")
        
        # 1. 读取图像
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("无法读取图像文件")
            
            # 检查图像格式
            if image.ndim != 3:
                print(f"警告: 图像维度不是3 (实际是 {image.ndim})，尝试修复")
                if image.ndim == 2:  # 灰度图像
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            print(f"成功读取图像，尺寸: {image.shape}")
        except Exception as e:
            print(f"读取图像时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"无法处理图像文件: {str(e)}")
        
        # 2. 生成唯一输出文件名及原始图像文件名
        output_filename = f"{uuid.uuid4().hex}.jpg"
        original_image_filename = f"original_{uuid.uuid4().hex}.jpg"
        
        # 统一路径格式，使用正斜杠
        output_path = os.path.join('static', 'outputs', output_filename).replace('\\', '/')
        original_image_path = os.path.join('static', 'outputs', original_image_filename).replace('\\', '/')
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"生成的输出文件名: {output_filename}")
        print(f"输出路径: {output_path}")
        
        # 保存原始图像（用于后续自定义配文）
        try:
            # 转换OpenCV图像为PIL图像并保存
            original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(original_image_rgb).save(original_image_path)
            print(f"已保存原始图像到: {original_image_path}")
        except Exception as e:
            print(f"保存原始图像时出错: {str(e)}，将使用处理后的图像")
            original_image_filename = None
        
        # 3. 分析图像内容获取描述
        description = {}
        try:
            # 将图像放入列表，以便使用相同的API函数
            description = api_client.analyze_video_frames([image])
            description['是图像'] = True
        except Exception as e:
            print(f"API分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            description = generate_mock_description(caption_style, anime_params)
            description['是图像'] = True
        
        print(f"图像分析结果: {description}")
        
        # 4. 如果用户提供了情感标签，添加到描述中
        if emotion_tags:
            description['用户情感标签'] = emotion_tags
        
        # 5. 根据描述生成配文或使用用户自定义配文
        caption_result = {}
        if use_custom_caption and custom_caption:
            caption_result = {'caption': custom_caption, 'alt_captions': []}
            print(f"使用用户自定义配文: {custom_caption}")
        else:
            caption_params = {
                'style': getattr(CaptionStyle, caption_style.upper(), CaptionStyle.FUNNY),
                'rag_enabled': rag_enabled
            }
            
            # 合并动漫参数（如果是动漫风格）
            if caption_style == 'anime':
                caption_params.update(anime_params)
            
            # 生成配文
            try:
                print(f"开始生成配文，参数: {caption_params}")
                caption_result = agent.generate_caption(description, caption_params)
                print(f"生成的配文结果: {caption_result}")
            except Exception as e:
                print(f"生成配文时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                # 生成一个基本的配文作为回退选项
                caption_result = {
                    'caption': '生成配文出错，请重试',
                    'alt_captions': []
                }
        
        # 6. 添加配文到图像并保存
        try:
            # 直接添加配文到图像
            result_image = image_generator.add_caption(image, caption_result['caption'])
            
            # 保存图像
            if isinstance(result_image, Image.Image):
                # 是PIL图像对象，直接保存
                print(f"保存PIL图像到: {output_path}")
                result_image.save(output_path)
            else:
                # 是OpenCV图像对象，转换后保存
                print(f"保存OpenCV图像到: {output_path}")
                if isinstance(result_image, np.ndarray):
                    # 转换为PIL图像保存
                    if result_image.ndim == 3 and result_image.shape[2] == 3:
                        # 转换BGR为RGB
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        Image.fromarray(result_image_rgb).save(output_path)
                    else:
                        # 直接保存
                        Image.fromarray(result_image).save(output_path)
                else:
                    # 不是有效图像对象，创建错误图像
                    print(f"警告: 结果不是有效的图像对象，类型为: {type(result_image)}")
                    error_img = Image.new('RGB', (640, 480), color=(0, 0, 0))
                    draw = ImageDraw.Draw(error_img)
                    draw.text((20, 240), "图像生成错误", fill=(255, 255, 255))
                    draw.text((20, 280), caption_result['caption'][:50], fill=(255, 255, 255))
                    error_img.save(output_path)
            
            print(f"已保存结果图像到: {output_path}")
        except Exception as e:
            print(f"添加配文或保存图像时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 尝试创建一个基本的错误图像
            try:
                # 创建一个简单的错误图像
                error_img = Image.new('RGB', (640, 480), color=(0, 0, 0))
                draw = ImageDraw.Draw(error_img)
                draw.text((20, 240), f"图像处理失败: {str(e)[:50]}", fill=(255, 255, 255))
                draw.text((20, 280), caption_result['caption'][:50], fill=(255, 255, 255))
                
                # 保存错误图像
                error_img.save(output_path)
                print(f"已创建并保存错误图像到: {output_path}")
            except Exception as inner_e:
                print(f"创建错误图像也失败了: {str(inner_e)}")
                raise ValueError(f"无法生成和保存图像: {str(e)}")
        
        # 8. 准备结果
        result = {
            'caption': caption_result['caption'],
            'alt_captions': caption_result.get('alt_captions', []),
            'image_path': output_filename,
            'analysis': description,
            'is_video': False,
            'is_user_caption': use_custom_caption and bool(custom_caption),
            'emotion_tags': emotion_tags if emotion_tags else [],
            'style': caption_style  # 添加样式信息
        }
        
        # 添加原始图像路径（如果存在）
        if original_image_filename:
            result['original_image_path'] = original_image_filename
            print(f"已添加原始图像路径到结果中: {original_image_filename}")
        
        print(f"准备的结果数据: {result}")
        
        # 9. 如果是用户自定义配文，则将其保存到数据库
        if use_custom_caption and custom_caption:
            try:
                print("保存用户自定义配文到数据库")
                save_to_database(description, custom_caption, caption_style, emotion_tags)
            except Exception as e:
                print(f"保存到数据库时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 存储结果到会话中，以便后续使用
        session['last_result'] = result
        session['is_video_result'] = False
        session.modified = True  # 确保会话被保存
        print(f"已将结果保存到会话: {session['last_result']}")
        
        # 确认输出文件存在
        if os.path.exists(output_path):
            print(f"输出文件已确认存在: {output_path}")
        else:
            print(f"警告: 输出文件不存在: {output_path}")
        
        # 将控制权直接传递给result视图，而不是使用重定向
        print("直接渲染result模板而不是重定向")
        try:
            return render_template('result.html', result=result)
        except Exception as e:
            print(f"渲染result.html时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f'渲染结果页面时出错：{str(e)}', 'error')
            return redirect(url_for('index'))
        
    except Exception as e:
        print(f'处理图像时出错：{str(e)}')
        import traceback
        traceback.print_exc()
        
        # 尝试创建一个基本的错误结果
        error_result = {
            'caption': '处理图像出错，请重试',
            'alt_captions': [],
            'image_path': None,
            'error_message': str(e),
            'is_error': True
        }
        
        flash(f'处理图像时出错：{str(e)}', 'error')
        
        # 尝试直接显示错误页面，而不是重定向
        try:
            return render_template('error.html', error=str(e), result=error_result)
        except:
            # 如果错误模板不存在，则重定向到首页
            return redirect(url_for('index'))

def generate_mock_description(style, anime_params=None):
    """生成模拟描述数据，用于API不可用时测试"""
    description = {
        '人物表情': '惊讶',
        '人物动作': '睁大眼睛，张开嘴',
        '场景': '室内，办公环境',
        '人物着装': '正式，白色衬衫',
        '主要颜色': '白色，蓝色'
    }
    
    # 如果是动漫风格，添加动漫相关信息
    if style == 'anime' and anime_params:
        description.update({
            '动漫类型': anime_params.get('anime_type', '日漫'),
            '角色特征': anime_params.get('character_trait', '元气'),
            '常见梗': '惊讶表情',
            '使用颜文字': anime_params.get('use_emoticons', True)
        })
    
    return description

@app.errorhandler(413)
def too_large(e):
    flash('文件太大（最大16MB）', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

@app.route('/update_video_caption', methods=['POST'])
def update_video_caption():
    """更新视频表情包的配文"""
    try:
        # 获取请求数据
        data = request.json
        if not data or 'video_path' not in data or 'caption' not in data:
            print("更新视频配文请求缺少必要参数")
            return jsonify({"error": "缺少必要参数"}), 400
            
        video_path = data['video_path']
        caption = data['caption']
        print(f"接收到更新视频配文请求: video_path={video_path}, caption={caption}")
        
        # 默认生成视频
        generate_video = True
        
        # 找到对应的视频文件
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)
        if not os.path.exists(input_path):
            print(f"视频文件不存在: {input_path}")
            return jsonify({"error": "视频文件不存在"}), 404
            
        # 提取原始帧
        video_processor = VideoProcessor()
        layout_type = session.get('last_result', {}).get('layout_type', 'grid')
        print(f"更新视频配文使用的布局类型: {layout_type}")
        
        try:
            # 1. 提取视频关键帧
            print("开始提取视频关键帧")
            frames = video_processor.extract_keyframes(input_path)
            print(f"成功提取 {len(frames)} 个关键帧")
            
            # 2. 根据布局组织帧
            layout_frames = video_processor.organize_frames(
                frames, 
                getattr(LayoutType, layout_type.upper(), LayoutType.GRID)
            )
            print(f"成功根据布局 {layout_type} 组织帧")
            
            # 3. 生成唯一输出文件名
            output_filename = f"{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            print(f"生成输出文件路径: {output_path}")
            
            # 4. 生成最终图片
            image_generator.generate_image(
                layout_frames,
                caption,
                getattr(LayoutType, layout_type.upper(), LayoutType.GRID),
                output_path
            )
            print(f"成功生成图片: {output_path}")
            
            # 保存新的配文到会话
            if 'last_result' in session:
                session['last_result']['caption'] = caption
                session['last_result']['image_path'] = output_filename
                session['last_result']['is_user_caption'] = True
                print("成功更新会话中的配文和图片路径")
            else:
                print("警告: session中没有last_result")
            
            result = {
                "success": True,
                "image_path": output_filename
            }
            
            # 生成带配文的视频
            if generate_video:
                # 生成唯一的输出文件名
                timestamp = int(time.time())
                filename_parts = os.path.splitext(video_path)
                video_output_filename = f"{filename_parts[0]}_captioned_{timestamp}.mp4"
                video_output_path = os.path.join(app.config['OUTPUT_FOLDER'], video_output_filename)
                
                print(f"开始生成带配文的视频: {video_output_path}")
                # 为视频添加配文
                add_caption_to_video(input_path, video_output_path, caption)
                print(f"成功生成带配文的视频")
                
                # 添加视频路径到结果
                result["video_path"] = video_output_filename
                result["video_url"] = url_for('static', filename=f'outputs/{video_output_filename}')
                session['last_processed_video'] = video_output_filename
                print(f"成功添加视频信息到结果和会话")
            
            print(f"返回更新视频配文成功结果: {result}")
            return jsonify(result)
            
        except Exception as e:
            print(f"更新视频配文时出错：{str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        print(f"处理请求时出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/generate_video', methods=['POST'])
def generate_video():
    """将用户选择的配文添加到视频中"""
    if 'video_path' not in request.form or not request.form['video_path']:
        return jsonify({"error": "无效的视频路径"}), 400

    # 获取请求数据
    video_path = request.form['video_path']
    caption = request.form['caption']
    
    # 确保视频文件存在
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)
    if not os.path.exists(input_path):
        return jsonify({"error": "视频文件不存在"}), 404
    
    try:
        # 生成唯一的输出文件名
        timestamp = int(time.time())
        filename_parts = os.path.splitext(video_path)
        output_filename = f"{filename_parts[0]}_captioned_{timestamp}{filename_parts[1]}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # 添加配文到视频
        add_caption_to_video(input_path, output_path, caption)
        
        # 返回成功结果和输出路径
        session['last_processed_video'] = output_filename
        session['is_video_result'] = True
        return jsonify({"success": True, "video_path": output_filename})
    
    except Exception as e:
        # 记录错误并返回
        print(f"处理视频时出错：{str(e)}")
        return jsonify({"error": str(e)}), 500

def add_caption_to_video(input_path, output_path, caption):
    """向视频添加字幕"""
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频: {input_path}")
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 加载字体
        font_size = int(width * 0.04)  # 减小字体尺寸
        font_size = max(font_size, 14)  # 最小字体大小
        
        # 字体路径列表，优先使用支持中文和颜文字的字体
        font_paths = [
            "./fonts/msyh.ttc",     # 微软雅黑
            "./fonts/simhei.ttf",   # 黑体
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc"
        ]
        
        # 尝试加载字体
        pil_font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pil_font = ImageFont.truetype(font_path, font_size)
                    print(f"已加载字体: {font_path}")
                    break
                except Exception as e:
                    print(f"加载字体失败 {font_path}: {str(e)}")
        
        if pil_font is None:
            pil_font = ImageFont.load_default()
            print("使用默认字体")
        
        # 准备一个临时图片用于计算
        temp_img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(temp_img)
        
        # 增加文本背景高度，给更多文本显示空间
        text_bg_height = int(height * 0.2)  # 保持背景高度为图像高度的20%
        text_height = font_size
        
        # 计算背景位置 - 将位置上移
        bg_bottom = height - int(height * 0.05)  # 距离底部5%的位置
        bg_top = bg_bottom - text_bg_height     # 背景顶部位置
        
        # 文本换行
        max_width = width - 40  # 左右各留20像素的边距
        lines = []
        
        # 优化的文本分割算法
        parts = caption.split(' ')
        current_line = ""
        
        for i, part in enumerate(parts):
            # 如果当前部分是空格，则直接添加到当前行
            if part == "":
                current_line += " "
                continue
                
            # 测试添加这个部分后是否会超出宽度
            test_line = current_line + part
            if current_line:  # 如果当前行有内容，添加一个空格
                test_line += " "
                
            try:
                line_bbox = draw.textbbox((0, 0), test_line, font=pil_font)
                text_width = line_bbox[2] - line_bbox[0]
            except:
                text_width = font_size * len(test_line) * 0.6  # 估计
                
            # 如果一个单词都超过了最大宽度，需要逐字符分割
            if text_width > max_width:
                # 如果当前行有内容，先保存
                if current_line:
                    lines.append(current_line.rstrip())
                    current_line = ""
                
                # 逐字符分割这个部分
                char_line = ""
                for char in part:
                    test_char_line = char_line + char
                    try:
                        char_bbox = draw.textbbox((0, 0), test_char_line, font=pil_font)
                        char_width = char_bbox[2] - char_bbox[0]
                    except:
                        char_width = font_size * len(test_char_line) * 0.6
                        
                    if char_width <= max_width:
                        char_line = test_char_line
                    else:
                        lines.append(char_line)
                        char_line = char
                
                if char_line:
                    current_line = char_line + " "
            else:
                current_line = test_line
                
            # 检查是否需要换行（到达最后一个部分或已接近最大宽度）
            if i == len(parts) - 1 or text_width > max_width * 0.8:
                lines.append(current_line.rstrip())
                current_line = ""
        
        # 处理末尾可能遗留的内容
        if current_line:
            lines.append(current_line.rstrip())
            
        # 确保分割后的行数不为0
        if not lines:
            lines = [caption]
            
        # 如果行数太多，可能需要减小字体
        if len(lines) > 3:
            new_font_size = max(int(font_size * 0.8), 14)
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pil_font = ImageFont.truetype(font_path, new_font_size)
                        text_height = new_font_size
                        break
                    except:
                        pass
        
        # 处理每一帧
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为PIL图像
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_frame)
            
            # 绘制文本背景（更靠上的位置）
            overlay = Image.new('RGBA', pil_frame.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [(0, bg_top), (width, bg_bottom)],
                fill=(0, 0, 0, 200)  # 保持不透明度
            )
            
            # 将半透明覆盖层与原始帧合并
            pil_frame = Image.alpha_composite(pil_frame.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(pil_frame)
            
            # 添加文本（每行）- 在新的背景位置中垂直居中
            y_offset = bg_top + (text_bg_height - (len(lines) * (text_height + 8))) // 2  # 垂直居中
            for line in lines:
                # 计算居中位置
                try:
                    line_bbox = draw.textbbox((0, 0), line, font=pil_font)
                    line_width = line_bbox[2] - line_bbox[0]
                except:
                    line_width = font_size * len(line) * 0.6
                
                x_position = (width - line_width) // 2
                draw.text((x_position, y_offset), line, font=pil_font, fill=(255, 255, 255))
                y_offset += text_height + 8  # 增加行间距
            
            # 转换回OpenCV格式并写入输出视频
            cv_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            out.write(cv_frame)
            
            # 更新进度
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"处理进度: {frame_index}/{frame_count} ({frame_index/frame_count*100:.2f}%)")
        
        # 释放资源
        cap.release()
        out.release()
        
        print(f"视频生成完成: {output_path}")
        
    except Exception as e:
        print(f"添加配文到视频时出错: {str(e)}")
        raise

@app.route('/download/<path:filename>')
def download_file(filename):
    """下载生成的文件"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/update_image_caption', methods=['POST'])
def update_image_caption():
    try:
        data = request.json
        if not data or 'image_path' not in data or 'caption' not in data:
            return jsonify({'success': False, 'error': '无效的请求参数'})
        
        image_path = os.path.join(app.config['OUTPUT_FOLDER'], data['image_path'])
        caption = data['caption']
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': '图像文件不存在'})
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return jsonify({'success': False, 'error': '无法读取图像'})
            
            # 添加配文
            image_generator = ImageGenerator()
            output_image = image_generator.add_caption(image, caption)
            
            # 生成唯一的输出文件名
            output_filename = f"updated_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # 保存图像
            cv2.imwrite(output_path, output_image)
            
            return jsonify({
                'success': True, 
                'image_path': output_filename
            })
            
        except Exception as e:
            app.logger.error(f"更新图像配文时出错: {str(e)}")
            return jsonify({'success': False, 'error': f'更新图像配文时出错: {str(e)}'})
    
    except Exception as e:
        app.logger.error(f"更新图像配文时出错: {str(e)}")
        return jsonify({'success': False, 'error': f'更新图像配文时出错: {str(e)}'})

@app.route('/apply_custom_caption', methods=['POST'])
def apply_custom_caption():
    try:
        app.logger.info(f"接收到自定义配文请求: {request.json}")
        data = request.json
        if not data or 'caption' not in data:
            app.logger.error("无效的请求参数: 缺少配文")
            return jsonify({'success': False, 'error': '无效的请求参数'})
        
        caption = data.get('caption')
        emotion_tags = data.get('emotion_tags', [])
        is_video = data.get('is_video', False)
        
        # 确保emotion_tags是列表
        if isinstance(emotion_tags, str):
            emotion_tags = [emotion_tags]
        
        app.logger.info(f"处理{'视频' if is_video else '图像'}自定义配文: {caption}, 情感标签: {emotion_tags}")
        
        if is_video:
            # 处理视频自定义配文
            video_path = data.get('video_path')
            if not video_path:
                video_path = session.get('video_path')
                
            # 检查视频路径
            if not video_path:
                app.logger.error("视频路径未提供")
                return jsonify({'success': False, 'error': '视频路径未提供'})
                
            # 确保视频文件存在
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)
            if not os.path.exists(input_path):
                app.logger.error(f"视频文件不存在: {input_path}")
                return jsonify({'success': False, 'error': '视频文件不存在'})
            
            try:
                # 获取关键帧
                video_processor = VideoProcessor()
                frames = video_processor.extract_keyframes(input_path, 4)  # 提取4个关键帧
                
                # 获取当前布局设置
                layout_type = data.get('layout_type', 'grid')
                if not layout_type:
                    layout_type = session.get('layout_type', 'grid')
                
                app.logger.info(f"提取了 {len(frames)} 帧，使用布局: {layout_type}")
                
                # 创建表情包
                image_generator = ImageGenerator()
                layout_frames = video_processor.organize_frames(
                    frames, 
                    getattr(LayoutType, layout_type.upper(), LayoutType.GRID)
                )
                
                # 生成唯一的输出文件名
                output_filename = f"custom_{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                
                # 生成最终图片
                image_generator.generate_image(
                    layout_frames,
                    caption,
                    getattr(LayoutType, layout_type.upper(), LayoutType.GRID),
                    output_path
                )
                
                app.logger.info(f"已生成自定义配文图像: {output_path}")
                
                # 保存描述信息
                description = session.get('description', {})
                
                # 更新会话中的数据
                session['caption'] = caption
                session['emotion_tags'] = emotion_tags
                session['output_image_path'] = output_path
                
                # 保存到数据库
                try:
                    save_to_database(description, caption, session.get('style', 'custom'), emotion_tags)
                    app.logger.info("已保存自定义配文到数据库")
                except Exception as db_err:
                    app.logger.error(f"保存到数据库时出错: {str(db_err)}")
                
                return jsonify({
                    'success': True, 
                    'image_path': url_for('static', filename=os.path.basename(output_path))
                })
                
            except Exception as e:
                app.logger.error(f"处理视频时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': f'处理视频时出错: {str(e)}'})
        else:
            # 处理图像自定义配文
            image_path = data.get('image_path')
            if not image_path:
                image_path = session.get('image_path')
                
            # 检查图像路径
            if not image_path:
                app.logger.error("图像路径未提供")
                return jsonify({'success': False, 'error': '图像路径未提供'})
            
            # 确保图像文件存在
            if os.path.isabs(image_path):
                full_image_path = image_path
            else:
                # 尝试不同的可能路径
                possible_paths = [
                    image_path,
                    os.path.join(app.config['OUTPUT_FOLDER'], image_path),
                    os.path.join('static', 'outputs', image_path)
                ]
                
                full_image_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        full_image_path = path
                        break
                        
                if not full_image_path:
                    app.logger.error(f"找不到图像文件: {image_path}")
                    return jsonify({'success': False, 'error': '图像文件不存在'})
            
            try:
                # 读取原图
                image = None
                try:
                    # 尝试使用OpenCV读取
                    image = cv2.imread(full_image_path)
                    if image is None:
                        # 如果OpenCV读取失败，尝试使用PIL
                        pil_image = Image.open(full_image_path)
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as img_err:
                    app.logger.error(f"读取图像失败: {str(img_err)}")
                    return jsonify({'success': False, 'error': f'无法读取图像: {str(img_err)}'})
                
                # 添加配文
                image_generator = ImageGenerator()
                output_image = image_generator.add_caption(image, caption)
                
                # 生成唯一的输出文件名
                output_filename = f"custom_{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                
                # 保存图像 (使用PIL更可靠)
                if isinstance(output_image, Image.Image):
                    # 如果是PIL图像对象
                    output_image.save(output_path)
                else:
                    # 如果是OpenCV (numpy) 图像对象
                    if output_image.ndim == 3 and output_image.shape[2] == 3:
                        # 转换BGR为RGB
                        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                        Image.fromarray(output_image_rgb).save(output_path)
                    else:
                        # 直接保存
                        Image.fromarray(output_image).save(output_path)
                
                app.logger.info(f"已保存自定义配文图像到: {output_path}")
                
                # 保存描述信息
                description = session.get('description', {})
                
                # 更新会话中的数据
                session['caption'] = caption
                session['emotion_tags'] = emotion_tags
                session['output_image_path'] = output_path
                session.modified = True
                
                # 保存到数据库
                try:
                    save_to_database(description, caption, session.get('style', 'custom'), emotion_tags)
                    app.logger.info("已保存自定义配文到数据库")
                except Exception as db_err:
                    app.logger.error(f"保存到数据库时出错: {str(db_err)}")
                
                return jsonify({
                    'success': True, 
                    'image_path': url_for('static', filename=os.path.basename(output_path))
                })
                
            except Exception as e:
                app.logger.error(f"处理图像时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': f'处理图像时出错: {str(e)}'})
    
    except Exception as e:
        app.logger.error(f"应用自定义配文时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'应用自定义配文时出错: {str(e)}'})

@app.route('/save_custom_caption', methods=['POST'])
def save_custom_caption():
    """保存用户自定义的配文到examples.json"""
    try:
        # 获取表单数据
        custom_caption = request.form.get('custom_caption')
        custom_style = request.form.get('custom_style')
        emotion_tags_str = request.form.get('emotion_tags', '')
        analysis_json = request.form.get('analysis', '{}')
        
        # 解析情感标签
        emotion_tags = [tag.strip() for tag in emotion_tags_str.split(',') if tag.strip()]
        
        # 解析分析数据
        try:
            analysis = json.loads(analysis_json)
        except Exception as json_err:
            print(f"解析JSON失败: {json_err}, 原始数据: {analysis_json}")
            analysis = {}
        
        # 对于空的情感标签，使用默认值
        if not emotion_tags:
            if custom_style == 'funny':
                emotion_tags = ['搞笑', '幽默']
            elif custom_style == 'sarcastic':
                emotion_tags = ['讽刺', '戏谑']
            elif custom_style == 'cute':
                emotion_tags = ['可爱', '温馨']
            elif custom_style == 'anime':
                emotion_tags = ['动漫', '热血']
            else:
                emotion_tags = ['其他']
        
        # 创建示例数据
        example_data = {
            "description": analysis,
            "style": custom_style,
            "caption": custom_caption,
            "emotion_tags": emotion_tags
        }
        
        # 对于字典形式的description，转换为字符串
        if isinstance(analysis, dict):
            desc_text = " ".join([f"{key}：{value}" for key, value in analysis.items() if isinstance(value, str)])
            example_data["description"] = desc_text
        
        # 添加到RAG系统
        rag_engine.add_example(example_data)
        langchain_rag.add_example(analysis, custom_style, custom_caption, emotion_tags)
        
        print(f"成功保存用户自定义配文: {custom_caption}")
        return jsonify({"success": True})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"保存自定义配文失败: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/save_favorite_caption', methods=['POST'])
def save_favorite_caption():
    """将用户收藏的系统生成配文保存到examples.json"""
    try:
        # 获取请求数据
        data = request.json
        if not data or 'caption' not in data or 'style' not in data:
            return jsonify({"success": False, "error": "缺少必要参数"})
        
        caption = data.get('caption')
        style = data.get('style')
        analysis = data.get('analysis', {})
        
        # 为收藏的配文添加情感标签
        emotion_tags = []
        
        # 情感关键词列表 - 扩展更多情绪词汇
        emotion_keywords = [
            # 积极情绪
            '高兴', '开心', '愉悦', '兴奋', '微笑', '快乐', '喜悦', '欣喜', '欢乐', '雀跃', '欢快', '惊喜',
            '自信', '满足', '幸福', '温暖', '轻松', '舒适', '温馨', '感动', '激动', '振奋', '得意', '活力',
            
            # 消极情绪
            '悲伤', '伤心', '难过', '忧伤', '哀愁', '沮丧', '失落', '落寞', '痛苦', '悲痛', '悲痛欲绝',
            '愤怒', '生气', '恼怒', '暴怒', '愠怒', '恨', '怨恨', '烦躁', '不满', '不悦', '发火',
            '害怕', '恐惧', '惊恐', '惊吓', '惧怕', '紧张', '不安', '焦虑', '担忧', '忧虑', '恐慌',
            
            # 中性情绪
            '平静', '冷静', '淡然', '从容', '镇定', '严肃', '认真', '专注', '坚定', '沉着', '沉静',
            '惊讶', '吃惊', '震惊', '错愕', '疑惑', '困惑', '好奇', '迷惑', '不解', '茫然',
            
            # 其他情绪
            '羞涩', '羞耻', '尴尬', '羞怯', '害羞', '内向', '腼腆', '傲娇', '冷漠', '冷酷', '无情', 
            '孤独', '寂寞', '思念', '想念', '期待', '向往', '渴望', '希望', '憧憬', '梦幻', '浪漫'
        ]
        
        # 处理人物表情
        if '人物表情' in analysis:
            expression = analysis['人物表情']
            print(f"提取人物表情的情感标签: '{expression}'")
            
            # 1. 首先尝试直接匹配情感关键词
            found_emotions = []
            for keyword in emotion_keywords:
                if keyword in expression:
                    found_emotions.append(keyword)
            
            # 2. 如果找到了情感关键词，直接使用
            if found_emotions:
                print(f"  - 从人物表情中匹配到情感关键词: {found_emotions}")
                emotion_tags.extend(found_emotions)
            # 3. 否则尝试分割句子
            else:
                # 处理常见的分隔符
                segments = []
                if '、' in expression:
                    segments = [seg.strip() for seg in expression.split('、')]
                elif '，' in expression:
                    segments = [seg.strip() for seg in expression.split('，')]
                elif '。' in expression:
                    segments = [seg.strip() for seg in expression.split('。')]
                
                # 从分段中找出可能的情感词
                for segment in segments:
                    # 如果分段很短(小于5个字)，可能直接是情感词
                    if len(segment) < 5:
                        emotion_tags.append(segment)
                    else:
                        # 尝试从分段中匹配情感词
                        for keyword in emotion_keywords:
                            if keyword in segment:
                                emotion_tags.append(keyword)
                                break
                        # 如果没找到匹配的情感词，且分段不太长，添加整个分段
                        if len(segment) < 8 and not any(keyword in segment for keyword in emotion_keywords):
                            emotion_tags.append(segment)
        
        # 处理情感表现
        if '情感表现' in analysis:
            emotion_expression = analysis['情感表现']
            print(f"提取情感表现的情感标签: '{emotion_expression}'")
            
            # 类似于处理人物表情的逻辑
            found_emotions = []
            for keyword in emotion_keywords:
                if keyword in emotion_expression:
                    found_emotions.append(keyword)
            
            if found_emotions:
                print(f"  - 从情感表现中匹配到情感关键词: {found_emotions}")
                emotion_tags.extend(found_emotions)
            else:
                segments = []
                if '、' in emotion_expression:
                    segments = [seg.strip() for seg in emotion_expression.split('、')]
                elif '，' in emotion_expression:
                    segments = [seg.strip() for seg in emotion_expression.split('，')]
                elif '。' in emotion_expression:
                    segments = [seg.strip() for seg in emotion_expression.split('。')]
                
                for segment in segments:
                    if len(segment) < 5:
                        emotion_tags.append(segment)
                    else:
                        for keyword in emotion_keywords:
                            if keyword in segment:
                                emotion_tags.append(keyword)
                                break
                        if len(segment) < 8 and not any(keyword in segment for keyword in emotion_keywords):
                            emotion_tags.append(segment)
        
        # 如果有用户自定义的情感标签，优先使用这些
        if '用户情感标签' in analysis and isinstance(analysis['用户情感标签'], list):
            print(f"使用用户自定义情感标签: {analysis['用户情感标签']}")
            emotion_tags.extend(analysis['用户情感标签'])
        
        # 去重并限制标签数量
        emotion_tags = list(set(emotion_tags))[:5]  # 最多保留5个标签
        
        # 如果仍然没有提取到情感标签，添加默认标签
        if not emotion_tags:
            if style == 'funny':
                emotion_tags = ['搞笑', '幽默']
            elif style == 'sarcastic':
                emotion_tags = ['讽刺', '戏谑']
            elif style == 'cute':
                emotion_tags = ['可爱', '温馨']
            elif style == 'anime':
                emotion_tags = ['动漫', '热血']
            else:
                emotion_tags = ['其他']
            print(f"使用默认情感标签: {emotion_tags}")
        
        # 将analysis字典转换为description字符串
        desc_text = ""
        if isinstance(analysis, dict):
            desc_text = " ".join([f"{key}：{value}" for key, value in analysis.items() if isinstance(value, str)])
        
        # 创建示例数据
        example_data = {
            "description": desc_text,
            "style": style,
            "caption": caption,
            "emotion_tags": emotion_tags
        }
        
        print(f"最终情感标签: {emotion_tags}")
        
        # 添加到RAG系统
        rag_engine.add_example(example_data)
        langchain_rag.add_example(analysis, style, caption, emotion_tags)
        
        print(f"成功保存用户收藏的配文: {caption}")
        return jsonify({"success": True})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"保存收藏配文失败: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

# 添加自定义过滤器
@app.template_filter('basename')
def basename_filter(path):
    """获取文件路径的基本名称"""
    import os
    return os.path.basename(path)

def save_to_database(description, caption, style, emotion_tags=None):
    """
    只保存用户自定义的配文及其相关信息到数据库
    """
    try:
        # 创建要保存的数据对象
        example_data = {
            "caption": caption,
            "style": style
        }
        
        # 添加视觉描述，仅保留重要字段
        if '人物表情' in description:
            example_data['emotion_expression'] = description['人物表情']
        
        if '情感表现' in description:
            example_data['emotional_performance'] = description['情感表现']
            
        if '人物动作' in description:
            example_data['action'] = description['人物动作']
            
        if '人物外貌' in description:
            example_data['appearance'] = description['人物外貌']
            
        if '场景描述' in description:
            example_data['scene'] = description['场景描述']
        
        # 添加用户自定义的情感标签
        if emotion_tags:
            example_data['emotion_tags'] = emotion_tags
        
        # 添加时间戳
        example_data['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存到数据库
        rag_engine.add_example(example_data)
        langchain_rag.add_example(description, style, caption, emotion_tags)
        
        print(f"已保存用户自定义配文到数据库: {caption}")
        return True
        
    except Exception as e:
        print(f"保存到数据库时出错: {str(e)}")
        return False

@app.route('/rebuild_index', methods=['POST'])
def rebuild_index():
    """重建RAG系统的FAISS索引，在对examples.json进行修改后调用"""
    try:
        # 获取数据库路径
        database_dir = "database"
        faiss_index_path = os.path.join(database_dir, "faiss_index")
        examples_file = os.path.join(database_dir, "examples.json")
        
        # 确认examples.json存在
        if not os.path.exists(examples_file):
            return jsonify({"success": False, "error": "示例文件不存在"}), 404
        
        # 1. 重建RAGEngine的索引
        try:
            # 重新初始化RAG引擎以重新加载examples.json
            global rag_engine
            rag_engine = RAGEngine()
            print("已重新初始化RAGEngine")
        except Exception as e:
            print(f"重新初始化RAGEngine出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 2. 重建LangchainRAG的索引
        try:
            # 重新初始化Langchain RAG系统，它会自动重建索引
            global langchain_rag
            
            # 如果FAISS索引目录存在，先删除它
            import shutil
            if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
                print(f"正在删除现有FAISS索引目录: {faiss_index_path}")
                shutil.rmtree(faiss_index_path)
                print("索引目录已删除")
            
            # 重新初始化会创建新的索引
            langchain_rag = LangchainRAG()
            print("已重新初始化LangchainRAG并重建索引")
        except Exception as e:
            print(f"重新初始化LangchainRAG出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": f"重建Langchain索引失败: {str(e)}"}), 500
        
        return jsonify({"success": True, "message": "FAISS索引已成功重建"})
    except Exception as e:
        print(f"重建索引时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/admin')
def admin_page():
    """管理界面，用于查看和管理examples.json文件"""
    try:
        # 获取数据库路径和文件
        database_dir = "database"
        examples_file = os.path.join(database_dir, "examples.json")
        
        # 检查文件是否存在
        examples = []
        if os.path.exists(examples_file):
            try:
                with open(examples_file, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
            except Exception as e:
                flash(f'加载examples.json时出错：{str(e)}', 'error')
        
        # 返回管理页面
        return render_template('admin.html', examples=examples)
    except Exception as e:
        flash(f'加载管理页面时出错：{str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/delete_example', methods=['POST'])
def delete_example():
    """删除examples.json中的示例"""
    try:
        # 获取索引
        index = request.form.get('index')
        if index is None:
            return jsonify({"success": False, "error": "缺少索引参数"}), 400
        
        index = int(index)
        
        # 获取数据库路径和文件
        database_dir = "database"
        examples_file = os.path.join(database_dir, "examples.json")
        
        # 检查文件是否存在
        if not os.path.exists(examples_file):
            return jsonify({"success": False, "error": "示例文件不存在"}), 404
        
        # 读取文件
        examples = []
        with open(examples_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        # 检查索引是否有效
        if index < 0 or index >= len(examples):
            return jsonify({"success": False, "error": "无效的索引"}), 400
        
        # 删除示例
        deleted_example = examples.pop(index)
        
        # 保存文件
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "success": True, 
            "message": f"已删除示例: {deleted_example.get('caption', '')[:30]}...",
            "need_rebuild": True
        })
        
    except Exception as e:
        print(f"删除示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/update_example', methods=['POST'])
def update_example():
    """更新examples.json中的示例"""
    try:
        # 获取数据
        data = request.form
        index = int(data.get('index'))
        caption = data.get('caption')
        style = data.get('style')
        emotion_tags = data.get('emotion_tags', '').split(',')
        emotion_tags = [tag.strip() for tag in emotion_tags if tag.strip()]
        
        # 获取数据库路径和文件
        database_dir = "database"
        examples_file = os.path.join(database_dir, "examples.json")
        
        # 检查文件是否存在
        if not os.path.exists(examples_file):
            return jsonify({"success": False, "error": "示例文件不存在"}), 404
        
        # 读取文件
        examples = []
        with open(examples_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        # 检查索引是否有效
        if index < 0 or index >= len(examples):
            return jsonify({"success": False, "error": "无效的索引"}), 400
        
        # 更新示例
        examples[index]['caption'] = caption
        examples[index]['style'] = style
        examples[index]['emotion_tags'] = emotion_tags
        
        # 保存文件
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "success": True, 
            "message": "示例已更新",
            "need_rebuild": True
        })
        
    except Exception as e:
        print(f"更新示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # 启动应用
    app.run(debug=True, port=5000) 