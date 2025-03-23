import cv2
import numpy as np
from typing import List
from PIL import Image, ImageDraw, ImageFont
import os
from config import LayoutType
import re

class ImageGenerator:
    """图像生成器，用于生成带有配文的图像"""
    
    def __init__(self):
        """初始化图像生成器"""
        # 尝试加载字体
        try:
            # 对于中文，需要一个支持中文的字体
            font_path = os.path.join(os.path.dirname(__file__), "static", "fonts", "msyh.ttc")
            if os.path.exists(font_path):
                self.font_path = font_path
            else:
                # 使用系统默认字体
                self.font_path = None
                print("警告：未找到中文字体文件，将使用系统默认字体")
        except Exception as e:
            self.font_path = None
            print(f"加载字体时出错: {e}")
    
    def add_caption(self, image, caption):
        """
        在图像上添加配文
        
        Args:
            image: OpenCV图像(numpy数组)或PIL图像
            caption: 配文文本
            
        Returns:
            添加了配文的图像(与输入格式相同)
        """
        # 检查图像类型
        is_pil_image = isinstance(image, Image.Image)
        
        # 转换OpenCV图像为PIL图像
        if not is_pil_image:
            # 获取图像尺寸
            if len(image.shape) == 3:
                height, width, _ = image.shape
            else:
                height, width = image.shape
                
            # 转换为PIL图像
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR转RGB
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
            else:
                # 灰度图像或其他类型
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
            width, height = pil_image.size
            
        # 增加图像底部的空间用于配文
        caption_space = 150
        new_height = height + caption_space
        new_image = Image.new('RGB', (width, new_height), (255, 255, 255))
        new_image.paste(pil_image, (0, 0))
        
        # 添加配文
        self._add_caption(new_image, caption, position="bottom")
        
        # 如果输入是OpenCV图像，转换回OpenCV格式
        if not is_pil_image:
            # PIL转OpenCV
            new_image_np = np.array(new_image)
            # RGB转BGR
            if len(new_image_np.shape) == 3 and new_image_np.shape[2] == 3:
                new_image_np = cv2.cvtColor(new_image_np, cv2.COLOR_RGB2BGR)
            return new_image_np
        else:
            return new_image
    
    def generate_image(self, frames: List[np.ndarray], caption: str, layout_type, output_path: str):
        """
        生成带有配文的图像
        
        Args:
            frames: 关键帧列表
            caption: 配文
            layout_type: 布局类型(LayoutType枚举)
            output_path: 输出路径
        """
        # 如果没有帧，抛出异常
        if len(frames) == 0:
            raise ValueError("没有帧可以生成图像")
        
        # 根据布局类型生成图像
        if layout_type == LayoutType.GRID:
            # 网格布局
            image = self._create_grid_layout(frames, caption)
        elif layout_type == LayoutType.HORIZONTAL:
            # 水平布局
            image = self._create_horizontal_layout(frames, caption)
        elif layout_type == LayoutType.VERTICAL:
            # 垂直布局
            image = self._create_vertical_layout(frames, caption)
        elif layout_type == LayoutType.SINGLE:
            # 单图布局
            image = self._create_single_layout(frames[0], caption)
        elif layout_type == LayoutType.LIST:
            # 列表布局（带有序号和箭头的时间序列）
            image = self._create_list_layout(frames, caption)
        elif layout_type == LayoutType.GRID_IN_LIST:
            # 网格嵌入列表布局
            image = self._create_grid_in_list_layout(frames, caption)
        else:
            # 默认使用网格布局
            image = self._create_grid_layout(frames, caption)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 保存图像
        image.save(output_path)
    
    def _create_grid_layout(self, frames: List[np.ndarray], caption: str) -> Image.Image:
        """
        创建网格布局图像
        
        Args:
            frames: 关键帧列表
            caption: 配文
            
        Returns:
            PIL图像
        """
        # 获取帧数量，最多使用4帧
        num_frames = min(len(frames), 4)
        
        # 确定网格大小
        if num_frames <= 1:
            grid_size = (1, 1)
        elif num_frames <= 2:
            grid_size = (1, 2)
        elif num_frames <= 4:
            grid_size = (2, 2)
        
        # 计算图像大小
        frame_height, frame_width = frames[0].shape[:2]
        image_width = grid_size[1] * frame_width
        image_height = grid_size[0] * frame_height + 150  # 增加额外空间用于配文
        
        # 创建空白图像
        image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        
        # 放置帧
        for i in range(num_frames):
            row = i // grid_size[1]
            col = i % grid_size[1]
            
            # 将OpenCV的BGR图像转换为PIL的RGB图像
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # 放置在图像上
            x = col * frame_width
            y = row * frame_height
            image.paste(frame_pil, (x, y))
        
        # 添加配文
        self._add_caption(image, caption, position="bottom")
        
        return image
    
    def _create_horizontal_layout(self, frames: List[np.ndarray], caption: str) -> Image.Image:
        """
        创建水平布局图像
        
        Args:
            frames: 关键帧列表
            caption: 配文
            
        Returns:
            PIL图像
        """
        # 获取帧数量，最多使用4帧
        num_frames = min(len(frames), 4)
        
        # 计算图像大小
        frame_height, frame_width = frames[0].shape[:2]
        image_width = num_frames * frame_width
        image_height = frame_height + 150  # 增加额外空间用于配文
        
        # 创建空白图像
        image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        
        # 放置帧
        for i in range(num_frames):
            # 将OpenCV的BGR图像转换为PIL的RGB图像
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # 放置在图像上
            x = i * frame_width
            y = 0
            image.paste(frame_pil, (x, y))
        
        # 添加配文
        self._add_caption(image, caption, position="bottom")
        
        return image
    
    def _create_vertical_layout(self, frames: List[np.ndarray], caption: str) -> Image.Image:
        """
        创建垂直布局图像
        
        Args:
            frames: 关键帧列表
            caption: 配文
            
        Returns:
            PIL图像
        """
        # 获取帧数量，最多使用4帧
        num_frames = min(len(frames), 4)
        
        # 计算图像大小
        frame_height, frame_width = frames[0].shape[:2]
        image_width = frame_width
        image_height = num_frames * frame_height + 150  # 增加额外空间用于配文
        
        # 创建空白图像
        image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        
        # 放置帧
        for i in range(num_frames):
            # 将OpenCV的BGR图像转换为PIL的RGB图像
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # 放置在图像上
            x = 0
            y = i * frame_height
            image.paste(frame_pil, (x, y))
        
        # 添加配文
        self._add_caption(image, caption, position="bottom")
        
        return image
    
    def _create_single_layout(self, frame: np.ndarray, caption: str) -> Image.Image:
        """
        创建单图布局图像
        
        Args:
            frame: 关键帧
            caption: 配文
            
        Returns:
            PIL图像
        """
        # 计算图像大小
        frame_height, frame_width = frame.shape[:2]
        image_width = frame_width
        image_height = frame_height + 150  # 增加额外空间用于配文
        
        # 创建空白图像
        image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        
        # 将OpenCV的BGR图像转换为PIL的RGB图像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # 放置在图像上
        image.paste(frame_pil, (0, 0))
        
        # 添加配文
        self._add_caption(image, caption, position="bottom")
        
        return image
    
    def _create_list_layout(self, frames: List[np.ndarray], caption: str) -> Image.Image:
        """
        创建列表布局图像，带有时间序列指示（箭头和序号）
        
        Args:
            frames: 关键帧列表
            caption: 配文
            
        Returns:
            PIL图像
        """
        # 获取帧数量，最多使用6帧
        num_frames = min(len(frames), 6)
        
        # 计算一个帧的尺寸
        frame_height, frame_width = frames[0].shape[:2]
        
        # 在帧之间添加箭头的空间
        arrow_space = 50
        
        # 计算图像大小
        image_width = frame_width
        image_height = num_frames * frame_height + (num_frames - 1) * arrow_space + 100  # 额外空间用于配文
        
        # 创建空白图像
        image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 放置帧和箭头
        for i in range(num_frames):
            # 将OpenCV的BGR图像转换为PIL的RGB图像
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # 放置在图像上
            y = i * (frame_height + arrow_space)
            image.paste(frame_pil, (0, y))
            
            # 在帧上添加序号
            font_size = int(frame_height * 0.1)
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, font_size)
                else:
                    try:
                        font = ImageFont.truetype("simhei.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # 绘制序号背景
            number_bg_size = font_size * 2
            draw.rectangle(
                [(10, y + 10), (10 + number_bg_size, y + 10 + number_bg_size)],
                fill=(0, 0, 0, 200)
            )
            
            # 绘制序号
            draw.text(
                (10 + number_bg_size//2 - font_size//2, y + 10 + number_bg_size//2 - font_size//2),
                f"{i+1}",
                font=font,
                fill=(255, 255, 255)
            )
            
            # 如果不是最后一帧，添加箭头
            if i < num_frames - 1:
                arrow_y = y + frame_height + arrow_space // 2
                # 绘制箭头线
                draw.line(
                    [(image_width // 2, arrow_y - arrow_space // 2 + 10),
                     (image_width // 2, arrow_y + arrow_space // 2 - 10)],
                    fill=(0, 0, 0),
                    width=3
                )
                # 绘制箭头头部
                draw.polygon(
                    [(image_width // 2 - 10, arrow_y + arrow_space // 2 - 15),
                     (image_width // 2 + 10, arrow_y + arrow_space // 2 - 15),
                     (image_width // 2, arrow_y + arrow_space // 2)],
                    fill=(0, 0, 0)
                )
        
        # 添加配文
        self._add_caption(image, caption, position="bottom")
        
        return image
    
    def _create_grid_in_list_layout(self, frames: List[np.ndarray], caption: str) -> Image.Image:
        """
        创建网格嵌入列表布局图像，顶部显示2x2网格，底部显示时间序列
        
        Args:
            frames: 关键帧列表
            caption: 配文
            
        Returns:
            PIL图像
        """
        # 获取帧数量，至少需要4帧用于网格，最多额外4帧用于列表
        if len(frames) < 4:
            # 如果帧数不足4，复制最后一帧填充
            while len(frames) < 4:
                frames.append(frames[-1])
        
        num_frames = min(len(frames), 8)
        grid_frames = frames[:4]  # 前4帧用于网格
        list_frames = frames[4:num_frames]  # 剩余帧用于列表
        
        # 计算一个帧的尺寸
        frame_height, frame_width = frames[0].shape[:2]
        
        # 调整网格中的帧大小为原来的一半
        grid_frame_width = frame_width // 2
        grid_frame_height = frame_height // 2
        
        # 计算网格部分的高度
        grid_height = grid_frame_height * 2
        
        # 在列表帧之间添加箭头的空间
        arrow_space = 30
        
        # 计算列表部分的高度
        list_height = 0
        if list_frames:
            list_height = len(list_frames) * frame_height + (len(list_frames) - 1) * arrow_space
        
        # 网格和列表之间的间隔
        section_space = 50
        
        # 计算总图像大小
        image_width = frame_width
        image_height = grid_height + section_space + list_height + 100  # 额外空间用于配文
        
        # 创建空白图像
        image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 放置网格帧
        for i in range(len(grid_frames)):
            row = i // 2
            col = i % 2
            
            # 将OpenCV的BGR图像转换为PIL的RGB图像并调整大小
            frame_rgb = cv2.cvtColor(grid_frames[i], cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((grid_frame_width, grid_frame_height))
            
            # 放置在图像上
            x = col * grid_frame_width
            y = row * grid_frame_height
            image.paste(frame_pil, (x, y))
            
            # 在帧上添加序号
            font_size = int(grid_frame_height * 0.1)
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, font_size)
                else:
                    try:
                        font = ImageFont.truetype("simhei.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # 绘制序号背景
            number_bg_size = font_size * 2
            draw.rectangle(
                [(x + 5, y + 5), (x + 5 + number_bg_size, y + 5 + number_bg_size)],
                fill=(0, 0, 0, 200)
            )
            
            # 绘制序号
            draw.text(
                (x + 5 + number_bg_size//2 - font_size//2, y + 5 + number_bg_size//2 - font_size//2),
                f"{i+1}",
                font=font,
                fill=(255, 255, 255)
            )
        
        # 如果有列表帧，在网格和列表之间添加分隔线
        if list_frames:
            # 添加标题
            try:
                title_font_size = int(frame_width * 0.04)
                if self.font_path:
                    title_font = ImageFont.truetype(self.font_path, title_font_size)
                else:
                    try:
                        title_font = ImageFont.truetype("simhei.ttf", title_font_size)
                    except:
                        title_font = ImageFont.load_default()
            except:
                title_font = ImageFont.load_default()
            
            draw.text(
                (10, grid_height + 10),
                "关键场景详情:",
                font=title_font,
                fill=(0, 0, 0)
            )
            
            # 绘制分隔线
            draw.line(
                [(0, grid_height + section_space // 2),
                 (image_width, grid_height + section_space // 2)],
                fill=(200, 200, 200),
                width=2
            )
            
            # 放置列表帧
            list_start_y = grid_height + section_space
            for i in range(len(list_frames)):
                # 将OpenCV的BGR图像转换为PIL的RGB图像
                frame_rgb = cv2.cvtColor(list_frames[i], cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # 放置在图像上
                y = list_start_y + i * (frame_height + arrow_space)
                image.paste(frame_pil, (0, y))
                
                # 在帧上添加序号
                font_size = int(frame_height * 0.1)
                
                # 绘制序号背景
                number_bg_size = font_size * 2
                draw.rectangle(
                    [(10, y + 10), (10 + number_bg_size, y + 10 + number_bg_size)],
                    fill=(0, 0, 0, 200)
                )
                
                # 绘制序号
                draw.text(
                    (10 + number_bg_size//2 - font_size//2, y + 10 + number_bg_size//2 - font_size//2),
                    f"{i+5}",  # 接着网格的编号继续
                    font=title_font,
                    fill=(255, 255, 255)
                )
                
                # 如果不是最后一帧，添加箭头
                if i < len(list_frames) - 1:
                    arrow_y = y + frame_height + arrow_space // 2
                    # 绘制箭头线
                    draw.line(
                        [(image_width // 2, arrow_y - arrow_space // 2 + 5),
                         (image_width // 2, arrow_y + arrow_space // 2 - 5)],
                        fill=(0, 0, 0),
                        width=2
                    )
                    # 绘制箭头头部
                    draw.polygon(
                        [(image_width // 2 - 8, arrow_y + arrow_space // 2 - 10),
                         (image_width // 2 + 8, arrow_y + arrow_space // 2 - 10),
                         (image_width // 2, arrow_y + arrow_space // 2)],
                        fill=(0, 0, 0)
                    )
        
        # 添加配文
        self._add_caption(image, caption, position="bottom")
        
        return image
    
    def _add_caption(self, image: Image.Image, caption: str, position: str = "bottom"):
        """
        添加配文到图像
        
        Args:
            image: PIL图像
            caption: 配文
            position: 配文位置，可以是"top"或"bottom"
        """
        # 获取图像尺寸
        image_width, image_height = image.size
        
        # 设置字体，尝试从不同位置加载字体
        font_size = int(image_width * 0.04)  # 减小默认字体尺寸
        font_size = max(font_size, 14)  # 设置最小字体大小
        
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
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    pass
        
        # 如果无法加载字体，使用默认字体
        if font is None:
            font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(image)
        
        # 计算适合的文本宽度
        max_width = int(image_width * 0.9)  # 图像宽度的90%
        
        # 清理配文，移除格式符号和emoji
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
        
        # 自动换行处理
        lines = []
        
        # 优化的文本分割逻辑，更好地处理颜文字和中文混合文本
        # 首先尝试按空格分割
        parts = caption.split(' ')
        
        # 处理每个部分
        current_line = ""
        for part in parts:
            # 如果当前部分为空，跳过
            if not part:
                if current_line:
                    current_line += " "
                continue
            
            # 检查当前行加上这个部分是否会超出最大宽度
            test_line = current_line + part + " " if current_line else part + " "
            try:
                text_bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
            except Exception:
                # 如果无法计算确切尺寸，使用估计值
                text_width = font_size * len(test_line) // 2  # 简单估计
            
            if text_width <= max_width:
                current_line = test_line
            else:
                # 如果当前部分自身超过一行宽度，需要进一步分割
                if not current_line:
                    # 将部分按字符分割
                    chars = list(part)
                    char_line = ""
                    for char in chars:
                        # 检查当前行加上这个字符是否会超出最大宽度
                        test_char_line = char_line + char
                        try:
                            text_bbox = draw.textbbox((0, 0), test_char_line, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                        except Exception:
                            # 如果无法计算确切尺寸，使用估计值
                            text_width = font_size * len(test_char_line) // 2  # 简单估计
                        
                        if text_width <= max_width:
                            char_line = test_char_line
                        else:
                            lines.append(char_line)
                            char_line = char
                    
                    if char_line:
                        lines.append(char_line)
                else:
                    # 先保存当前行
                    lines.append(current_line.rstrip())
                    
                    # 处理新部分
                    chars = list(part)
                    char_line = ""
                    for char in chars:
                        # 检查当前行加上这个字符是否会超出最大宽度
                        test_char_line = char_line + char
                        try:
                            text_bbox = draw.textbbox((0, 0), test_char_line, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                        except Exception:
                            # 如果无法计算确切尺寸，使用估计值
                            text_width = font_size * len(test_char_line) // 2
                        
                        if text_width <= max_width:
                            char_line = test_char_line
                        else:
                            lines.append(char_line)
                            char_line = char
                    
                    if char_line:
                        current_line = char_line + " "
                    else:
                        current_line = ""
        
        # 处理最后一行
        if current_line:
            lines.append(current_line.rstrip())
        
        # 确保分割后的行数不为0
        if not lines:
            lines = [caption]
        
        # 计算总文本高度
        total_text_height = 0
        line_heights = []
        line_spacing = font_size * 0.3  # 增加行间距
        
        for line in lines:
            try:
                text_bbox = draw.textbbox((0, 0), line, font=font)
                line_height = text_bbox[3] - text_bbox[1]
            except Exception:
                line_height = font_size * 1.2  # 简单估计
            
            line_heights.append(line_height)
            total_text_height += line_height + line_spacing
        
        # 如果总高度太大，适当缩小字体
        if total_text_height > image_height // 3:  # 不超过图像高度的1/3
            scale_factor = (image_height // 3) / total_text_height
            new_font_size = max(int(font_size * scale_factor), 14)  # 最小字体大小为14
            
            try:
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, new_font_size)
                        break
            except Exception:
                font = ImageFont.load_default()
            
            # 重新计算行高
            line_heights = []
            line_spacing = new_font_size * 0.3
            total_text_height = 0
            
            for line in lines:
                try:
                    text_bbox = draw.textbbox((0, 0), line, font=font)
                    line_height = text_bbox[3] - text_bbox[1]
                except Exception:
                    line_height = new_font_size * 1.2
                
                line_heights.append(line_height)
                total_text_height += line_height + line_spacing
        
        # 增加文本背景高度
        background_padding = 20  # 增加背景内边距
        
        # 计算文字位置
        if position == "top":
            y = 10  # 顶部留出10像素
            
            # 绘制文本背景
            draw.rectangle(
                [(0, y - background_padding // 2), 
                 (image_width, y + total_text_height + background_padding)],
                fill=(0, 0, 0, 220)  # 增加背景不透明度
            )
            
            # 绘制文本
            current_y = y + 5
            for i, line in enumerate(lines):
                try:
                    text_bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                except Exception:
                    text_width = len(line) * font_size // 2
                
                x = (image_width - text_width) // 2  # 居中
                draw.text((x, current_y), line, font=font, fill=(255, 255, 255))
                current_y += line_heights[i] + line_spacing
        else:  # 底部
            # 计算新的垂直位置 - 更靠上一些
            y = image_height - total_text_height - background_padding - int(image_height * 0.05)  # 上移5%的图像高度
            
            # 绘制文本背景
            draw.rectangle(
                [(0, y - background_padding), 
                 (image_width, y + total_text_height + background_padding)],
                fill=(0, 0, 0, 220)  # 增加背景不透明度
            )
            
            # 绘制文本
            current_y = y
            for i, line in enumerate(lines):
                try:
                    text_bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                except Exception:
                    text_width = len(line) * font_size // 2
                
                x = (image_width - text_width) // 2  # 居中
                draw.text((x, current_y), line, font=font, fill=(255, 255, 255))
                current_y += line_heights[i] + line_spacing 