import cv2
import numpy as np
from typing import List, Tuple
import os
from config import LayoutType

class VideoProcessor:
    """视频处理器，用于提取关键帧并组织布局"""
    
    def __init__(self, video_path=None):
        """初始化视频处理器
        
        Args:
            video_path: 可选的视频文件路径
        """
        self.video_path = video_path
    
    def extract_frames(self, num_frames: int = 4) -> List[np.ndarray]:
        """
        从当前视频中提取帧
        
        Args:
            num_frames: 要提取的帧数，默认为4
            
        Returns:
            帧列表
        """
        if not self.video_path or not os.path.exists(self.video_path):
            raise FileNotFoundError(f"视频文件不存在或未指定: {self.video_path}")
        
        return self.extract_keyframes(self.video_path, num_frames)
    
    def extract_keyframes(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """
        从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            num_frames: 要提取的帧数，默认为8，支持新的复合布局
            
        Returns:
            关键帧列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件 {video_path} 不存在")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps
        
        # 如果帧数太少，调整要提取的帧数
        if frame_count < num_frames:
            num_frames = frame_count
        
        # 计算要提取的帧的索引
        indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
        
        # 提取帧
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        # 释放视频
        cap.release()
        
        # 如果没有成功提取帧，抛出异常
        if len(frames) == 0:
            raise ValueError(f"无法从视频 {video_path} 中提取帧")
        
        return frames
    
    def organize_frames(self, frames: List[np.ndarray], layout_type) -> List[np.ndarray]:
        """
        根据布局类型组织帧
        
        Args:
            frames: 关键帧列表
            layout_type: 布局类型(LayoutType枚举)
            
        Returns:
            组织后的帧列表
        """
        # 如果帧数为0，返回空列表
        if len(frames) == 0:
            return []
        
        # 根据布局类型组织帧
        if layout_type == LayoutType.GRID:
            # 网格布局，返回前4帧
            return frames[:min(4, len(frames))]
        elif layout_type == LayoutType.HORIZONTAL:
            # 水平布局，返回前4帧
            return frames[:min(4, len(frames))]
        elif layout_type == LayoutType.VERTICAL:
            # 垂直布局，返回前4帧
            return frames[:min(4, len(frames))]
        elif layout_type == LayoutType.SINGLE:
            # 单图布局，返回第一帧
            return [frames[0]]
        elif layout_type == LayoutType.LIST:
            # 列表布局，最多返回6帧
            return frames[:min(6, len(frames))]
        elif layout_type == LayoutType.GRID_IN_LIST:
            # 网格嵌入列表布局，最多返回8帧
            return frames[:min(8, len(frames))]
        else:
            # 默认返回所有帧，最多8帧
            return frames[:min(8, len(frames))]
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件 {video_path} 不存在")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        
        # 释放视频
        cap.release()
        
        return {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "file_name": os.path.basename(video_path),
            "file_path": video_path
        } 