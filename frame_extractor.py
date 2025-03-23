import cv2
import numpy as np
from typing import List, Tuple
import os
from config import NUM_FRAMES

class FrameExtractor:
    """视频关键帧提取类"""
    
    def __init__(self, video_path: str):
        """
        初始化关键帧提取器
        
        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def extract_frames(self, num_frames: int = NUM_FRAMES) -> List[np.ndarray]:
        """
        均匀提取指定数量的关键帧
        
        Args:
            num_frames: 要提取的关键帧数量
            
        Returns:
            关键帧列表
        """
        if num_frames > self.total_frames:
            num_frames = self.total_frames
            
        # 均匀分布的帧索引
        frame_indices = [int(i * self.total_frames / num_frames) for i in range(num_frames)]
        
        frames = []
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
        
        self.cap.release()
        return frames
    
    def get_video_info(self) -> dict:
        """
        获取视频信息
        
        Returns:
            包含视频信息的字典
        """
        return {
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration": self.total_frames / self.fps if self.fps > 0 else 0
        } 