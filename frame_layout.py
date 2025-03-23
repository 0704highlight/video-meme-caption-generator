import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image
from config import LayoutType

class FrameLayout:
    """关键帧布局处理类"""
    
    def __init__(self, frames: List[np.ndarray]):
        """
        初始化布局处理器
        
        Args:
            frames: 关键帧列表
        """
        self.frames = frames
        self.num_frames = len(frames)
        
    def create_layout(self, layout_type: str) -> np.ndarray:
        """
        根据布局类型创建关键帧布局
        
        Args:
            layout_type: 布局类型
            
        Returns:
            布局后的图像
        """
        if layout_type == LayoutType.GRID:
            return self._create_grid_layout()
        elif layout_type == LayoutType.LIST:
            return self._create_list_layout()
        elif layout_type == LayoutType.LIST_GRID_TOP:
            return self._create_mixed_layout("top")
        elif layout_type == LayoutType.LIST_GRID_MIDDLE:
            return self._create_mixed_layout("middle")
        elif layout_type == LayoutType.LIST_GRID_BOTTOM:
            return self._create_mixed_layout("bottom")
        else:
            raise ValueError(f"不支持的布局类型: {layout_type}")
    
    def _create_grid_layout(self) -> np.ndarray:
        """
        创建网格布局
        
        Returns:
            网格布局图像
        """
        # 确定网格大小
        if self.num_frames <= 2:
            rows, cols = 1, self.num_frames
        else:
            rows, cols = 2, 2
        
        # 确保所有帧具有相同大小
        frames = [cv2.resize(frame, (300, 300)) for frame in self.frames]
        
        # 创建空白画布
        canvas = np.ones((rows * 300, cols * 300, 3), dtype=np.uint8) * 255
        
        # 将帧放入画布
        for i in range(min(rows * cols, self.num_frames)):
            r, c = i // cols, i % cols
            canvas[r*300:(r+1)*300, c*300:(c+1)*300] = frames[i]
        
        return canvas
    
    def _create_list_layout(self) -> np.ndarray:
        """
        创建列表布局
        
        Returns:
            列表布局图像
        """
        # 确保所有帧具有相同宽度
        width = 500
        frames = []
        for frame in self.frames:
            h, w = frame.shape[:2]
            new_h = int(h * width / w)
            frames.append(cv2.resize(frame, (width, new_h)))
        
        # 计算总高度并创建画布
        total_height = sum(frame.shape[0] for frame in frames)
        canvas = np.ones((total_height, width, 3), dtype=np.uint8) * 255
        
        # 将帧放入画布
        y_offset = 0
        for frame in frames:
            h = frame.shape[0]
            canvas[y_offset:y_offset+h, :] = frame
            y_offset += h
        
        return canvas
    
    def _create_mixed_layout(self, position: str) -> np.ndarray:
        """
        创建混合布局（网格嵌入列表）
        
        Args:
            position: 网格位置 ("top", "middle", "bottom")
            
        Returns:
            混合布局图像
        """
        # 创建网格部分
        grid = self._create_grid_layout()
        grid_height, grid_width = grid.shape[:2]
        
        # 创建列表部分（每帧高度相同）
        width = grid_width
        frame_height = 150
        list_frames = [cv2.resize(frame, (width, frame_height)) for frame in self.frames]
        
        # 计算总高度并创建画布
        total_list_height = self.num_frames * frame_height
        total_height = total_list_height + grid_height
        canvas = np.ones((total_height, width, 3), dtype=np.uint8) * 255
        
        # 根据位置放置网格和列表
        if position == "top":
            # 网格在顶部，列表在底部
            canvas[:grid_height, :] = grid
            y_offset = grid_height
            for frame in list_frames:
                canvas[y_offset:y_offset+frame_height, :] = frame
                y_offset += frame_height
        
        elif position == "bottom":
            # 列表在顶部，网格在底部
            y_offset = 0
            for frame in list_frames:
                canvas[y_offset:y_offset+frame_height, :] = frame
                y_offset += frame_height
            canvas[y_offset:y_offset+grid_height, :] = grid
        
        elif position == "middle":
            # 列表分成两部分，网格在中间
            frames_per_part = self.num_frames // 2
            y_offset = 0
            
            # 上半部分列表
            for i in range(frames_per_part):
                canvas[y_offset:y_offset+frame_height, :] = list_frames[i]
                y_offset += frame_height
            
            # 中间网格
            canvas[y_offset:y_offset+grid_height, :] = grid
            y_offset += grid_height
            
            # 下半部分列表
            for i in range(frames_per_part, self.num_frames):
                canvas[y_offset:y_offset+frame_height, :] = list_frames[i]
                y_offset += frame_height
        
        return canvas
        
    def save_layout(self, layout: np.ndarray, output_path: str) -> None:
        """
        保存布局图像
        
        Args:
            layout: 布局图像
            output_path: 输出路径
        """
        cv2.imwrite(output_path, layout) 