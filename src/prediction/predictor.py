import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.models import get_model
from src.utils.logger import setup_logger, log_info, log_error

class ImagePredictor:
    def __init__(self):
        self.model = None
        self.device = None
        self.img_size = 224
        self.transform = None
        self.logger = setup_logger("ImagePredictor", "INFO")
    
    def load_model(self, model_path, model_name='efficientnet_b0'):
        """
        加载模型
        
        Args:
            model_path (str): 模型路径
            model_name (str): 模型名称
        """
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = get_model(model_name, num_classes=2)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 创建图像变换
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        log_info(f"模型 '{model_name}' 已加载到 {self.device}", self.logger)
    
    def predict_image(self, image_path):
        """
        预测图像
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            预测结果和概率
        """
        if self.model is None:
            log_error("模型尚未加载，请先调用 load_model", self.logger)
            raise ValueError("模型尚未加载，请先调用 load_model")
        
        try:
            # 加载并处理图像
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            
            # 应用变换
            img_tensor = self.transform(image=img_np)['image']
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probs).item()
            
            log_info(f"图像预测完成: {image_path}, 预测类别: {pred_class}", self.logger)
            
            # 返回结果
            return {
                'class': '伪造' if pred_class == 1 else '真实',
                'fake_prob': float(probs[1].item()) * 100,  # 伪造概率
                'real_prob': float(probs[0].item()) * 100   # 真实概率
            }
        except Exception as e:
            log_error(f"预测图像 {image_path} 时出错: {str(e)}", self.logger)
            raise

# 全局预测器实例
predictor = ImagePredictor()

def load_model(model_path, model_name='efficientnet_b0'):
    """全局函数，用于兼容现有代码"""
    predictor.load_model(model_path, model_name)

def predict_image(image_path):
    """全局函数，用于兼容现有代码"""
    return predictor.predict_image(image_path)