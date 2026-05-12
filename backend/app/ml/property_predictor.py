"""性能预测模型 - 使用多种ML算法预测合金性能"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json


class PropertyPredictor:
    """合金性能预测器"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        初始化预测器
        
        Args:
            model_type: 模型类型 ('random_forest', 'gradient_boosting', 'neural_network', 'ridge', 'lasso', 'svr')
        """
        self.model_type = model_type
        self.model = None
        self.target_property = None
        self.feature_names = None
        self.metrics = {}
        
    def build_model(self, n_features: int):
        """构建模型"""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,  # 减少树的数量
                max_depth=10,      # 减小树深度
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1          # 使用单线程避免Windows死锁
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            # 根据特征数调整隐藏层
            hidden_layer_sizes = (max(50, n_features * 2), max(25, n_features))
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=0.1, random_state=42)
        elif self.model_type == 'svr':
            self.model = SVR(kernel='rbf', C=1.0, gamma='scale')
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        target_property: str
    ) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            feature_names: 特征名称
            target_property: 目标属性名称
            
        Returns:
            训练指标字典
        """
        self.target_property = target_property
        self.feature_names = feature_names
        
        # 构建模型
        self.build_model(X_train.shape[1])
        
        # 训练
        print(f"\n训练 {self.model_type} 模型...")
        self.model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 计算指标
        self.metrics = {
            'model_type': self.model_type,
            'target_property': target_property,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(feature_names),
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'target_mean': float(np.mean(y_train)),
            'target_std': float(np.std(y_train)),
            'target_min': float(np.min(y_train)),
            'target_max': float(np.max(y_train))
        }
        
        # 特征重要性（仅适用于树模型）
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.metrics['feature_importance'] = {
                name: float(imp) 
                for name, imp in zip(feature_names, importances)
            }
            # 排序
            sorted_importance = sorted(
                self.metrics['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.metrics['top_features'] = sorted_importance[:10]
        
        # 打印结果
        print(f"\n训练完成!")
        print(f"  训练集 R²: {self.metrics['train_r2']:.4f}")
        print(f"  测试集 R²: {self.metrics['test_r2']:.4f}")
        print(f"  测试集 RMSE: {self.metrics['test_rmse']:.4f}")
        print(f"  测试集 MAE: {self.metrics['test_mae']:.4f}")
        
        if 'top_features' in self.metrics:
            print(f"\n  Top 5 重要特征:")
            for name, importance in self.metrics['top_features'][:5]:
                print(f"    {name}: {importance:.4f}")
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """预测单个样本"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 构建特征向量
        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        return self.model.predict(X)[0]
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'target_property': self.target_property,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PropertyPredictor':
        """加载模型"""
        model_data = joblib.load(filepath)
        predictor = cls(model_data['model_type'])
        predictor.model = model_data['model']
        predictor.target_property = model_data['target_property']
        predictor.feature_names = model_data['feature_names']
        predictor.metrics = model_data['metrics']
        return predictor
    
    def get_prediction_interval(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取预测区间（仅适用于随机森林）
        
        Args:
            X: 输入特征
            confidence: 置信水平
            
        Returns:
            (lower_bound, upper_bound)
        """
        if self.model_type != 'random_forest':
            raise ValueError("预测区间仅支持随机森林模型")
        
        # 获取所有树的预测
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # 计算分位数
        alpha = (1 - confidence) / 2
        lower = np.percentile(predictions, alpha * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)
        
        return lower, upper


class MultiPropertyPredictor:
    """多性能预测器 - 同时预测多个性能指标"""
    
    def __init__(self):
        self.predictors: Dict[str, PropertyPredictor] = {}
        self.target_properties: List[str] = []
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        target_properties: List[str],
        model_type: str = 'random_forest'
    ) -> Dict[str, Dict]:
        """训练多个预测模型"""
        self.target_properties = target_properties
        all_metrics = {}
        
        print(f"\n训练 {len(target_properties)} 个性能预测模型...")
        
        for i, target in enumerate(target_properties):
            print(f"\n{'='*50}")
            print(f"[{i+1}/{len(target_properties)}] 训练 {target} 预测模型")
            print(f"{'='*50}")
            
            predictor = PropertyPredictor(model_type)
            metrics = predictor.train(
                X_train, y_train[:, i],
                X_test, y_test[:, i],
                feature_names, target
            )
            
            self.predictors[target] = predictor
            all_metrics[target] = metrics
        
        return all_metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """预测所有性能"""
        predictions = {}
        for target, predictor in self.predictors.items():
            predictions[target] = predictor.predict(X)
        return predictions
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, float]:
        """预测单个样本的所有性能"""
        predictions = {}
        for target, predictor in self.predictors.items():
            predictions[target] = predictor.predict_single(features)
        return predictions
    
    def save(self, directory: str):
        """保存所有模型"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for target, predictor in self.predictors.items():
            filepath = f"{directory}/{target}_model.pkl"
            predictor.save(filepath)
        
        # 保存配置
        config = {
            'target_properties': self.target_properties
        }
        with open(f"{directory}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n所有模型已保存到: {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'MultiPropertyPredictor':
        """加载所有模型"""
        import os
        
        with open(f"{directory}/config.json", 'r') as f:
            config = json.load(f)
        
        multi_predictor = cls()
        multi_predictor.target_properties = config['target_properties']
        
        for target in config['target_properties']:
            filepath = f"{directory}/{target}_model.pkl"
            if os.path.exists(filepath):
                multi_predictor.predictors[target] = PropertyPredictor.load(filepath)
        
        return multi_predictor