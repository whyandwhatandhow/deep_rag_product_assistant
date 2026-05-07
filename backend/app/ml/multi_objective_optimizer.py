"""多目标优化模块 - 实现合金成分和工艺的多目标优化"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from scipy.optimize import minimize, differential_evolution
import warnings


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, predictor, composition_bounds: Dict[str, Tuple[float, float]] = None):
        """
        初始化优化器
        
        Args:
            predictor: 性能预测模型（MultiPropertyPredictor）
            composition_bounds: 成分范围约束
        """
        self.predictor = predictor
        self.composition_bounds = composition_bounds or self._default_bounds()
        self.feature_names = None
        
    def _default_bounds(self) -> Dict[str, Tuple[float, float]]:
        """默认成分范围"""
        return {
            'zinc': (85.0, 99.9),
            'magnesium': (0.0, 10.0),
            'aluminum': (0.0, 15.0),
            'copper': (0.0, 5.0),
            'silver': (0.0, 1.0),
            'melting_temperature': (400.0, 700.0),
            'annealing_temperature': (200.0, 400.0),
            'aging_temperature': (80.0, 200.0),
            'aging_time': (1.0, 100.0)
        }
    
    def set_feature_names(self, feature_names: List[str]):
        """设置特征名称"""
        self.feature_names = feature_names
    
    def _build_feature_vector(self, composition: Dict[str, float]) -> np.ndarray:
        """构建特征向量"""
        if self.feature_names is None:
            raise ValueError("请先设置特征名称")
        
        features = []
        for name in self.feature_names:
            features.append(composition.get(name, 0.0))
        
        return np.array(features).reshape(1, -1)
    
    def optimize_composition(
        self,
        objectives: Dict[str, str],  # {'ultimate_tensile_strength': 'max', 'elongation': 'max'}
        constraints: Dict[str, Tuple[float, float]] = None,
        method: str = 'differential_evolution',
        max_iter: int = 100
    ) -> Dict:
        """
        优化合金成分
        
        Args:
            objectives: 目标字典，key为性能名称，value为'max'或'min'
            constraints: 额外约束条件
            method: 优化方法
            max_iter: 最大迭代次数
            
        Returns:
            优化结果字典
        """
        print(f"\n开始多目标优化...")
        print(f"优化目标: {objectives}")
        
        # 构建目标函数
        def objective_function(x):
            # x是成分向量
            composition = {name: val for name, val in zip(self.feature_names, x)}
            
            # 预测性能
            predictions = self.predictor.predict_single(composition)
            
            # 计算目标函数值（加权求和）
            value = 0.0
            for prop, direction in objectives.items():
                if prop in predictions:
                    if direction == 'max':
                        value -= predictions[prop]  # 最大化 = 最小化负值
                    else:
                        value += predictions[prop]  # 最小化
            
            return value
        
        # 构建边界
        bounds = []
        for name in self.feature_names:
            if name in self.composition_bounds:
                bounds.append(self.composition_bounds[name])
            else:
                bounds.append((0.0, 100.0))  # 默认边界
        
        # 执行优化
        if method == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=max_iter,
                seed=42,
                workers=1,  # 使用单进程避免pickle错误
                polish=True
            )
        else:
            # 使用随机初始点
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter}
            )
        
        # 构建优化结果
        optimal_composition = {name: val for name, val in zip(self.feature_names, result.x)}
        predicted_properties = self.predictor.predict_single(optimal_composition)
        
        optimization_result = {
            'success': result.success,
            'optimal_composition': optimal_composition,
            'predicted_properties': predicted_properties,
            'objectives': objectives,
            'optimization_value': result.fun,
            'n_iterations': result.nit if hasattr(result, 'nit') else max_iter
        }
        
        print(f"\n优化完成!")
        print(f"  最优成分:")
        for elem, val in optimal_composition.items():
            if val > 0.01:  # 只显示主要成分
                print(f"    {elem}: {val:.2f}")
        print(f"  预测性能:")
        for prop, val in predicted_properties.items():
            print(f"    {prop}: {val:.2f}")
        
        return optimization_result
    
    def pareto_optimization(
        self,
        objectives: List[str],  # ['ultimate_tensile_strength', 'elongation']
        n_points: int = 50
    ) -> List[Dict]:
        """
        帕累托前沿优化 - 找到多个目标之间的最佳权衡
        
        Args:
            objectives: 目标性能列表
            n_points: 帕累托前沿上的点数
            
        Returns:
            帕累托最优解列表
        """
        print(f"\n计算帕累托前沿...")
        print(f"目标: {objectives}")
        
        if len(objectives) != 2:
            raise ValueError("帕累托优化目前只支持2个目标")
        
        # 随机采样
        bounds = []
        for name in self.feature_names:
            if name in self.composition_bounds:
                bounds.append(self.composition_bounds[name])
            else:
                bounds.append((0.0, 100.0))
        
        # 生成随机样本
        n_samples = 1000
        samples = []
        for _ in range(n_samples):
            sample = {name: np.random.uniform(b[0], b[1]) for name, b in zip(self.feature_names, bounds)}
            samples.append(sample)
        
        # 预测所有样本的性能
        predictions = []
        for sample in samples:
            pred = self.predictor.predict_single(sample)
            predictions.append(pred)
        
        # 提取目标值
        obj1_values = np.array([p.get(objectives[0], 0) for p in predictions])
        obj2_values = np.array([p.get(objectives[1], 0) for p in predictions])
        
        # 找到帕累托前沿
        pareto_indices = self._find_pareto_front(obj1_values, obj2_values, maximize=True)
        
        # 构建帕累托解
        pareto_solutions = []
        for idx in pareto_indices[:n_points]:
            pareto_solutions.append({
                'composition': samples[idx],
                'properties': predictions[idx],
                objectives[0]: obj1_values[idx],
                objectives[1]: obj2_values[idx]
            })
        
        print(f"  找到 {len(pareto_solutions)} 个帕累托最优解")
        
        return pareto_solutions
    
    def _find_pareto_front(
        self,
        obj1: np.ndarray,
        obj2: np.ndarray,
        maximize: bool = True
    ) -> List[int]:
        """找到帕累托前沿的索引"""
        n = len(obj1)
        is_pareto = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i != j and is_pareto[j]:
                    if maximize:
                        # 如果j在两个目标上都优于或等于i，则i不是帕累托最优
                        if obj1[j] >= obj1[i] and obj2[j] >= obj2[i]:
                            if obj1[j] > obj1[i] or obj2[j] > obj2[i]:
                                is_pareto[i] = False
                                break
                    else:
                        if obj1[j] <= obj1[i] and obj2[j] <= obj2[i]:
                            if obj1[j] < obj1[i] or obj2[j] < obj2[i]:
                                is_pareto[i] = False
                                break
        
        pareto_indices = np.where(is_pareto)[0]
        
        # 按第一个目标排序
        sorted_indices = pareto_indices[np.argsort(obj1[pareto_indices])]
        
        return sorted_indices.tolist()
    
    def sensitivity_analysis(
        self,
        base_composition: Dict[str, float],
        property_name: str,
        element_name: str,
        variation_range: Tuple[float, float] = None,
        n_points: int = 20
    ) -> Dict:
        """
        敏感性分析 - 分析某个元素对性能的影响
        
        Args:
            base_composition: 基础成分
            property_name: 要分析的性能
            element_name: 要分析的元素
            variation_range: 变化范围
            n_points: 采样点数
            
        Returns:
            敏感性分析结果
        """
        print(f"\n进行敏感性分析...")
        print(f"分析 {element_name} 对 {property_name} 的影响")
        
        if variation_range is None:
            current_val = base_composition.get(element_name, 0)
            variation_range = (max(0, current_val - 5), min(100, current_val + 5))
        
        # 生成变化范围
        values = np.linspace(variation_range[0], variation_range[1], n_points)
        predictions = []
        
        for val in values:
            composition = base_composition.copy()
            composition[element_name] = val
            pred = self.predictor.predict_single(composition)
            predictions.append(pred.get(property_name, 0))
        
        # 计算敏感性
        sensitivity = np.polyfit(values, predictions, 1)[0]
        
        result = {
            'element': element_name,
            'property': property_name,
            'base_composition': base_composition,
            'variation_range': variation_range,
            'values': values.tolist(),
            'predictions': predictions,
            'sensitivity': float(sensitivity),
            'max_change': max(predictions) - min(predictions)
        }
        
        print(f"  敏感性系数: {sensitivity:.4f}")
        print(f"  性能变化范围: {min(predictions):.2f} - {max(predictions):.2f}")
        
        return result