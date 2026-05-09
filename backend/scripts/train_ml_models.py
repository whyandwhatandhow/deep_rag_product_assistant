"""训练机器学习模型脚本"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.ml.data_preparation import MLDataPreparation
from app.ml.property_predictor import PropertyPredictor, MultiPropertyPredictor
from app.ml.multi_objective_optimizer import MultiObjectiveOptimizer
import json

# 进度条库
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("警告: 未安装 tqdm，将使用简单进度显示")


def train_single_property_model():
    """训练单性能预测模型"""
    print("\n" + "="*60)
    print("训练单性能预测模型")
    print("="*60)
    
    # 准备数据
    data_prep = MLDataPreparation("alloy_database.db")
    
    try:
        X_train, X_test, y_train, y_test, feature_names = data_prep.prepare_training_data(
            target_property='ultimate_tensile_strength',
            test_size=0.2
        )
        
        # 训练不同模型
        models = ['random_forest', 'gradient_boosting', 'neural_network']
        best_model = None
        best_r2 = -float('inf')
        
        for model_type in models:
            print(f"\n{'='*40}")
            print(f"训练 {model_type} 模型")
            print(f"{'='*40}")
            
            predictor = PropertyPredictor(model_type)
            metrics = predictor.train(
                X_train, y_train, X_test, y_test,
                feature_names, 'ultimate_tensile_strength'
            )
            
            if metrics['test_r2'] > best_r2:
                best_r2 = metrics['test_r2']
                best_model = predictor
        
        # 保存最佳模型
        if best_model:
            best_model.save("models/uts_predictor.pkl")
            print(f"\n最佳模型: {best_model.model_type}, R² = {best_r2:.4f}")
        
    except Exception as e:
        print(f"训练失败: {e}")


def train_multi_property_models():
    """训练多性能预测模型"""
    print("\n" + "="*60)
    print("训练多性能预测模型")
    print("="*60)
    
    # 准备数据
    print("\n[步骤 1/3] 加载数据...")
    data_prep = MLDataPreparation("alloy_database.db")
    
    try:
        print("  - 从数据库读取数据...")
        X_train, X_test, y_train, y_test, feature_names, target_properties = \
            data_prep.prepare_multi_target_data()
        
        print(f"\n[步骤 2/3] 训练模型 ({len(target_properties)} 个目标属性)...")
        # 训练多性能预测器
        multi_predictor = MultiPropertyPredictor()
        
        # 使用进度条训练每个属性模型
        all_metrics = {}
        total_targets = len(target_properties)
        
        if HAS_TQDM:
            target_iterator = tqdm(target_properties, desc="训练进度", bar_format="{l_bar}{bar:30}{r_bar}", ncols=80)
        else:
            target_iterator = target_properties
            print(f"开始训练 {total_targets} 个模型...")
        
        for i, target in enumerate(target_iterator, 1):
            if HAS_TQDM:
                target_iterator.set_description(f"训练 {target}")
            
            if not HAS_TQDM:
                print(f"\n[{i}/{total_targets}] 训练 {target}...")
            
            predictor = PropertyPredictor('random_forest')
            metrics = predictor.train(
                X_train, y_train[:, i-1],
                X_test, y_test[:, i-1],
                feature_names, target
            )
            
            multi_predictor.predictors[target] = predictor
            all_metrics[target] = metrics
            
            if HAS_TQDM:
                target_iterator.set_description(f"完成 {target}")
        
        if HAS_TQDM:
            target_iterator.close()
        
        # 保存模型
        print("\n[步骤 3/3] 保存模型...")
        multi_predictor.save("models/multi_property")
        
        # 保存指标
        print("  - 保存训练指标...")
        with open("models/training_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("模型训练完成!")
        print("="*60)
        print(f"模型保存位置: models/multi_property/")
        print(f"指标保存位置: models/training_metrics.json")
        
        return multi_predictor, data_prep
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def demo_prediction(multi_predictor, data_prep):
    """演示预测功能"""
    print("\n" + "="*60)
    print("预测演示")
    print("="*60)
    
    # 示例成分
    test_compositions = [
        {
            'zinc': 97.0,
            'magnesium': 2.0,
            'aluminum': 0.5,
            'copper': 0.3,
            'silver': 0.0,
            'melting_temperature': 500,
            'annealing_temperature': 300,
            'aging_temperature': 120,
            'aging_time': 24
        },
        {
            'zinc': 95.0,
            'magnesium': 3.5,
            'aluminum': 1.0,
            'copper': 0.2,
            'silver': 0.1,
            'melting_temperature': 520,
            'annealing_temperature': 320,
            'aging_temperature': 150,
            'aging_time': 48
        }
    ]
    
    print("\n测试成分:")
    for i, comp in enumerate(test_compositions, 1):
        print(f"\n合金 {i}:")
        for elem, val in comp.items():
            if val > 0:
                print(f"  {elem}: {val}")
        
        # 预测
        predictions = multi_predictor.predict_single(comp)
        print(f"  预测性能:")
        for prop, val in predictions.items():
            print(f"    {prop}: {val:.2f}")


def demo_optimization(multi_predictor, data_prep):
    """演示优化功能"""
    print("\n" + "="*60)
    print("优化演示")
    print("="*60)
    
    # 创建优化器
    optimizer = MultiObjectiveOptimizer(multi_predictor)
    
    # 确保特征名称有效
    feature_names = data_prep.get_feature_importance_names()
    if not feature_names:
        # 使用默认特征名称
        feature_names = ['zinc', 'magnesium', 'aluminum', 'copper', 'silver']
    
    optimizer.set_feature_names(feature_names)
    print(f"设置特征名称: {feature_names}")
    
    # 单目标优化
    print("\n单目标优化: 最大化抗拉强度")
    result = optimizer.optimize_composition(
        objectives={'ultimate_tensile_strength': 'max'},
        method='differential_evolution',
        max_iter=50
    )
    
    # 多目标优化
    print("\n多目标优化: 最大化抗拉强度和延伸率")
    result = optimizer.optimize_composition(
        objectives={
            'ultimate_tensile_strength': 'max',
            'elongation': 'max'
        },
        method='differential_evolution',
        max_iter=50
    )
    
    # 帕累托优化
    print("\n帕累托前沿优化")
    pareto_solutions = optimizer.pareto_optimization(
        objectives=['ultimate_tensile_strength', 'elongation'],
        n_points=10
    )
    
    print(f"\n帕累托前沿上的 3 个示例解:")
    for i, sol in enumerate(pareto_solutions[:3], 1):
        print(f"\n解 {i}:")
        print(f"  抗拉强度: {sol['ultimate_tensile_strength']:.2f} MPa")
        print(f"  延伸率: {sol['elongation']:.2f}%")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("锌合金性能预测模型训练系统")
    print("="*60)
    
    # 创建模型目录
    os.makedirs("models", exist_ok=True)
    
    # 训练多性能预测模型
    multi_predictor, data_prep = train_multi_property_models()
    
    if multi_predictor and data_prep:
        # 演示预测
        demo_prediction(multi_predictor, data_prep)
        
        # 演示优化
        demo_optimization(multi_predictor, data_prep)
        
        print("\n" + "="*60)
        print("所有任务完成!")
        print("="*60)
        print("\n您现在可以:")
        print("1. 使用训练好的模型预测新合金的性能")
        print("2. 进行多目标优化找到最佳成分")
        print("3. 进行敏感性分析了解各元素的影响")
        print("\n模型文件保存在 models/ 目录下")
    else:
        print("\n训练失败，请检查数据库中是否有足够的数据")


if __name__ == "__main__":
    main()