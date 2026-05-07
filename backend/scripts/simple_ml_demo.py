"""简化版ML演示脚本 - 适用于小数据集"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json


def load_and_prepare_data():
    """加载并准备数据"""
    # 从CSV加载数据
    df = pd.read_csv("alloy_training_data.csv")
    print(f"加载了 {len(df)} 条数据")
    print(f"列名: {df.columns.tolist()}")
    
    # 查看数据
    print("\n数据预览:")
    print(df.head())
    
    return df


def simple_prediction_demo():
    """简单预测演示"""
    print("\n" + "="*60)
    print("机器学习预测演示")
    print("="*60)
    
    df = load_and_prepare_data()
    
    # 选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n数值列: {numeric_cols}")
    
    # 查找成分和性能列
    composition_cols = [col for col in numeric_cols if col in ['zinc', 'magnesium', 'aluminum', 'copper', 'silver']]
    property_cols = [col for col in numeric_cols if col in ['ultimate_tensile_strength', 'yield_strength', 'elongation', 'hardness']]
    
    print(f"\n成分列: {composition_cols}")
    print(f"性能列: {property_cols}")
    
    if not composition_cols or not property_cols:
        print("\n警告: 数据中没有足够的成分或性能数据")
        print("请确保数据库中有完整的合金数据")
        return
    
    # 对每个性能进行简单分析
    for target_prop in property_cols:
        print(f"\n{'='*60}")
        print(f"分析 {target_prop}")
        print(f"{'='*60}")
        
        # 准备数据
        data = df[composition_cols + [target_prop]].dropna()
        
        if len(data) < 3:
            print(f"  数据不足（{len(data)}条），跳过")
            continue
        
        X = data[composition_cols].values
        y = data[target_prop].values
        
        print(f"  有效数据: {len(data)}条")
        print(f"  {target_prop} 范围: {y.min():.2f} - {y.max():.2f}")
        print(f"  平均值: {y.mean():.2f}")
        
        # 如果数据足够，训练简单模型
        if len(data) >= 5:
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 训练模型
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_scaled, y)
            
            # 预测
            y_pred = model.predict(X_scaled)
            
            # 计算R²
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            print(f"  模型R²: {r2:.4f}")
            
            # 特征重要性
            importances = model.feature_importances_
            print(f"  特征重要性:")
            for col, imp in zip(composition_cols, importances):
                print(f"    {col}: {imp:.4f}")
            
            # 预测示例
            print(f"\n  预测示例:")
            test_compositions = [
                [97.0, 2.0, 0.5, 0.3, 0.0],  # 高Zn
                [95.0, 3.5, 1.0, 0.2, 0.1],  # 高Mg
                [98.0, 1.0, 0.5, 0.3, 0.0],  # 平衡
            ]
            
            for i, comp in enumerate(test_compositions, 1):
                comp_scaled = scaler.transform([comp])
                pred = model.predict(comp_scaled)[0]
                print(f"    合金{i}: Zn={comp[0]}%, Mg={comp[1]}%, Al={comp[2]}% -> {target_prop}={pred:.2f}")


def composition_recommendation():
    """成分推荐"""
    print("\n" + "="*60)
    print("成分推荐演示")
    print("="*60)
    
    df = pd.read_csv("alloy_training_data.csv")
    
    # 查找最佳性能的成分
    if 'ultimate_tensile_strength' in df.columns:
        best_idx = df['ultimate_tensile_strength'].idxmax()
        best_row = df.loc[best_idx]
        
        print("\n最高抗拉强度的合金:")
        print(f"  UTS: {best_row['ultimate_tensile_strength']:.2f} MPa")
        for col in ['zinc', 'magnesium', 'aluminum', 'copper']:
            if col in best_row and pd.notna(best_row[col]):
                print(f"  {col}: {best_row[col]:.2f}%")
    
    if 'elongation' in df.columns:
        best_idx = df['elongation'].idxmax()
        best_row = df.loc[best_idx]
        
        print("\n最高延伸率的合金:")
        print(f"  延伸率: {best_row['elongation']:.2f}%")
        for col in ['zinc', 'magnesium', 'aluminum', 'copper']:
            if col in best_row and pd.notna(best_row[col]):
                print(f"  {col}: {best_row[col]:.2f}%")
    
    # 简单相关性分析
    print("\n成分-性能相关性分析:")
    composition_cols = ['zinc', 'magnesium', 'aluminum', 'copper']
    property_cols = ['ultimate_tensile_strength', 'yield_strength', 'elongation', 'hardness']
    
    for prop in property_cols:
        if prop not in df.columns:
            continue
        print(f"\n  {prop}:")
        for comp in composition_cols:
            if comp not in df.columns:
                continue
            corr = df[comp].corr(df[prop])
            if pd.notna(corr):
                direction = "正相关" if corr > 0 else "负相关"
                strength = "强" if abs(corr) > 0.5 else "中等" if abs(corr) > 0.3 else "弱"
                print(f"    {comp}: {corr:.3f} ({strength}{direction})")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("锌合金机器学习演示系统")
    print("="*60)
    print("\n注意: 当前数据量较小，演示使用简化方法")
    print("建议收集更多数据以获得更好的预测效果")
    
    try:
        simple_prediction_demo()
        composition_recommendation()
        
        print("\n" + "="*60)
        print("演示完成!")
        print("="*60)
        print("\n建议:")
        print("1. 继续提取更多论文数据以扩充数据库")
        print("2. 当数据量达到50+条时，可以训练更准确的模型")
        print("3. 使用完整的ML系统进行多目标优化")
        
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
