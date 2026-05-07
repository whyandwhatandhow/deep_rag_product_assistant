"""机器学习快速开始 - 直接从数据库训练模型"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import json


def load_data_from_db():
    """从数据库加载数据"""
    conn = sqlite3.connect("alloy_database.db")
    
    # 查询所有相关数据
    query = """
    SELECT 
        a.alloy_id, a.alloy_system,
        c.zinc, c.magnesium, c.aluminum, c.copper, c.silver,
        p.melting_temperature, p.annealing_temperature, p.aging_temperature, p.aging_time,
        mp.ultimate_tensile_strength, mp.yield_strength, mp.elongation, mp.hardness
    FROM alloys a
    LEFT JOIN compositions c ON a.alloy_id = c.alloy_id
    LEFT JOIN processing p ON a.alloy_id = p.alloy_id
    LEFT JOIN mechanical_properties mp ON a.alloy_id = mp.alloy_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def train_and_predict():
    """训练模型并进行预测"""
    print("="*60)
    print("锌合金性能预测系统")
    print("="*60)
    
    # 加载数据
    df = load_data_from_db()
    print(f"\n从数据库加载了 {len(df)} 条数据")
    
    # 查看数据
    print("\n数据预览:")
    print(df[['alloy_id', 'alloy_system', 'zinc', 'magnesium', 'ultimate_tensile_strength']].head(10))
    
    # 选择特征和目标
    feature_cols = ['zinc', 'magnesium', 'aluminum', 'copper']
    target_cols = ['ultimate_tensile_strength', 'yield_strength', 'elongation', 'hardness']
    
    # 检查可用的列
    available_features = [col for col in feature_cols if col in df.columns]
    available_targets = [col for col in target_cols if col in df.columns]
    
    print(f"\n可用特征: {available_features}")
    print(f"可用目标: {available_targets}")
    
    if not available_features or not available_targets:
        print("\n错误: 没有足够的特征或目标数据")
        return
    
    # 对每个目标训练模型
    models = {}
    scalers = {}
    
    for target in available_targets:
        print(f"\n{'='*60}")
        print(f"训练 {target} 预测模型")
        print(f"{'='*60}")
        
        # 准备数据
        data = df[available_features + [target]].dropna()
        
        if len(data) < 3:
            print(f"  数据不足 ({len(data)}条)，跳过")
            continue
        
        X = data[available_features].values
        y = data[target].values
        
        print(f"  有效数据: {len(data)}条")
        print(f"  {target} 范围: {y.min():.2f} - {y.max():.2f}")
        print(f"  平均值: {y.mean():.2f}")
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scalers[target] = scaler
        
        # 划分训练集和测试集
        if len(data) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
        else:
            X_train, y_train = X_scaled, y
            X_test, y_test = X_scaled, y
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[target] = model
        
        # 评估
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"  模型性能:")
        print(f"    R²: {r2:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        
        # 特征重要性
        importances = model.feature_importances_
        print(f"  特征重要性:")
        for feat, imp in zip(available_features, importances):
            print(f"    {feat}: {imp:.4f}")
    
    # 预测新合金
    if models:
        print(f"\n{'='*60}")
        print("预测新合金性能")
        print(f"{'='*60}")
        
        new_alloys = [
            {"name": "高锌合金", "zinc": 98.0, "magnesium": 1.5, "aluminum": 0.3, "copper": 0.1},
            {"name": "高镁合金", "zinc": 95.0, "magnesium": 4.0, "aluminum": 0.5, "copper": 0.2},
            {"name": "平衡合金", "zinc": 96.5, "magnesium": 2.5, "aluminum": 0.5, "copper": 0.3},
        ]
        
        for alloy in new_alloys:
            print(f"\n{alloy['name']}:")
            print(f"  成分: Zn={alloy['zinc']}%, Mg={alloy['magnesium']}%, Al={alloy['aluminum']}%, Cu={alloy['copper']}%")
            print(f"  预测性能:")
            
            for target, model in models.items():
                scaler = scalers[target]
                X_new = np.array([[alloy[f] for f in available_features]])
                X_new_scaled = scaler.transform(X_new)
                pred = model.predict(X_new_scaled)[0]
                print(f"    {target}: {pred:.2f}")
    
    return models, scalers


def main():
    """主函数"""
    try:
        models, scalers = train_and_predict()
        
        if models:
            print("\n" + "="*60)
            print("训练完成!")
            print("="*60)
            print("\n您现在可以:")
            print("1. 使用训练好的模型预测新合金的性能")
            print("2. 调整成分来优化性能")
            print("3. 继续提取更多论文数据以提高模型准确性")
        else:
            print("\n未能训练模型，请检查数据库中的数据")
            
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
