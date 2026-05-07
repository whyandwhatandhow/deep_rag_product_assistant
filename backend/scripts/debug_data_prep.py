"""调试数据准备过程"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from app.ml.data_preparation import MLDataPreparation


def debug_data_prep():
    """调试数据准备过程"""
    print("="*60)
    print("调试数据准备过程")
    print("="*60)
    
    data_prep = MLDataPreparation("alloy_database.db")
    
    # 加载原始数据
    df = data_prep.load_data_from_db()
    print(f"\n原始数据: {len(df)} 条")
    print(f"列名: {df.columns.tolist()}")
    
    # 提取特征
    df, feature_cols = data_prep.extract_features(df)
    print(f"\n提取特征后: {len(df)} 条")
    print(f"特征列: {feature_cols}")
    
    # 检查目标属性
    target_properties = ['ultimate_tensile_strength', 'yield_strength', 'elongation', 'hardness']
    
    print("\n各目标属性的非空值数量:")
    for target in target_properties:
        if target in df.columns:
            non_null = df[target].notna().sum()
            print(f"  {target}: {non_null}")
        else:
            print(f"  {target}: 列不存在")
    
    # 检查可用的目标属性
    available_targets = []
    for target in target_properties:
        if target in df.columns and df[target].notna().sum() >= 3:
            available_targets.append(target)
    
    print(f"\n可用的目标属性: {available_targets}")
    
    if available_targets:
        # 移除任何目标属性为空的行
        df_filtered = df.dropna(subset=available_targets)
        print(f"\n过滤后的数据: {len(df_filtered)} 条")
        
        if len(df_filtered) > 0:
            print("\n过滤后的数据预览:")
            print(df_filtered[available_targets].head())
        else:
            print("\n过滤后没有数据")
    else:
        print("\n没有可用的目标属性")


def main():
    """主函数"""
    try:
        debug_data_prep()
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
