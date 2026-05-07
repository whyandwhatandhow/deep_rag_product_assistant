"""ML数据准备模块 - 从数据库提取和预处理训练数据"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json


class MLDataPreparation:
    """机器学习数据准备类"""
    
    def __init__(self, db_path: str = "alloy_database.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data_from_db(self) -> pd.DataFrame:
        """从数据库加载数据"""
        conn = sqlite3.connect(self.db_path)
        
        # 构建完整的查询，关联所有相关表
        query = """
        SELECT 
            a.alloy_id,
            a.alloy_system,
            a.year,
            c.zinc, c.magnesium, c.aluminum, c.copper, c.silver,
            c.other_elements,
            p.melting_temperature, p.casting_method, p.annealing_temperature,
            p.aging_temperature, p.aging_time,
            m.phases, m.grain_size,
            mp.ultimate_tensile_strength, mp.yield_strength, mp.elongation, mp.hardness,
            mp.elastic_modulus, mp.fatigue_strength,
            cp.corrosion_rate
        FROM alloys a
        LEFT JOIN compositions c ON a.alloy_id = c.alloy_id
        LEFT JOIN processing p ON a.alloy_id = p.alloy_id
        LEFT JOIN microstructure m ON a.alloy_id = m.alloy_id
        LEFT JOIN mechanical_properties mp ON a.alloy_id = mp.alloy_id
        LEFT JOIN corrosion_properties cp ON a.alloy_id = cp.alloy_id
        -- WHERE c.zinc IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"从数据库加载了 {len(df)} 条数据")
        return df
    
    def parse_other_elements(self, other_elements_json: str) -> Dict[str, float]:
        """解析其他元素JSON"""
        if not other_elements_json or other_elements_json == 'null':
            return {}
        try:
            elements = json.loads(other_elements_json)
            result = {}
            for elem, value in elements.items():
                # 尝试转换为数值
                if isinstance(value, (int, float)):
                    result[elem] = float(value)
                elif isinstance(value, str):
                    # 提取字符串中的数值
                    import re
                    numbers = re.findall(r'[\d.]+', value)
                    if numbers:
                        result[elem] = float(numbers[0])
            return result
        except:
            return {}
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取特征"""
        # 基础成分特征
        feature_cols = ['zinc', 'magnesium', 'aluminum', 'copper', 'silver']
        
        # 处理其他元素
        other_elements_list = []
        for _, row in df.iterrows():
            other = self.parse_other_elements(row.get('other_elements', ''))
            other_elements_list.append(other)
        
        # 找出所有可能的元素
        all_elements = set()
        for other in other_elements_list:
            all_elements.update(other.keys())
        
        # 为每个元素创建列
        for elem in all_elements:
            df[f'elem_{elem}'] = [other.get(elem, 0.0) for other in other_elements_list]
            feature_cols.append(f'elem_{elem}')
        
        # 工艺特征
        if 'melting_temperature' in df.columns:
            df['melting_temperature'] = pd.to_numeric(df['melting_temperature'], errors='coerce')
            feature_cols.append('melting_temperature')
        
        if 'annealing_temperature' in df.columns:
            df['annealing_temperature'] = pd.to_numeric(df['annealing_temperature'], errors='coerce')
            feature_cols.append('annealing_temperature')
        
        if 'aging_temperature' in df.columns:
            df['aging_temperature'] = pd.to_numeric(df['aging_temperature'], errors='coerce')
            feature_cols.append('aging_temperature')
        
        if 'aging_time' in df.columns:
            df['aging_time'] = pd.to_numeric(df['aging_time'], errors='coerce')
            feature_cols.append('aging_time')
        
        # 编码铸造方法
        if 'casting_method' in df.columns:
            le = LabelEncoder()
            df['casting_method_encoded'] = le.fit_transform(df['casting_method'].fillna('Unknown'))
            self.label_encoders['casting_method'] = le
            feature_cols.append('casting_method_encoded')
        
        # 微观结构特征
        if 'grain_size' in df.columns:
            df['grain_size'] = pd.to_numeric(df['grain_size'], errors='coerce')
            feature_cols.append('grain_size')
        
        # 填充缺失值
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df, feature_cols
    
    def prepare_training_data(
        self, 
        target_property: str = 'ultimate_tensile_strength',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        Args:
            target_property: 目标性能属性
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # 加载数据
        df = self.load_data_from_db()
        
        if len(df) == 0:
            raise ValueError("数据库中没有数据，请先导入合金数据")
        
        # 提取特征
        df, feature_cols = self.extract_features(df)
        
        # 检查目标属性
        if target_property not in df.columns:
            raise ValueError(f"目标属性 {target_property} 不在数据中")
        
        # 移除目标属性为空的行
        df = df[df[target_property].notna()]
        
        if len(df) < 3:
            raise ValueError(f"目标属性 {target_property} 的有效数据太少（{len(df)}条），无法训练模型")
        elif len(df) < 5:
            print(f"警告: 目标属性 {target_property} 的数据较少（{len(df)}条），模型可能不够准确")
        
        # 准备特征和标签
        X = df[feature_cols].values
        y = df[target_property].values
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n训练数据准备完成:")
        print(f"  总样本数: {len(df)}")
        print(f"  训练集: {len(X_train)}")
        print(f"  测试集: {len(X_test)}")
        print(f"  特征数: {len(feature_cols)}")
        print(f"  目标属性: {target_property}")
        print(f"  目标范围: {y.min():.2f} - {y.max():.2f}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def prepare_multi_target_data(
        self,
        target_properties: List[str] = None,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        准备多目标训练数据
        
        Args:
            target_properties: 目标性能属性列表
            test_size: 测试集比例
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names, target_names
        """
        if target_properties is None:
            target_properties = [
                'yield_strength', 
                'elongation',
                'hardness'
            ]
        
        # 加载数据
        df = self.load_data_from_db()
        
        if len(df) == 0:
            raise ValueError("数据库中没有数据")
        
        # 提取特征
        df, feature_cols = self.extract_features(df)
        
        # 检查哪些目标属性有数据
        available_targets = []
        for target in target_properties:
            if target in df.columns and df[target].notna().sum() >= 3:
                available_targets.append(target)
        
        if not available_targets:
            raise ValueError("没有足够的数据来训练任何目标属性")
        
        # 为每个目标属性单独处理，确保没有 NaN 值
        valid_indices = []
        for i, row in df.iterrows():
            # 检查是否至少有一个目标属性有值
            has_value = False
            for target in available_targets:
                if not pd.isna(row[target]):
                    has_value = True
                    break
            if has_value:
                valid_indices.append(i)
        
        df = df.loc[valid_indices]
        
        if len(df) < 3:
            raise ValueError(f"有效数据太少（{len(df)}条）")
        elif len(df) < 5:
            print(f"警告: 数据较少（{len(df)}条），模型可能不够准确")
        
        # 准备特征和标签
        X = df[feature_cols].values
        
        # 处理 y 中的 NaN 值，用均值填充
        y = df[available_targets].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        y = imputer.fit_transform(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        print(f"\n多目标训练数据准备完成:")
        print(f"  总样本数: {len(df)}")
        print(f"  训练集: {len(X_train)}")
        print(f"  测试集: {len(X_test)}")
        print(f"  特征数: {len(feature_cols)}")
        print(f"  目标属性: {available_targets}")
        
        return X_train, X_test, y_train, y_test, feature_cols, available_targets
    
    def transform_new_data(self, new_data: pd.DataFrame) -> np.ndarray:
        """转换新数据用于预测"""
        df, feature_cols = self.extract_features(new_data)
        X = df[feature_cols].values
        return self.scaler.transform(X)
    
    def get_feature_importance_names(self) -> List[str]:
        """获取特征名称列表"""
        return list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else []