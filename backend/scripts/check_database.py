"""检查数据库中性能数据的情况"""

import sqlite3
import pandas as pd


def check_mechanical_properties():
    """检查力学性能数据"""
    conn = sqlite3.connect("alloy_database.db")
    
    # 查询力学性能表
    query = """
    SELECT 
        a.alloy_id, a.alloy_system,
        mp.ultimate_tensile_strength, mp.yield_strength, 
        mp.elongation, mp.hardness
    FROM alloys a
    LEFT JOIN mechanical_properties mp ON a.alloy_id = mp.alloy_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("力学性能数据检查:")
    print(f"总记录数: {len(df)}")
    
    # 统计每个性能的非空值数量
    for col in ['ultimate_tensile_strength', 'yield_strength', 'elongation', 'hardness']:
        non_null_count = df[col].notna().sum()
        print(f"{col} 非空值: {non_null_count}")
    
    # 查看前10条数据
    print("\n前10条数据:")
    print(df.head(10))
    
    # 查看有性能数据的记录
    has_data = df[(df['ultimate_tensile_strength'].notna()) | 
                 (df['yield_strength'].notna()) | 
                 (df['elongation'].notna()) | 
                 (df['hardness'].notna())]
    
    print(f"\n有性能数据的记录: {len(has_data)}")
    if len(has_data) > 0:
        print(has_data)


def check_composition():
    """检查成分数据"""
    conn = sqlite3.connect("alloy_database.db")
    
    query = """
    SELECT 
        a.alloy_id, a.alloy_system,
        c.zinc, c.magnesium, c.aluminum, c.copper
    FROM alloys a
    LEFT JOIN compositions c ON a.alloy_id = c.alloy_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\n成分数据检查:")
    print(f"总记录数: {len(df)}")
    
    # 统计每个成分的非空值数量
    for col in ['zinc', 'magnesium', 'aluminum', 'copper']:
        non_null_count = df[col].notna().sum()
        print(f"{col} 非空值: {non_null_count}")
    
    # 查看前10条数据
    print("\n前10条数据:")
    print(df.head(10))


def main():
    """主函数"""
    print("="*60)
    print("数据库数据检查")
    print("="*60)
    
    check_mechanical_properties()
    check_composition()


if __name__ == "__main__":
    main()
