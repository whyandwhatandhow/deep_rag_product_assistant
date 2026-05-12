"""数据库数据清洗脚本"""
import sqlite3
import pandas as pd

def check_duplicates():
    """检查数据库中的重复数据"""
    conn = sqlite3.connect('alloy_database.db')
    cursor = conn.cursor()
    
    print("=== 数据库重复数据检查 ===")
    
    # 检查各个表的记录数
    tables = ['alloys', 'compositions', 'processing', 'microstructure', 'mechanical_properties', 'corrosion_properties']
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} 条记录")
    
    # 检查重复的 alloy_id
    print("\n=== 重复数据统计 ===")
    
    for table in tables:
        cursor.execute(f"SELECT alloy_id, COUNT(*) as cnt FROM {table} GROUP BY alloy_id HAVING cnt > 1")
        duplicates = cursor.fetchall()
        if duplicates:
            print(f"{table} 表中有 {len(duplicates)} 个重复的 alloy_id")
        else:
            print(f"{table} 表中没有重复数据")
    
    # 检查空值较多的行
    print("\n=== 数据完整性检查 ===")
    df = pd.read_sql("SELECT * FROM alloys LIMIT 5", conn)
    print("alloys 表示例数据:")
    print(df)
    
    conn.close()

def remove_duplicates():
    """移除数据库中的重复数据"""
    conn = sqlite3.connect('alloy_database.db')
    cursor = conn.cursor()
    
    print("\n=== 开始清理重复数据 ===")
    
    # 创建备份
    cursor.execute("VACUUM INTO 'alloy_database_backup.db'")
    print("✓ 已创建数据库备份: alloy_database_backup.db")
    
    tables = ['alloys', 'compositions', 'processing', 'microstructure', 'mechanical_properties', 'corrosion_properties']
    
    for table in tables:
        cursor.execute(f"""
            DELETE FROM {table} 
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM {table} GROUP BY alloy_id
            )
        """)
        deleted = cursor.rowcount
        if deleted > 0:
            print(f"✓ 从 {table} 表删除了 {deleted} 条重复数据")
        else:
            print(f"✓ {table} 表没有重复数据")
    
    conn.commit()
    
    # 清理后统计
    print("\n=== 清理后统计 ===")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} 条记录")
    
    conn.close()
    print("\n✓ 数据清理完成!")

if __name__ == "__main__":
    check_duplicates()
    remove_duplicates()