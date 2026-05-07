# backend/scripts/clean_databases.py
import os
import sys
from pathlib import Path
import shutil

# 定位项目根目录
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
os.chdir(project_root)

print("=" * 60)
print("开始清理数据库")
print("=" * 60)

# 1. 清理 ChromaDB
chroma_path = "backend/data/chroma_db"
print(f"\n1. 清理 ChromaDB 目录: {chroma_path}")
if os.path.exists(chroma_path):
    try:
        shutil.rmtree(chroma_path)
        print("[成功] ChromaDB 清理成功")
    except Exception as e:
        print(f"[失败] ChromaDB 清理失败: {e}")
        print("[信息] 请确保没有其他进程正在使用 ChromaDB 目录")
else:
    print("[信息] ChromaDB 目录不存在，跳过")

# 2. 清理 PostgreSQL（可选）
print("\n2. 检查 PostgreSQL 状态")
print("   注意：如果不需要 PostgreSQL 清理，可以忽略此步骤")

try:
    import subprocess
    # 检查 psql 是否可用
    result = subprocess.run(
        ["psql", "--version"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("[成功] PostgreSQL 客户端可用")
        print("   建议执行: psql -U postgres -d deep_rag -c 'DROP TABLE IF EXISTS documents, document_chunks CASCADE;'")
    else:
        print("[信息] PostgreSQL 客户端不可用，跳过")
        print("   系统中可能未安装 PostgreSQL")
except Exception as e:
    print("[信息] PostgreSQL 客户端不可用，跳过")
    print(f"   错误: {e}")

print("\n" + "=" * 60)
print("数据库清理完成")
print("=" * 60)