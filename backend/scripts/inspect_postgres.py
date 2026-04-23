# backend/scripts/inspect_postgres.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from backend.app.core.config import settings  # ← 使用统一配置

load_dotenv()

print("🔍 开始连接 PostgreSQL（使用 config.py 配置）...")

DATABASE_URL = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

try:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    with SessionLocal() as session:
        # 查看表
        tables = session.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';"))
        print("✅ 连接成功！public schema 中的表：")
        for t in tables.fetchall():
            print(f"   • {t[0]}")

        # 查询 documents 表
        result = session.execute(text("""
            SELECT document_id, product_name, doc_type, file_name, total_chunks, created_at 
            FROM documents 
            ORDER BY created_at DESC 
            LIMIT 5;
        """))

        rows = result.fetchall()
        print(f"\n📄 documents 表记录（共 {len(rows)} 条）：")
        for row in rows:
            print(f"document_id : {row.document_id}")
            print(f"  产品     : {row.product_name}")
            print(f"  类型     : {row.doc_type}")
            print(f"  文件     : {row.file_name}")
            print(f"  chunks   : {row.total_chunks}")
            print(f"  时间     : {row.created_at}")
            print("-" * 80)

except Exception as e:
    print(f"❌ PostgreSQL 连接失败: {e}")
    print("\n💡 修复建议：")
    print(
        f"   当前配置 → 用户: {settings.postgres_user} | 密码: {settings.postgres_password} | DB: {settings.postgres_db}")
    print("   1. 修改 backend/app/core/config.py 中的 postgres_password（推荐改成你的真实密码）")
    print("   2. 或在 .env 文件中覆盖：POSTGRES_PASSWORD=你的真实密码")
    print("   3. 确认 PostgreSQL 服务正在运行（端口 5432）")