# backend/tests/test_loader.py
from pathlib import Path
from app.ingest.loader import DocumentLoader
from app.ingest.preprocessor import DocumentPreprocessor


def main():
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    test_file = project_root / "data" / "raw" / "product_manual.pdf"

    print(f"📍 项目根目录: {project_root}")
    print(f"📄 正在加载测试文件: {test_file}")

    if not test_file.exists():
        print(f"❌ 文件不存在: {test_file}")
        return

    # === Step 1: 加载 ===
    loader = DocumentLoader()
    raw_docs = loader.load_file(str(test_file))
    print(f"✅ Loader 加载成功: {len(raw_docs)} 个原始 chunks")

    # === Step 2: 预处理 ===
    preprocessor = DocumentPreprocessor()
    processed_docs = preprocessor.preprocess(
        raw_docs,
        product_name="医学影像产品",  # ← 这里可以改成你真实的产品名
        doc_type="paper"  # manual / faq / policy / paper 等
    )

    print(f"✅ Preprocessor 预处理完成: {len(processed_docs)} 个干净 chunk")

    if processed_docs:
        print("\n第一个预处理后的 chunk 预览:")
        print("=" * 80)
        print(processed_docs[0].page_content[:400])
        print("=" * 80)
        print("丰富后的 metadata:")
        print(processed_docs[0].metadata)


if __name__ == "__main__":
    main()