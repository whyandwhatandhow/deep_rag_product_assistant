# backend/tests/test_loader.py
from pathlib import Path
from app.ingest.loader import DocumentLoader
from app.ingest.preprocessor import DocumentPreprocessor
from app.ingest.chunker import DocumentChunker

def main():
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    test_file = project_root / "data" / "raw" / "product_manual.pdf"

    print(f"📍 项目根目录: {project_root}")
    print(f"📄 正在加载测试文件: {test_file}\n")

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
        product_name="医学影像产品",
        doc_type="paper"
    )
    print(f"✅ Preprocessor 预处理完成: {len(processed_docs)} 个干净 chunk")

    # === Step 3: 分块 ===
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=100)
    final_chunks = chunker.chunk(processed_docs)
    print(f"✅ Chunker 分块完成: {len(final_chunks)} 个最终 chunk")

    # 展示结果
    if final_chunks:
        print("\n" + "="*80)
        print("第一个最终 chunk 预览（前 400 字符）:")
        print(final_chunks[0].page_content[:400])
        print("="*80)
        print("完整 metadata（关键字段）:")
        for k, v in list(final_chunks[0].metadata.items())[:8]:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()