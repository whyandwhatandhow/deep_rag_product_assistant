from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.ingest.ingest_router import router as ingest_router

load_dotenv()

app = FastAPI(
    title="深 RAG 产品知识助手",
    description="严格基于证据的产品知识问答系统 - 第一版 MVP",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(ingest_router)

@app.get("/")
async def root():
    return {
        "message": "深 RAG 产品知识助手已启动！（ingest 模块已接入）",
        "status": "ready",
        "docs": "/docs",
        "ingest_api": "/ingest/document"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)