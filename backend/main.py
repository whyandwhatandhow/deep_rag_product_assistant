from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.ingest.ingest_router import router as ingest_router
from app.api.routes import router as api_router
from app.api.eval_routes import router as eval_router

load_dotenv()

app = FastAPI(
    title="深 RAG 产品知识助手",
    description="严格基于证据的产品知识问答系统 - 第四阶段增强版",
    version="0.4.0"
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
app.include_router(api_router)
app.include_router(eval_router)

@app.get("/")
async def root():
    return {
        "message": "深 RAG 产品知识助手已启动！（query API 已接入）",
        "status": "ready",
        "docs": "/docs",
        "query_api": "/api/v1/query",
        "health_api": "/api/v1/health",
        "ingest_api": "/ingest/document"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)