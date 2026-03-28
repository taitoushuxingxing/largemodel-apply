"""
FastAPI 推理服务 - 新闻分类
基于 Qwen2.5-0.5B-Instruct 全参数微调模型，通过 model.generate() 生成分类标签。
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "outputs/news_cls_v1/best_model")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "你是一个新闻分类助手。根据给定的新闻标题，判断它属于哪个类别。"
    "只返回类别标签，不要返回其他内容。"
    "可选类别：sports(体育)、technology(科技)、entertainment(娱乐)、finance(财经)、military(军事)"
)

VALID_LABELS = ["sports", "technology", "entertainment", "finance", "military"]

# 中文名称到英文标签的映射，用于模糊匹配
CHINESE_TO_LABEL = {
    "体育": "sports",
    "科技": "technology",
    "娱乐": "entertainment",
    "财经": "finance",
    "军事": "military",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("news_cls_api")

# ---------------------------------------------------------------------------
# 全局模型 & Tokenizer（在 lifespan 中初始化）
# ---------------------------------------------------------------------------
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时释放资源。"""
    global model, tokenizer

    logger.info("正在加载模型: %s，设备: %s", MODEL_DIR, DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    model.eval()
    logger.info("模型加载完成")

    yield  # -------- 应用运行 --------

    model = None
    tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("模型已卸载")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="新闻分类推理服务",
    description="基于 Qwen2.5-0.5B-Instruct 微调模型的新闻文本五分类 API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 请求/响应 Schema
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1, description="待分类的新闻文本")


class ClassifyResponse(BaseModel):
    query: str
    prediction: str = Field(..., description="模型原始输出")
    label: str = Field(..., description="校验后的有效标签（无法匹配时保留原始输出）")


# ---------------------------------------------------------------------------
# 标签校验与模糊匹配
# ---------------------------------------------------------------------------
def validate_label(raw: str) -> str:
    """
    将模型原始输出与有效标签列表匹配。
    1. 精确匹配有效标签（忽略大小写）。
    2. 检查输出是否包含某英文标签。
    3. 检查输出是否包含中文名称，映射到对应英文标签。
    4. 无法匹配时返回原始输出。
    """
    text = raw.strip()

    # 1. 精确匹配（忽略大小写）
    for label in VALID_LABELS:
        if text.lower() == label:
            return label

    # 2. 包含英文标签
    text_lower = text.lower()
    for label in VALID_LABELS:
        if label in text_lower:
            return label

    # 3. 包含中文名称
    for chinese, label in CHINESE_TO_LABEL.items():
        if chinese in text:
            return label

    return text


# ---------------------------------------------------------------------------
# 推理核心函数
# ---------------------------------------------------------------------------
def _run_inference(query: str) -> str:
    """对单条新闻文本执行推理，返回模型原始输出字符串。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"请判断以下新闻的类别：\n{query}"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------
@app.post("/classify", response_model=ClassifyResponse, summary="单条新闻分类")
async def classify(req: ClassifyRequest):
    """接收一条新闻文本，返回分类预测结果。"""
    try:
        prediction = _run_inference(req.query)
    except Exception as e:
        logger.exception("推理失败")
        raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")

    return ClassifyResponse(
        query=req.query,
        prediction=prediction,
        label=validate_label(prediction),
    )


@app.post(
    "/batch_classify",
    response_model=List[ClassifyResponse],
    summary="批量新闻分类",
)
async def batch_classify(requests: List[ClassifyRequest]):
    """接收多条新闻文本（最多 64 条），返回批量分类结果。"""
    if len(requests) > 64:
        raise HTTPException(status_code=400, detail="单次最多支持 64 条文本")

    results = []
    for req in requests:
        try:
            prediction = _run_inference(req.query)
        except Exception as e:
            logger.exception("推理失败: %s", req.query[:50])
            raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")
        results.append(
            ClassifyResponse(
                query=req.query,
                prediction=prediction,
                label=validate_label(prediction),
            )
        )
    return results


@app.get("/health", summary="健康检查")
async def health():
    """返回服务状态、模型路径、设备和支持的标签列表。"""
    return {
        "status": "healthy",
        "model": MODEL_DIR,
        "device": DEVICE,
        "labels": VALID_LABELS,
    }


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.serve:app", host=HOST, port=PORT, reload=False, workers=1)
