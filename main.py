from fastapi import FastAPI, Query
from typing import List, Optional
from pydantic import BaseModel
import uvicorn
import random
from datetime import datetime, timedelta

app = FastAPI(title="VisionUber API")

# --- 数据模型定义 ---

class Task(BaseModel):
    id: str
    user_id: str
    title: str
    description: str
    lat: float
    lng: float
    budget: float
    status: str  # pending, matching, completed
    created_at: str
    valid_from: str
    valid_to: str

class Supply(BaseModel):
    id: str
    user_id: str
    provider_name: str
    lat: float
    lng: float
    rating: float
    description: str
    created_at: str
    valid_from: str
    valid_to: str

# --- Mock 数据生成器 ---

def get_mock_tasks():
    return [
        {
            "id": f"task_{i}",
            "user_id": f"user_{100+i}",
            "title": f"我想看涩谷十字路口 #{i}",
            "description": "希望能看到现在的人流情况，最好能拍到大屏幕。",
            "lat": 35.6595,
            "lng": 139.7005,
            "budget": 10.0 + i,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "valid_from": datetime.now().isoformat(timespec='minutes'),
            "valid_to": (datetime.now() + timedelta(hours=2)).isoformat(timespec='minutes')
        } for i in range(5)
    ]

def get_mock_supplies():
    return [
        {
            "id": f"supply_{i}",
            "user_id": f"provider_{200+i}",
            "provider_name": f"主播小张 #{i}",
            "lat": 35.6595,
            "lng": 139.7005,
            "rating": 4.8,
            "description": "人在东京，专业接单，画质清晰。",
            "created_at": datetime.now().isoformat(),
            "valid_from": datetime.now().isoformat(timespec='minutes'),
            "valid_to": (datetime.now() + timedelta(hours=4)).isoformat(timespec='minutes')
        } for i in range(5)
    ]

# --- API 路由 ---

@app.get("/")
async def root():
    return {"status": "ok", "message": "VisionUber API is running"}

@app.get("/feed", response_model=List[dict])
async def get_feed(is_consumer: bool = True):
    """
    根据角色返回信息流:
    - 消费者 (is_consumer=True): 看到的是周围的 [供给/服务]
    - 供给者 (is_consumer=False): 看到的是周围的 [任务/需求]
    """
    if is_consumer:
        # 消费者看“谁能帮我看”
        return get_mock_supplies()
    else:
        # 供给者看“谁想看什么”
        return get_mock_tasks()

@app.post("/tasks/create")
async def create_task(task: Task):
    """消费者发布新需求"""
    print(f"接收到新需求: {task.title}")
    return {"status": "success", "task_id": task.id}

@app.post("/supplies/create")
async def create_supply(supply: Supply):
    """供给者发布服务能力"""
    print(f"接收到新供给: {supply.provider_name}")
    return {"status": "success", "supply_id": supply.id}

@app.get("/search")
async def search(q: str, target: str = "task"):
    """
    搜索功能:
    - target='task': 搜任务库
    - target='supply': 搜供给库
    """
    return {"query": q, "target": target, "results": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)