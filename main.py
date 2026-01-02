from fastapi import FastAPI, Query
from typing import List, Optional, Union
from pydantic import BaseModel, Field
import uvicorn
import random
from datetime import datetime, timedelta

app = FastAPI(title="VisionUber API")

# --- 数据模型定义 ---

class Task(BaseModel):
    id: str = Field(..., description="任务唯一ID", example="task_1001")
    user_id: str = Field(..., description="发布者ID", example="user_888")
    title: str = Field(..., description="任务标题", example="想看涩谷十字路口")
    description: str = Field(None, description="详细描述", example="希望能看到现在的人流情况")
    lat: float = Field(..., description="纬度", example=35.6595)
    lng: float = Field(..., description="经度", example=139.7005)
    budget: float = Field(..., description="预算", example=50.0)
    status: str = Field("pending", description="状态: pending, matching, completed")
    created_at: str = Field(..., example="2024-01-01T12:00:00")
    valid_from: str = Field(..., example="2024-01-01T12:00:00")
    valid_to: str = Field(..., example="2024-01-01T14:00:00")

class Supply(BaseModel):
    id: str
    user_id: str
    title: str
    description: str
    lat: float
    lng: float
    rating: float
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
            "title": f"提供东京涩谷直播服务 #{i}",
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
async def get_feed(is_consumer: bool = Query(True, description="角色标识: True 为消费者(看供给), False 为供给者(看需求)")):
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

@app.post("/create")
async def create_entry(
    item: Union[Task, Supply],
    is_consumer: bool = Query(True, description="角色标识: True 为消费者(发布需求), False 为供给者(发布服务)")
):
    """
    创建发布:
    - 消费者 (is_consumer=True): 发布需求
    - 供给者 (is_consumer=False): 发布服务
    """
    if is_consumer:
        print(f"接收到新需求: {item.title}")
        return {"status": "success", "task_id": item.id}
    else:
        print(f"接收到新供给: {item.title}")
        return {"status": "success", "supply_id": item.id}

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