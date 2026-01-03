from fastapi import FastAPI, Query, Depends, HTTPException
from typing import List, Optional, Union
from pydantic import ValidationError
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import BigInteger
import uvicorn

app = FastAPI(title="VisionUber API")

# --- 数据库配置 ---

DATABASE_URL = "postgresql://admin:admin@localhost:5432/vision_uber"
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    #仅用于修改数据库，清库时使用，正常使用需要注释掉
    SQLModel.metadata.drop_all(engine)
    
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# --- 数据模型定义 ---

class Task(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger, description="任务唯一ID")
    user_id: str = Field(..., description="发布者ID")
    title: str = Field(..., description="任务标题")
    description: str = Field(None, description="详细描述")
    lat: float = Field(..., description="纬度")
    lng: float = Field(..., description="经度")
    budget: float = Field(..., description="预算")
    status: str = Field("pending", description="状态: pending, matching, completed")
    created_at: str = Field(...)
    valid_from: str = Field(...)
    valid_to: str = Field(...)

class Supply(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger)
    user_id: str
    title: str
    description: str
    lat: float
    lng: float
    rating: float
    created_at: str
    valid_from: str
    valid_to: str

# --- API 路由 ---

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
async def root():
    return {"status": "ok", "message": "VisionUber API is running"}

@app.get("/feed", response_model=List[Union[Supply, Task]])
async def get_feed(
    is_consumer: bool = Query(True, description="角色标识: True 为消费者(看供给), False 为供给者(看需求)"),
    session: Session = Depends(get_session)
):
    """
    根据角色返回信息流:
    - 消费者 (is_consumer=True): 看到的是周围的 [供给/服务]
    - 供给者 (is_consumer=False): 看到的是周围的 [任务/需求]
    """
    if is_consumer:
        # 消费者看“谁能帮我看”
        return session.exec(select(Supply)).all()
    else:
        # 供给者看“谁想看什么”
        return session.exec(select(Task)).all()

@app.post("/create")
async def create_entry(
    item: dict,
    is_consumer: bool = Query(True, description="角色标识: True 为消费者(发布需求), False 为供给者(发布服务)"),
    session: Session = Depends(get_session)
):
    """
    创建发布:
    - 消费者 (is_consumer=True): 发布需求
    - 供给者 (is_consumer=False): 发布服务
    """
    if is_consumer:
        try:
            task_item = Task.model_validate(item)
            session.add(task_item)
            session.commit()
            session.refresh(task_item)
            return {"status": "success", "task_id": task_item.id}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"数据格式错误 (期望 Task): {e}")
    else:
        try:
            supply_item = Supply.model_validate(item)
            session.add(supply_item)
            session.commit()
            session.refresh(supply_item)
            return {"status": "success", "supply_id": supply_item.id}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"数据格式错误 (期望 Supply): {e}")

@app.get("/search")
async def search(
    q: str,
    is_consumer: bool = Query(True, description="角色标识: True 为消费者(搜供给), False 为供给者(搜需求)"),
    session: Session = Depends(get_session)
):
    """
    搜索功能:
    - 消费者 (is_consumer=True): 搜索供给库
    - 供给者 (is_consumer=False): 搜索任务库
    """
    if is_consumer:
        results = session.exec(select(Supply).where(Supply.title.contains(q))).all()
        return {"query": q, "target": "supply", "results": results}
    else:
        results = session.exec(select(Task).where(Task.title.contains(q))).all()
        return {"query": q, "target": "task", "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)