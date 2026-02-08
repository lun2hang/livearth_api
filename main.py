from fastapi import FastAPI, Query, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from typing import List, Optional, Union
from pydantic import ValidationError
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from sqlalchemy import BigInteger, func, Index
from sqlalchemy.orm import aliased
import uvicorn
import uuid
import urllib.request
import json
from datetime import datetime, timedelta
import auth_utils
from jose import jwt, JWTError
import math

app = FastAPI(title="VisionUber API")

# --- 数据库配置 ---

DATABASE_URL = "postgresql://admin:admin@localhost:5432/vision_uber"
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    #仅用于修改数据库，清库时使用，正常使用需要注释掉
    #SQLModel.metadata.drop_all(engine)
    
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
    status: str = Field("created", description="状态: created, matched, live_start, live_end, paied, canceled, timeout")
    created_at: datetime = Field(...)
    valid_from: datetime = Field(...)
    valid_to: datetime = Field(...)

class Supply(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger)
    user_id: str
    title: str
    description: str
    lat: float
    lng: float
    rating: float
    price: float = Field(default=0.0, description="价格")
    status: str = Field(default="created", description="状态: created, matched, live_start, live_end, paied, canceled, timeout")
    created_at: datetime
    valid_from: datetime
    valid_to: datetime

class TaskRead(Task):
    nickname: Optional[str] = None
    avatar: Optional[str] = None

class SupplyRead(Supply):
    nickname: Optional[str] = None
    avatar: Optional[str] = None

class SocialAccount(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    provider: str = Field(index=True) # "google", "apple"
    provider_user_id: str = Field(index=True) # Google 里的 sub ID
    user_id: str = Field(foreign_key="user.id")
    user: Optional["User"] = Relationship(back_populates="social_accounts")

class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, description="用户唯一ID")
    username: str = Field(index=True, unique=True, description="用户名")
    email: str = Field(index=True, unique=True, description="邮箱")
    nickname: Optional[str] = Field(default=None, description="用户昵称")
    hashed_password: Optional[str] = Field(None, description="加密后的密码")
    avatar: Optional[str] = Field(None, description="头像URL")
    status: str = Field(default="active", description="账号状态: active, suspended")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # 关联多个三方账号
    social_accounts: List[SocialAccount] = Relationship(back_populates="user")

class Order(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger, description="订单ID")
    consumer_id: str = Field(..., description="消费者ID", index=True)
    provider_id: str = Field(..., description="服务者ID", index=True)
    task_id: Optional[int] = Field(default=None, description="关联的需求ID", sa_type=BigInteger)
    supply_id: Optional[int] = Field(default=None, description="关联的供给ID", sa_type=BigInteger)
    amount: float = Field(..., description="交易金额")
    status: str = Field(default="matched", description="状态: matched, live_start, live_end, paied, canceled, timeout")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TaskLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger)
    task_id: int = Field(..., sa_type=BigInteger, index=True)
    operator_id: str = Field(..., description="操作人ID")
    action: str = Field(..., description="操作类型: create, cancel, match, etc.")
    previous_status: Optional[str] = Field(None, description="修改前状态")
    new_status: str = Field(..., description="修改后状态")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SupplyLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger)
    supply_id: int = Field(..., sa_type=BigInteger, index=True)
    operator_id: str = Field(..., description="操作人ID")
    action: str = Field(..., description="操作类型")
    previous_status: Optional[str] = Field(None)
    new_status: str = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class OrderLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger)
    order_id: int = Field(..., sa_type=BigInteger, index=True)
    operator_id: str = Field(..., description="操作人ID")
    action: str = Field(..., description="操作类型")
    previous_status: Optional[str] = Field(None)
    new_status: str = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(SQLModel, table=True):
    # 添加复合索引以加速未读数计算
    # 场景: WHERE receiver_id = ? GROUP BY order_id HAVING id > ?
    # 索引覆盖: 直接在索引中完成计数，无需回表查询
    __table_args__ = (
        Index("idx_receiver_unread", "receiver_id", "order_id", "id"),
    )
    
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger)
    order_id: int = Field(..., index=True, sa_type=BigInteger, description="关联订单ID")
    sender_id: str = Field(..., index=True, description="发送者ID")
    receiver_id: str = Field(..., index=True, description="接收者ID")
    content: str = Field(..., description="消息内容")
    msg_type: str = Field(default="text", description="消息类型: text, image, etc.")
    client_timestamp: Optional[int] = Field(None, sa_type=BigInteger, description="客户端发送时间戳(ms)")
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True, description="服务端入库时间")

class ChatMessageCreate(SQLModel):
    order_id: int
    content: str
    type: str = "text"
    timestamp: Optional[int] = None

class OrderReadStatus(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, sa_type=BigInteger)
    order_id: int = Field(..., index=True, sa_type=BigInteger, description="关联订单ID")
    user_id: str = Field(..., index=True, description="用户ID")
    last_read_msg_id: int = Field(default=0, sa_type=BigInteger, description="最后读取的消息ID")
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ReadAckRequest(SQLModel):
    order_id: int
    latest_message_id: int

class UserInfo(SQLModel):
    id: str
    username: str
    nickname: Optional[str] = None
    avatar: Optional[str] = None

class OrderWithDetails(SQLModel):
    id: int
    consumer: UserInfo
    provider: UserInfo
    task_id: Optional[int] = None
    supply_id: Optional[int] = None
    amount: float
    status: str
    created_at: datetime
    start_time: Optional[datetime] = None

class UserRegister(SQLModel):
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    password: str = Field(..., description="密码")

class SocialLoginRequest(SQLModel):
    provider: str
    token: str

class CancelRequest(SQLModel):
    id: int
    type: str = Field(..., description="类型: task, supply, order")

# --- API 路由 ---

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, auth_utils.SECRET_KEY, algorithms=[auth_utils.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = session.get(User, user_id)
    if user is None:
        raise credentials_exception
    return user

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的哈弗辛距离 (km)"""
    R = 6371  # 地球半径 (km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@app.get("/")
async def root():
    return {"status": "ok", "message": "VisionUber API is running"}

@app.get("/feed", response_model=List[Union[SupplyRead, TaskRead]])
async def get_feed(
    is_consumer: bool = Query(True, description="角色标识: True 为消费者(看供给), False 为供给者(看需求)"),
    user_lat: Optional[float] = Query(None, description="用户当前纬度"),
    user_lng: Optional[float] = Query(None, description="用户当前经度"),
    session: Session = Depends(get_session)
):
    """
    根据角色返回信息流:
    - 消费者 (is_consumer=True): 看到的是周围的 [供给/服务]
    - 供给者 (is_consumer=False): 看到的是周围的 [任务/需求]
    """
    now = datetime.utcnow()

    if is_consumer:
        # 消费者看“谁能帮我看”
        results = session.exec(select(Supply, User.nickname, User.avatar).join(User, Supply.user_id == User.id).where(Supply.status == "created").where(Supply.valid_to > now)).all()
        items = []
        for supply, nickname, avatar in results:
            item = SupplyRead.model_validate(supply)
            item.nickname = nickname
            item.avatar = avatar
            items.append(item)
    else:
        # 供给者看“谁想看什么”
        results = session.exec(select(Task, User.nickname, User.avatar).join(User, Task.user_id == User.id).where(Task.status == "created").where(Task.valid_to > now)).all()
        items = []
        for task, nickname, avatar in results:
            item = TaskRead.model_validate(task)
            item.nickname = nickname
            item.avatar = avatar
            items.append(item)
            
        if user_lat is not None and user_lng is not None:
            items = sorted(items, key=lambda x: haversine_distance(user_lat, user_lng, x.lat, x.lng))
        
    return items

@app.post("/create")
async def create_entry(
    item: dict,
    is_consumer: bool = Query(True, description="角色标识: True 为消费者(发布需求), False 为供给者(发布服务)"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    创建发布:
    - 消费者 (is_consumer=True): 发布需求
    - 供给者 (is_consumer=False): 发布服务
    """
    # 强制使用当前登录用户的 ID，防止伪造
    item["user_id"] = current_user.id

    # 强制设置状态为 created，忽略前端传入的状态
    item["status"] = "created"

    if is_consumer:
        try:
            task_item = Task.model_validate(item)
            session.add(task_item)
            session.commit()
            session.refresh(task_item)
            session.add(TaskLog(task_id=task_item.id, operator_id=current_user.id, action="create", previous_status=None, new_status="created"))
            session.commit()
            return {"status": "success", "task_id": task_item.id}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"数据格式错误 (期望 Task): {e}")
    else:
        try:
            supply_item = Supply.model_validate(item)
            session.add(supply_item)
            session.commit()
            session.refresh(supply_item)
            session.add(SupplyLog(supply_id=supply_item.id, operator_id=current_user.id, action="create", previous_status=None, new_status="created"))
            session.commit()
            return {"status": "success", "supply_id": supply_item.id}
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"数据格式错误 (期望 Supply): {e}")

@app.post("/cancel")
async def cancel_entry(
    req: CancelRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    取消发布的任务、供给或订单
    """
    if req.type not in ["task", "supply", "order"]:
        raise HTTPException(status_code=400, detail="Invalid type. Must be 'task', 'supply' or 'order'")

    if req.type == "order":
        order = session.get(Order, req.id)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        if current_user.id not in [order.consumer_id, order.provider_id]:
            raise HTTPException(status_code=403, detail="Permission denied")

        if order.status != "matched":
             raise HTTPException(status_code=400, detail="Only matched orders can be canceled")

        item = None
        is_publisher = False
        
        if order.task_id:
            item = session.get(Task, order.task_id)
            if item and item.user_id == current_user.id:
                is_publisher = True
        elif order.supply_id:
            item = session.get(Supply, order.supply_id)
            if item and item.user_id == current_user.id:
                is_publisher = True
        
        previous_order_status = order.status
        # 1. 发布者取消 -> 逻辑同取消 Task/Supply (双向取消)
        if is_publisher:
            order.status = "canceled"
            session.add(order)
            session.add(OrderLog(order_id=order.id, operator_id=current_user.id, action="cancel", previous_status=previous_order_status, new_status="canceled"))
            if item:
                prev_item_status = item.status
                item.status = "canceled"
                session.add(item)
                if isinstance(item, Task):
                    session.add(TaskLog(task_id=item.id, operator_id=current_user.id, action="cancel_by_order", previous_status=prev_item_status, new_status="canceled"))
                elif isinstance(item, Supply):
                    session.add(SupplyLog(supply_id=item.id, operator_id=current_user.id, action="cancel_by_order", previous_status=prev_item_status, new_status="canceled"))
        # 2. 接单者/下单者取消 -> 订单取消，Task/Supply 回归池子
        else:
            order.status = "canceled"
            session.add(order)
            session.add(OrderLog(order_id=order.id, operator_id=current_user.id, action="cancel", previous_status=previous_order_status, new_status="canceled"))
            if item:
                prev_item_status = item.status
                item.status = "created"
                session.add(item)
                if isinstance(item, Task):
                    session.add(TaskLog(task_id=item.id, operator_id=current_user.id, action="revert_by_order_cancel", previous_status=prev_item_status, new_status="created"))
                elif isinstance(item, Supply):
                    session.add(SupplyLog(supply_id=item.id, operator_id=current_user.id, action="revert_by_order_cancel", previous_status=prev_item_status, new_status="created"))
        
        session.commit()
        return {"status": "success", "message": "Order canceled"}

    item = None
    if req.type == "task":
        item = session.get(Task, req.id)
    else:
        item = session.get(Supply, req.id)

    if not item:
        raise HTTPException(status_code=404, detail=f"{req.type} not found")

    # 1. 校验是否是用户发布的
    if item.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Permission denied")

    # 2. 校验状态
    if item.status not in ["created", "matched"]:
        raise HTTPException(status_code=400, detail="Only created or matched items can be canceled")

    # 3. 修改状态
    previous_status = item.status
    item.status = "canceled"
    session.add(item)
    
    if req.type == "task":
        session.add(TaskLog(task_id=item.id, operator_id=current_user.id, action="cancel", previous_status=previous_status, new_status="canceled"))
    else:
        session.add(SupplyLog(supply_id=item.id, operator_id=current_user.id, action="cancel", previous_status=previous_status, new_status="canceled"))

    if previous_status == "matched":
        # 找到关联的订单也取消
        if req.type == "task":
            order = session.exec(select(Order).where(Order.task_id == item.id).where(Order.status != "canceled")).first()
        else:
            order = session.exec(select(Order).where(Order.supply_id == item.id).where(Order.status != "canceled")).first()
        
        if order:
            prev_ord_status = order.status
            order.status = "canceled"
            session.add(order)
            session.add(OrderLog(order_id=order.id, operator_id=current_user.id, action="cancel_by_item", previous_status=prev_ord_status, new_status="canceled"))

    session.commit()
    return {"status": "success", "message": f"{req.type} canceled"}

@app.get("/search")
async def search(
    q: str,
    is_consumer: bool = Query(True, description="角色标识: True 为消费者(搜供给), False 为供给者(搜需求)"),
    user_lat: Optional[float] = Query(None, description="用户当前纬度"),
    user_lng: Optional[float] = Query(None, description="用户当前经度"),
    session: Session = Depends(get_session)
):
    """
    搜索功能:
    - 消费者 (is_consumer=True): 搜索供给库
    - 供给者 (is_consumer=False): 搜索任务库
    """
    now = datetime.utcnow()

    if is_consumer:
        db_results = session.exec(select(Supply, User.nickname, User.avatar).join(User, Supply.user_id == User.id).where(Supply.title.contains(q)).where(Supply.status == "created").where(Supply.valid_to > now).order_by(Supply.rating.desc())).all()
        results = []
        for supply, nickname, avatar in db_results:
            item = SupplyRead.model_validate(supply)
            item.nickname = nickname
            item.avatar = avatar
            results.append(item)
        target = "supply"
    else:
        db_results = session.exec(select(Task, User.nickname, User.avatar).join(User, Task.user_id == User.id).where(Task.title.contains(q)).where(Task.status == "created").where(Task.valid_to > now)).all()
        results = []
        for task, nickname, avatar in db_results:
            item = TaskRead.model_validate(task)
            item.nickname = nickname
            item.avatar = avatar
            results.append(item)
        target = "task"
        if user_lat is not None and user_lng is not None:
            results = sorted(results, key=lambda x: haversine_distance(user_lat, user_lng, x.lat, x.lng))

    return {"query": q, "target": target, "results": results}

@app.get("/history/task", response_model=List[Task])
async def get_task_history(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    获取当前用户的发布需求(Task)历史
    状态逻辑: 优先读取关联 Order 的状态 (如 live_end, paied)，若无 Order 则使用 Task 自身状态
    """
    # 关联查询 Task 和 Order 的状态
    # 过滤掉已取消的订单，只关注有效流转中的订单状态
    statement = (
        select(Task, Order.status)
        .outerjoin(Order, (Order.task_id == Task.id) & (Order.status != "canceled"))
        .where(Task.user_id == current_user.id)
        .order_by(Task.created_at.desc())
    )
    results = session.exec(statement).all()
    
    tasks = []
    now = datetime.utcnow()
    for task, order_status in results:
        # 1. 如果存在关联订单，订单状态即为任务的最新状态 (覆盖 matched)
        if order_status:
            task.status = order_status
        
        # 2. 如果没有订单 (处于 created 状态)，检查是否超时
        if task.status in ["created", "matched"] and task.valid_to < now:
            task.status = "timeout" # 仅修改内存对象，不写入数据库
        
        tasks.append(task)
        
    return tasks

@app.get("/history/supply", response_model=List[Supply])
async def get_supply_history(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    
    """
    获取当前用户的发布服务(Supply)历史
    状态逻辑: 优先读取关联 Order 的状态
    """
    statement = (
        select(Supply, Order.status)
        .outerjoin(Order, (Order.supply_id == Supply.id) & (Order.status != "canceled"))
        .where(Supply.user_id == current_user.id)
        .order_by(Supply.created_at.desc())
    )
    results = session.exec(statement).all()
    
    supplies = []
    now = datetime.utcnow()
    for supply, order_status in results:
        if order_status:
            supply.status = order_status
            
        if supply.status in ["created", "matched"] and supply.valid_to < now:
            supply.status = "timeout"
        supplies.append(supply)
        
    return supplies

def verify_google_token(token: str) -> dict:
    """
    验证 Google ID Token。
    生产环境建议使用 google-auth 库，这里使用标准 HTTP 请求以减少依赖。
    """
    url = f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Invalid Google token")
            data = json.loads(response.read().decode())
            if "sub" not in data:
                 raise HTTPException(status_code=400, detail="Invalid Google token payload")
            return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token verification failed: {str(e)}")

def verify_facebook_token(token: str) -> dict:
    """
    验证 Facebook Access Token
    """
    url = f"https://graph.facebook.com/me?access_token={token}&fields=id,name,email,picture"
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Invalid Facebook token")
            data = json.loads(response.read().decode())
            
            # 映射 Facebook 字段到系统标准字段
            return {
                "sub": data["id"],
                "email": data.get("email"),
                "name": data.get("name"),
                "picture": data.get("picture", {}).get("data", {}).get("url")
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Facebook verification failed: {str(e)}")

def verify_apple_token(token: str) -> dict:
    """
    验证 Apple ID Token (JWT)
    """
    keys_url = "https://appleid.apple.com/auth/keys"
    try:
        # 1. 获取 Token Header 中的 kid
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        
        # 2. 获取 Apple 公钥集
        with urllib.request.urlopen(keys_url) as response:
            jwks = json.loads(response.read().decode())
            
        # 3. 验证并解码
        # 注意：生产环境应验证 aud (client_id)，这里为通用性暂时跳过 aud 验证
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={"verify_aud": False} 
        )
        
        return {
            "sub": payload["sub"],
            "email": payload.get("email"),
            "name": payload.get("name"), # Apple Token 仅在首次授权时包含 name
            "picture": None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Apple verification failed: {str(e)}")

@app.post("/auth/social-login")
async def social_login(
    req: SocialLoginRequest,
    session: Session = Depends(get_session)
):
    if req.provider not in ["google", "facebook", "apple"]:
        raise HTTPException(status_code=400, detail=f"Provider '{req.provider}' is not supported")

    # 1. 验证身份 (分发到对应的验证函数)
    if req.provider == "google":
        ext_user_info = verify_google_token(req.token)
    elif req.provider == "facebook":
        ext_user_info = verify_facebook_token(req.token)
    elif req.provider == "apple":
        ext_user_info = verify_apple_token(req.token)
    
    provider_user_id = ext_user_info["sub"]
    email = ext_user_info.get("email")
    name = ext_user_info.get("name", "")
    picture = ext_user_info.get("picture", None)

    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    # 2. 匹配或创建账号 (影子账号逻辑)
    # A. 查关联表 SocialAccount
    social_acc = session.exec(select(SocialAccount).where(
        SocialAccount.provider == req.provider,
        SocialAccount.provider_user_id == provider_user_id
    )).first()

    user = None
    if social_acc:
        user = session.get(User, social_acc.user_id)
    else:
        # B. 没绑定过，查 User 表看 email 是否存在 (账号合并逻辑)
        user = session.exec(select(User).where(User.email == email)).first()
        if not user:
            # C. 彻底的新用户，创建 User
            # 生成一个随机或基于邮箱的用户名
            base_username = email.split("@")[0]
            new_username = f"{base_username}_{str(uuid.uuid4())[:4]}"
            user = User(username=new_username, email=email, hashed_password=None, avatar=picture, nickname=name)
            session.add(user)
            session.commit()
            session.refresh(user)
        
        # D. 建立关联关系
        new_social = SocialAccount(provider=req.provider, provider_user_id=provider_user_id, user_id=user.id)
        session.add(new_social)
        session.commit()

    # 3. 发放系统 JWT
    access_token = auth_utils.create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id, "username": user.username, "nickname": user.nickname, "email": user.email, "avatar": user.avatar}

@app.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
):
    # 1. 查询用户
    user = session.exec(select(User).where(
        (User.username == form_data.username) | (User.email == form_data.username)
    )).first()
    
    # 2. 验证用户是否存在及密码是否正确
    if not user or not user.hashed_password or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 3. 生成 JWT 令牌
    access_token_expires = timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_utils.create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "nickname": user.nickname,
        "email": user.email,
        "avatar": user.avatar
    }

@app.post("/register")
async def register(
    user_in: UserRegister,
    session: Session = Depends(get_session)
):
    # 1. 检查用户名或邮箱是否已存在
    existing_user = session.exec(select(User).where((User.username == user_in.username) | (User.email == user_in.email))).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    # 2. 加密密码并创建用户
    hashed_password = auth_utils.get_password_hash(user_in.password)
    db_user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed_password,
        nickname=user_in.username
    )
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return {"status": "success", "user_id": db_user.id, "username": db_user.username}

# --- 订单系统路由 ---

@app.post("/orders/task/{task_id}/accept")
async def accept_task(
    task_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    供给者接单: 供给者接受消费者的需求 (Task)
    """
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot accept your own task")
    if task.status != "created":
        raise HTTPException(status_code=400, detail="Task is not created")

    # 双重校验（Double Check）：
    # 即使列表接口过滤了过期数据，用户可能在页面停留过久导致数据过期，或通过 API 直接调用，因此必须在写入前再次检查
    if task.valid_to < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Task has expired")

    # 创建订单
    order = Order(
        consumer_id=task.user_id,
        provider_id=current_user.id,
        task_id=task.id,
        amount=task.budget,
        status="matched" # 接单即开始
    )
    
    # 更新任务状态
    task.status = "matched"
    
    session.add(order)
    session.add(task)
    session.commit()
    session.refresh(order)
    
    session.add(OrderLog(order_id=order.id, operator_id=current_user.id, action="create", previous_status=None, new_status="matched"))
    session.add(TaskLog(task_id=task.id, operator_id=current_user.id, action="match", previous_status="created", new_status="matched"))
    session.commit()
    return {"status": "success", "order_id": order.id, "message": "Task accepted", "start_time": task.valid_from}

@app.post("/orders/supply/{supply_id}/book")
async def book_supply(
    supply_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    消费者下单: 消费者预订供给者的服务 (Supply)
    """
    supply = session.get(Supply, supply_id)
    if not supply:
        raise HTTPException(status_code=404, detail="Supply not found")
    if supply.user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot book your own supply")
    
    if supply.status != "created":
        raise HTTPException(status_code=400, detail="Supply is not created")

    # 双重校验（Double Check）：
    # 防止并发情况下的过期接单（如用户在 09:59 加载页面，在 10:01 点击下单，而过期时间是 10:00）
    if supply.valid_to < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Supply has expired")

    # 创建订单
    order = Order(
        consumer_id=current_user.id,
        provider_id=supply.user_id,
        supply_id=supply.id,
        amount=supply.price,
        status="matched" # 待确认或直接开始，这里假设创建即待服务
    )
    
    # 更新供给状态
    supply.status = "matched"
    session.add(supply)
    
    session.add(order)
    session.commit()
    session.refresh(order)
    
    session.add(OrderLog(order_id=order.id, operator_id=current_user.id, action="create", previous_status=None, new_status="matched"))
    session.add(SupplyLog(supply_id=supply.id, operator_id=current_user.id, action="match", previous_status="created", new_status="matched"))
    session.commit()
    return {"status": "success", "order_id": order.id, "message": "Supply booked", "start_time": supply.valid_from}

@app.get("/orders", response_model=List[OrderWithDetails])
async def get_orders(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    获取当前用户的订单列表 (作为消费者或供给者)，包含用户信息
    """
    Consumer = aliased(User)
    Provider = aliased(User)

    statement = select(Order, Consumer, Provider, Task, Supply).join(
        Consumer, Order.consumer_id == Consumer.id
    ).join(
        Provider, Order.provider_id == Provider.id
    ).outerjoin(
        Task, Order.task_id == Task.id
    ).outerjoin(
        Supply, Order.supply_id == Supply.id
    ).where(
        (Order.consumer_id == current_user.id) | 
        (Order.provider_id == current_user.id)
    )
    
    results = session.exec(statement).all()
    
    now = datetime.utcnow()
    orders_data = []
    for order, consumer, provider, task, supply in results:
        # 惰性计算 (Lazy Evaluation):
        # 如果订单处于 created 状态，但关联的任务或供给已过期，则显示为 timeout
        if order.status == "matched":
            if task and task.valid_to < now:
                order.status = "timeout"
            elif supply and supply.valid_to < now:
                order.status = "timeout"

        start_time = None
        if task:
            start_time = task.valid_from
        elif supply:
            start_time = supply.valid_from

        orders_data.append(OrderWithDetails(
            id=order.id,
            consumer=UserInfo(id=consumer.id, username=consumer.username, nickname=consumer.nickname, avatar=consumer.avatar),
            provider=UserInfo(id=provider.id, username=provider.username, nickname=provider.nickname, avatar=provider.avatar),
            task_id=order.task_id,
            supply_id=order.supply_id,
            amount=order.amount,
            status=order.status,
            created_at=order.created_at,
            start_time=start_time
        ))

    # 排序逻辑：
    # 1. 优先显示 status="created" 的订单
    # 2. 对于 "created" 订单，按 start_time 升序排列 (即将开始的在前)
    # 3. 对于其他状态，按 created_at 降序排列 (最近创建的在前)
    orders_data.sort(key=lambda x: (
        0 if x.status == "matched" else 1,
        x.start_time if x.status == "matched" and x.start_time else datetime.max,
        -x.created_at.timestamp()
    ))
    
    return orders_data

# --- Agora 通话路由 ---

@app.get("/agora/rtm-token")
async def get_rtm_token(
    current_user: User = Depends(get_current_user)
):
    """
    获取 Agora RTM Token (用于 App 启动时登录信令系统)
    """
    agora_uid = current_user.id
    rtm_token = auth_utils.create_agora_rtm_token(agora_uid)
    
    return {
        "app_id": auth_utils.AGORA_APP_ID,
        "rtm_token": rtm_token,
        "uid": agora_uid
    }

@app.get("/agora/rtc-token")
async def get_rtc_token(
    order_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    获取 Agora RTC Token (用于订单内的音视频通话)
    """
    order = session.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # 1. 权限校验: 只有订单的双方可以获取 Token
    if current_user.id not in [order.consumer_id, order.provider_id]:
        raise HTTPException(status_code=403, detail="Permission denied: Not a participant of this order")
        
    # 2. 状态校验: 只有进行中的订单允许通话
    allow_rtc = order.status in ["matched", "live_start"]

    # 3. 生成 Token
    agora_uid = current_user.id
    channel_name = f"order_{order.id}"
    
    # 计算对方的 UID (Peer UID)
    if current_user.id == order.consumer_id:
        peer_user_id = order.provider_id
    else:
        peer_user_id = order.consumer_id
    peer_uid = peer_user_id

    # 仅在允许通话的状态下生成 RTC Token，否则返回 None
    token = None
    if allow_rtc:
        token = auth_utils.create_agora_token(channel_name, agora_uid)
    
    print(f"[Agora Debug] User={agora_uid} requesting RTC Token for Channel={channel_name}")
    
    return {
        "app_id": auth_utils.AGORA_APP_ID,
        "token": token,
        "channel_name": channel_name,
        "uid": agora_uid, # 返回处理后的 UID (无减号)
        "peer_uid": peer_uid # 返回对方的 UID，用于 P2P 消息
    }

@app.post("/messages/send")
async def save_chat_message(
    msg_in: ChatMessageCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    保存聊天记录 (客户端通过 RTM 发送消息时同步调用此接口)
    """
    order = session.get(Order, msg_in.order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # 根据当前用户和订单关系，自动推断接收者，防止伪造
    if current_user.id == order.consumer_id:
        receiver_id = order.provider_id
    elif current_user.id == order.provider_id:
        receiver_id = order.consumer_id
    else:
        raise HTTPException(status_code=403, detail="Permission denied: Not a participant")

    chat_msg = ChatMessage(
        order_id=order.id,
        sender_id=current_user.id,
        receiver_id=receiver_id,
        content=msg_in.content,
        msg_type=msg_in.type,
        client_timestamp=msg_in.timestamp
    )
    session.add(chat_msg)
    session.commit()
    session.refresh(chat_msg)
    return {"status": "success", "msg_id": chat_msg.id}

@app.get("/messages/history", response_model=List[ChatMessage])
async def get_chat_history(
    order_id: Optional[int] = None,
    since_id: Optional[int] = Query(None, description="上次同步的消息ID (用于增量同步)"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    获取聊天历史记录。
    - 场景1 (App启动): 不传 order_id，传本地最新一条消息的 since_id，拉取所有新消息。
    - 场景2 (进入某订单): 传 order_id，拉取该订单的所有历史。
    """
    statement = select(ChatMessage).where(
        (ChatMessage.sender_id == current_user.id) | 
        (ChatMessage.receiver_id == current_user.id)
    )
    
    if order_id:
        statement = statement.where(ChatMessage.order_id == order_id)
    
    if since_id:
        statement = statement.where(ChatMessage.id > since_id)
        
    # 按 ID 升序排列 (旧 -> 新)，方便客户端追加到 UI 底部
    statement = statement.order_by(ChatMessage.id.asc())
    
    return session.exec(statement).all()

@app.get("/unread-counts")
async def get_unread_counts(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    获取全局未读消息计数 (App 启动/重连时调用)
    返回格式: {"order_id_1": 5, "order_id_2": 1}
    """
    # 使用聚合查询一次性计算所有订单的未读数，避免 N+1 查询
    # 逻辑: 统计 receiver_id 是我，且 消息ID > 我的 last_read_msg_id 的消息数量
    statement = (
        select(ChatMessage.order_id, func.count(ChatMessage.id))
        .outerjoin(
            OrderReadStatus, 
            (OrderReadStatus.order_id == ChatMessage.order_id) & 
            (OrderReadStatus.user_id == current_user.id)
        )
        .where(ChatMessage.receiver_id == current_user.id)
        .where(ChatMessage.id > func.coalesce(OrderReadStatus.last_read_msg_id, 0))
        .group_by(ChatMessage.order_id)
    )
    
    results = session.exec(statement).all()
    return {order_id: count for order_id, count in results}

@app.post("/read-ack")
async def ack_read_message(
    req: ReadAckRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    提交已读回执 (进入聊天页面时调用)
    """
    # 查询或创建阅读状态记录
    read_status = session.exec(select(OrderReadStatus).where(
        OrderReadStatus.order_id == req.order_id,
        OrderReadStatus.user_id == current_user.id
    )).first()
    
    if not read_status:
        read_status = OrderReadStatus(order_id=req.order_id, user_id=current_user.id, last_read_msg_id=req.latest_message_id)
        session.add(read_status)
    else:
        # 仅当新 ID 更大时更新 (防止乱序请求导致回退)
        if req.latest_message_id > read_status.last_read_msg_id:
            read_status.last_read_msg_id = req.latest_message_id
            read_status.updated_at = datetime.utcnow()
            session.add(read_status)
            
    session.commit()
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)