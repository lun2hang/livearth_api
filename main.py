from fastapi import FastAPI, Query, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from typing import List, Optional, Union
from pydantic import ValidationError
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from sqlalchemy import BigInteger
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
    status: str = Field(default="created", description="状态: created, live_start, live_end, paied, canceled, timeout")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserInfo(SQLModel):
    id: str
    username: str
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

@app.get("/feed", response_model=List[Union[Supply, Task]])
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
        items = session.exec(select(Supply).where(Supply.status == "created").where(Supply.valid_to > now)).all()
    else:
        # 供给者看“谁想看什么”
        items = session.exec(select(Task).where(Task.status == "created").where(Task.valid_to > now)).all()
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

        if order.status != "created":
             raise HTTPException(status_code=400, detail="Only created orders can be canceled")

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
        
        # 1. 发布者取消 -> 逻辑同取消 Task/Supply (双向取消)
        if is_publisher:
            order.status = "canceled"
            session.add(order)
            if item:
                item.status = "canceled"
                session.add(item)
        # 2. 接单者/下单者取消 -> 订单取消，Task/Supply 回归池子
        else:
            order.status = "canceled"
            session.add(order)
            if item:
                item.status = "created"
                session.add(item)
        
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

    if previous_status == "matched":
        # 找到关联的订单也取消
        if req.type == "task":
            order = session.exec(select(Order).where(Order.task_id == item.id).where(Order.status != "canceled")).first()
        else:
            order = session.exec(select(Order).where(Order.supply_id == item.id).where(Order.status != "canceled")).first()
        
        if order:
            order.status = "canceled"
            session.add(order)

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
        results = session.exec(select(Supply).where(Supply.title.contains(q)).where(Supply.status == "created").where(Supply.valid_to > now).order_by(Supply.rating.desc())).all()
        target = "supply"
    else:
        results = session.exec(select(Task).where(Task.title.contains(q)).where(Task.status == "created").where(Task.valid_to > now)).all()
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
    """
    tasks = session.exec(select(Task).where(Task.user_id == current_user.id)).all()
    
    # 惰性计算 (Lazy Evaluation):
    # 数据库中保留原始状态，但在返回给前端时，根据时间判断是否已超时。
    # 这样既不需要高频写数据库，又能让用户看到正确的状态。
    now = datetime.utcnow()
    for task in tasks:
        if task.status == "created" and task.valid_to < now:
            task.status = "timeout" # 仅修改内存对象，不写入数据库
            
    return tasks

@app.get("/history/supply", response_model=List[Supply])
async def get_supply_history(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    
    """
    获取当前用户的发布服务(Supply)历史
    """
    supplies = session.exec(select(Supply).where(Supply.user_id == current_user.id)).all()
    
    now = datetime.utcnow()
    for supply in supplies:
        if supply.status == "created" and supply.valid_to < now:
            supply.status = "timeout"
            
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
            user = User(username=new_username, email=email, hashed_password=None, avatar=picture)
            session.add(user)
            session.commit()
            session.refresh(user)
        
        # D. 建立关联关系
        new_social = SocialAccount(provider=req.provider, provider_user_id=provider_user_id, user_id=user.id)
        session.add(new_social)
        session.commit()

    # 3. 发放系统 JWT
    access_token = auth_utils.create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id, "username": user.username, "email": user.email, "avatar": user.avatar}

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
        hashed_password=hashed_password
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
        status="created" # 接单即开始
    )
    
    # 更新任务状态
    task.status = "matched"
    
    session.add(order)
    session.add(task)
    session.commit()
    session.refresh(order)
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
        status="created" # 待确认或直接开始，这里假设创建即待服务
    )
    
    # 更新供给状态
    supply.status = "matched"
    session.add(supply)
    
    session.add(order)
    session.commit()
    session.refresh(order)
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
        if order.status == "created":
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
            consumer=UserInfo(id=consumer.id, username=consumer.username, avatar=consumer.avatar),
            provider=UserInfo(id=provider.id, username=provider.username, avatar=provider.avatar),
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
        0 if x.status == "created" else 1,
        x.start_time if x.status == "created" and x.start_time else datetime.max,
        -x.created_at.timestamp()
    ))
    
    return orders_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)