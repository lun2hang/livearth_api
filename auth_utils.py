import bcrypt
import time
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt

# 尝试导入 Agora SDK，请确保已安装: pip install agora-token-builder
try:
    from agora_token_builder import RtcTokenBuilder, RtmTokenBuilder
except ImportError:
    RtcTokenBuilder = None
    RtmTokenBuilder = None

# 密钥配置 (生产环境请务必使用环境变量管理)
SECRET_KEY = "YOUR_SUPER_SECRET_KEY_CHANGE_THIS"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7天

# Agora 配置
AGORA_APP_ID = "0506c3636bcb401bbc282c9901611da2"  # 替换为你的 App ID
AGORA_APP_CERTIFICATE = "fda15748889249a3afb0d74807cc4d77" # 替换为你的 App Certificate

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_agora_token(channel_name: str, user_account: str) -> str:
    """
    生成 Agora RTC Token (使用 String User ID / Account)
    """
    if not RtcTokenBuilder:
        raise ImportError("agora-token-builder not installed. Run `pip install agora-token-builder`")

    # Token 有效期 (例如 1 小时)
    expiration_time_in_seconds = 3600
    current_timestamp = int(time.time())
    privilege_expired_ts = current_timestamp + expiration_time_in_seconds
    
    # Role_Publisher = 1 (主播/连麦者), Role_Subscriber = 2 (观众)
    # 双向通话双方都是 Publisher
    role = 1 

    token = RtcTokenBuilder.buildTokenWithAccount(
        AGORA_APP_ID, AGORA_APP_CERTIFICATE, channel_name, user_account, role, privilege_expired_ts
    )
    return token

def create_agora_rtm_token(user_account: str) -> str:
    """
    生成 Agora RTM Token (用于信令/文字聊天)
    """
    if not RtmTokenBuilder:
        raise ImportError("agora-token-builder not installed. Run `pip install agora-token-builder`")

    # RTM Token 有效期 (例如 1 小时)
    expiration_time_in_seconds = 3600
    current_timestamp = int(time.time())
    privilege_expired_ts = current_timestamp + expiration_time_in_seconds
    
    # Role_Rtm_User = 1
    role = 1

    token = RtmTokenBuilder.buildToken(
        AGORA_APP_ID, AGORA_APP_CERTIFICATE, user_account, role, privilege_expired_ts
    )
    return token
