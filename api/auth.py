import time
import hashlib
import logging
from typing import Optional, List, Dict, Any, Callable
from fastapi import Depends, HTTPException, status, Security, Request
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    SecurityScopes,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError
from datetime import datetime, timedelta

# Constants and Config
SECRET_KEY = "CHANGE_ME_TO_A_RANDOM_SECRET"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_HOURS = 24
JWT_ISSUER = "my_data_platform"
CSRF_HEADER_NAME = "X-CSRF-Token"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt", "argon2"], deprecated="auto")

# OAuth2/JWT setup
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scopes={"admin": "Admin privileges", "user": "Standard user", "viewer": "Read-only access"},
)

# --- Models ---

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []
    roles: List[str] = []
    exp: Optional[int] = None

class User(BaseModel):
    username: str
    hashed_password: str
    roles: List[str]
    is_active: bool = True

# Simulated database for users, roles, and permissions
fake_users_db: Dict[str, User] = {
    "admin": User(
        username="admin",
        hashed_password=pwd_context.hash("adminpw"),
        roles=["admin", "user"],
    ),
    "alice": User(
        username="alice",
        hashed_password=pwd_context.hash("alicepw"),
        roles=["user"],
    ),
    "bob": User(
        username="bob",
        hashed_password=pwd_context.hash("bobpw"),
        roles=["viewer"],
    ),
}
# Role-based access control matrix
ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage_users"],
    "user": ["read", "write"],
    "viewer": ["read"]
}

# --- Utility Functions ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password using the hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password securely."""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[User]:
    return fake_users_db.get(username)

def authenticate_user(username: str, password: str) -> Optional[User]:
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def get_user_permissions(user: User) -> List[str]:
    """Get all permissions for the user based on roles."""
    perms = set()
    for role in user.roles:
        perms.update(ROLE_PERMISSIONS.get(role, []))
    return list(perms)

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
    csrf_token: Optional[str] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iss": JWT_ISSUER})
    if csrf_token:
        to_encode["csrf"] = csrf_token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=REFRESH_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire, "iss": JWT_ISSUER, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> TokenData:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], issuer=JWT_ISSUER)
        username: str = payload.get("sub")
        scopes = payload.get("scopes") or []
        roles = payload.get("roles") or []
        return TokenData(username=username, scopes=scopes, roles=roles, exp=payload.get("exp"))
    except JWTError as e:
        raise credentials_exception(str(e))

def credentials_exception(msg: str = "Could not validate credentials"):
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=msg,
        headers={"WWW-Authenticate": "Bearer"},
    )

def check_csrf(request: Request, csrf_from_jwt: Optional[str]):
    """Check CSRF header matches token."""
    csrf_from_header = request.headers.get(CSRF_HEADER_NAME)
    if not csrf_from_header or not csrf_from_jwt or csrf_from_header != csrf_from_jwt:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing or invalid",
        )

# --- OAuth2, JWT and FastAPI route dependencies ---

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
) -> User:
    """Get current user from token, enforcing required scopes."""
    token_data = decode_token(token)
    user = get_user(token_data.username)
    if not user:
        raise credentials_exception("User not found")
    # Check scopes/permissions
    token_scopes = set(token_data.scopes or [])
    required_scopes = set(security_scopes.scopes)
    if required_scopes and not required_scopes.issubset(token_scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

def require_permissions(required: List[str]) -> Callable:
    """Return a dependency that enforces the required permissions."""
    def dependency(user: User = Depends(get_current_user)):
        user_perms = get_user_permissions(user)
        if not set(required).issubset(set(user_perms)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user
    return dependency

# --- Token Endpoints (to be used in FastAPI routes) ---

from fastapi import APIRouter, Form

router = APIRouter()

@router.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Issue access and refresh tokens."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise credentials_exception("Incorrect username or password")
    csrf_token = hashlib.sha256(f"{user.username}{time.time()}".encode()).hexdigest()
    access_token = create_access_token(
        data={
            "sub": user.username,
            "scopes": form_data.scopes,
            "roles": user.roles,
        },
        csrf_token=csrf_token
    )
    refresh_token = create_refresh_token(
        data={
            "sub": user.username,
            "scopes": form_data.scopes,
            "roles": user.roles,
        }
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=refresh_token,
    )

@router.post("/auth/refresh", response_model=Token)
async def refresh_access_token(refresh_token: str = Form(...)):
    """Refresh access token using a valid refresh token."""
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM], issuer=JWT_ISSUER)
        if payload.get("type") != "refresh":
            raise credentials_exception("Not a refresh token")
        username = payload.get("sub")
        user = get_user(username)
        if not user:
            raise credentials_exception("User not found")
        csrf_token = hashlib.sha256(f"{user.username}{time.time()}".encode()).hexdigest()
        access_token = create_access_token(
            data={
                "sub": user.username,
                "scopes": payload.get("scopes"),
                "roles": user.roles,
            },
            csrf_token=csrf_token
        )
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
    except JWTError as e:
        raise credentials_exception(str(e))

@router.post("/auth/logout")
async def revoke_token(token: str = Depends(oauth2_scheme)):
    """Revoke the current token (stateless, client must discard)."""
    # For stateless JWT, revocation is typically client-side or with a denylist.
    # Here, we just instruct the client to discard.
    return {"message": "Token revoked. Please discard your token on the client."}

# --- External Identity Providers (stub for integration) ---

def authenticate_with_provider(provider: str, token: str) -> Optional[User]:
    """Stub for authenticating with external identity providers, e.g., LDAP, OAuth."""
    # Implement as needed for SSO/enterprise
    return None

# --- Example usage in FastAPI routes ---

# from fastapi import FastAPI, Depends, Security
# app = FastAPI()
# app.include_router(router)
#
# @app.get("/protected-route")
# async def protected_route(user: User = Security(get_current_user, scopes=["user"])):
#     return {"message": f"Hello, {user.username}!"}
