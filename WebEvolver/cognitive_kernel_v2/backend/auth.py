from datetime import datetime, timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette import status
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from database import User, UserSessionLocal

router = APIRouter(prefix="/auth", tags=["auth"])

SECRET_KEY = "CognitiveKernelYYDS211115"
ALGORITHM = "HS256"

bcyrpt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/token")

developer_usernames = list()
with open("/app/developer_users.txt", "r") as f:
    for line in f:
        developer_usernames.append(line.strip())


class CreateUserRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


def get_db():
    try:
        db = UserSessionLocal()
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(create_user_request: CreateUserRequest, db: db_dependency):
    existing_user = (
        db.query(User).filter(User.username == create_user_request.username).first()
    )
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    create_user_model = User(
        username=create_user_request.username,
        password=bcyrpt_context.hash(create_user_request.password),
    )
    db.add(create_user_model)
    db.commit()
    return {"message": "User created successfully"}


def authenticate_user(username: str, password: str, db: db_dependency):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not bcyrpt_context.verify(password, user.password):
        return False
    return user


def create_access_token(
    username: str, user_id: int, expires_delta: timedelta = timedelta(minutes=30)
):
    to_encode = {"sub": username, "id": user_id}
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encode_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encode_jwt


@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: db_dependency
):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token_expires = timedelta(days=30)
    access_token = create_access_token(
        username=user.username, user_id=user.id, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/check_developer")
async def check_developer(username: str):
    if username in developer_usernames:
        return {"is_developer": True}
    else:
        return {"is_developer": False}


@router.get("/get_current_user")
async def get_current_user(db: db_dependency, token: str = Depends(oauth2_bearer)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("id")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user
