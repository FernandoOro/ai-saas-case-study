# 🚀 Ejemplos de Arquitectura FastAPI - FastStrat

"""
Este archivo contiene ejemplos generalizados de la arquitectura FastAPI implementada
en el proyecto FastStrat. Los ejemplos muestran patrones y estructuras utilizadas
sin revelar código específico de la empresa.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from enum import Enum
from firebase_admin import firestore
from fastapi.requests import Request
from fastapi.responses import JSONResponse

# =================== MODELOS PYDANTIC ===================

class UserRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    USER = "user"
    CLIENT = "client"

class PlanType(str, Enum):
    TRIAL = "trial"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    SUSPENDED = "suspended"

# Modelos de Request
class CreateUserRequest(BaseModel):
    email: str = Field(..., description="Email del usuario")
    full_name: str = Field(..., description="Nombre completo")
    password: str = Field(..., description="Contraseña")
    company: Optional[str] = Field(None, description="Empresa")
    role: UserRole = Field(UserRole.USER, description="Rol del usuario")

class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = Field(None, description="Nombre completo")
    company: Optional[str] = Field(None, description="Empresa")
    status: Optional[UserStatus] = Field(None, description="Estado del usuario")

class CreatePlanRequest(BaseModel):
    plan_type: PlanType = Field(..., description="Tipo de plan")
    max_seats: int = Field(..., ge=1, le=100, description="Número máximo de seats")
    user_id: str = Field(..., description="ID del usuario propietario")

# Modelos de Response
class UserResponse(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: UserRole
    status: UserStatus
    created_date: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True

class PlanResponse(BaseModel):
    plan_id: str
    plan_type: PlanType
    status: str
    max_seats: int
    unassigned_seats: int
    created_at: datetime
    end_date: datetime
    
    class Config:
        from_attributes = True

# =================== SERVICIOS ===================

class UserService:
    """Servicio para gestión de usuarios"""
    
    def __init__(self, db_client, auth_service):
        self.db = db_client
        self.auth_service = auth_service
    
    async def create_user(self, user_data: CreateUserRequest) -> UserResponse:
        """Crear un nuevo usuario"""
        # Validación de datos
        if await self._email_exists(user_data.email):
            raise ValueError("Email ya existe")
        
        # Crear usuario en base de datos
        user_id = await self._create_user_in_db(user_data)
        
        # Crear usuario en sistema de autenticación
        await self.auth_service.create_auth_user(user_data.email, user_data.password)
        
        return await self.get_user(user_id)
    
    async def get_user(self, user_id: str) -> Optional[UserResponse]:
        """Obtener usuario por ID"""
        user_data = await self.db.collection("users").document(user_id).get()
        if not user_data.exists:
            return None
        return UserResponse(**user_data.to_dict())
    
    async def update_user(self, user_id: str, update_data: UpdateUserRequest) -> UserResponse:
        """Actualizar usuario"""
        user_ref = self.db.collection("users").document(user_id)
        update_dict = update_data.dict(exclude_unset=True)
        
        await user_ref.update(update_dict)
        return await self.get_user(user_id)
    
    async def _email_exists(self, email: str) -> bool:
        """Verificar si el email ya existe"""
        users = self.db.collection("users").where("email", "==", email).limit(1).stream()
        return len(list(users)) > 0
    
    async def _create_user_in_db(self, user_data: CreateUserRequest) -> str:
        """Crear usuario en base de datos"""
        user_ref = self.db.collection("users").document()
        user_dict = {
            "user_id": user_ref.id,
            "email": user_data.email.lower(),
            "full_name": user_data.full_name,
            "role": user_data.role,
            "status": UserStatus.ACTIVE,
            "created_date": datetime.now(),
            "company": user_data.company
        }
        await user_ref.set(user_dict)
        return user_ref.id

class PlanService:
    """Servicio para gestión de planes"""
    
    def __init__(self, db_client, payment_service):
        self.db = db_client
        self.payment_service = payment_service
    
    async def create_plan(self, plan_data: CreatePlanRequest) -> PlanResponse:
        """Crear un nuevo plan"""
        # Validar límites según tipo de plan
        self._validate_plan_limits(plan_data.plan_type, plan_data.max_seats)
        
        # Crear plan en base de datos
        plan_id = await self._create_plan_in_db(plan_data)
        
        # Configurar pagos si es necesario
        if plan_data.plan_type != PlanType.TRIAL:
            await self.payment_service.setup_plan_payments(plan_id, plan_data)
        
        return await self.get_plan(plan_id)
    
    async def get_plan(self, plan_id: str) -> Optional[PlanResponse]:
        """Obtener plan por ID"""
        plan_data = await self.db.collection("plans").document(plan_id).get()
        if not plan_data.exists:
            return None
        return PlanResponse(**plan_data.to_dict())
    
    async def update_plan_usage(self, plan_id: str, generation_used: bool = False) -> bool:
        """Actualizar uso del plan"""
        plan_ref = self.db.collection("plans").document(plan_id)
        
        update_data = {}
        if generation_used:
            update_data["content_generations_used"] = firestore.Increment(1)
        
        await plan_ref.update(update_data)
        return True
    
    def _validate_plan_limits(self, plan_type: PlanType, max_seats: int):
        """Validar límites del plan"""
        limits = {
            PlanType.TRIAL: 1,
            PlanType.BASIC: 5,
            PlanType.PREMIUM: 20,
            PlanType.ENTERPRISE: 100
        }
        
        if max_seats > limits[plan_type]:
            raise ValueError(f"Plan {plan_type} no permite {max_seats} seats")
    
    async def _create_plan_in_db(self, plan_data: CreatePlanRequest) -> str:
        """Crear plan en base de datos"""
        plan_ref = self.db.collection("plans").document()
        plan_dict = {
            "plan_id": plan_ref.id,
            "plan_type": plan_data.plan_type,
            "status": "active",
            "max_seats": plan_data.max_seats,
            "unassigned_seats": plan_data.max_seats,
            "created_at": datetime.now(),
            "end_date": datetime.now() + timedelta(days=30),
            "user_ids": [plan_data.user_id],
            "content_generations_used": 0,
            "content_generations_limit": self._get_generation_limit(plan_data.plan_type)
        }
        await plan_ref.set(plan_dict)
        return plan_ref.id
    
    def _get_generation_limit(self, plan_type: PlanType) -> int:
        """Obtener límite de generaciones según tipo de plan"""
        limits = {
            PlanType.TRIAL: 10,
            PlanType.BASIC: 100,
            PlanType.PREMIUM: 1000,
            PlanType.ENTERPRISE: -1  # Ilimitado
        }
        return limits[plan_type]

# =================== MIDDLEWARE DE AUTENTICACIÓN ===================

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)) -> Dict[str, Any]:
    """Obtener usuario actual desde token"""
    try:
        # Decodificar y validar token
        payload = decode_jwt_token(token.credentials)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )
        
        return {"user_id": user_id, "role": payload.get("role")}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido"
        )

def require_permission(permission: str):
    """Decorador para requerir permisos específicos"""
    def decorator(func):
        async def wrapper(*args, current_user: Dict = Depends(get_current_user), **kwargs):
            if not has_permission(current_user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permisos insuficientes"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

def has_permission(user: Dict[str, Any], permission: str) -> bool:
    """Verificar si usuario tiene permiso específico"""
    role_permissions = {
        UserRole.OWNER: ["*"],  # Todos los permisos
        UserRole.ADMIN: ["read:plans", "write:plans", "read:users", "write:users"],
        UserRole.USER: ["read:plans", "read:content", "write:content"],
        UserRole.CLIENT: ["read:content"]
    }
    
    user_role = user.get("role", UserRole.USER)
    permissions = role_permissions.get(user_role, [])
    
    return "*" in permissions or permission in permissions

# =================== ENDPOINTS ===================

def create_app() -> FastAPI:
    """Crear aplicación FastAPI"""
    app = FastAPI(
        title="FastStrat API",
        description="API para plataforma de marketing inteligente",
        version="1.0.0"
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configurar en producción
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Inicializar servicios
    db_client = get_firestore_client()
    auth_service = get_auth_service()
    payment_service = get_payment_service()
    
    user_service = UserService(db_client, auth_service)
    plan_service = PlanService(db_client, payment_service)
    
    # =================== RUTAS DE USUARIOS ===================
    
    @app.post("/users", response_model=UserResponse)
    async def create_user(request: CreateUserRequest):
        """Crear nuevo usuario"""
        try:
            return await user_service.create_user(request)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    
    @app.get("/users/{user_id}", response_model=UserResponse)
    @require_permission("read:users")
    async def get_user(user_id: str, current_user: Dict = Depends(get_current_user)):
        """Obtener usuario por ID"""
        user = await user_service.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )
        return user
    
    @app.put("/users/{user_id}", response_model=UserResponse)
    @require_permission("write:users")
    async def update_user(
        user_id: str, 
        request: UpdateUserRequest,
        current_user: Dict = Depends(get_current_user)
    ):
        """Actualizar usuario"""
        try:
            return await user_service.update_user(user_id, request)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    
    # =================== RUTAS DE PLANES ===================
    
    @app.post("/plans", response_model=PlanResponse)
    @require_permission("write:plans")
    async def create_plan(
        request: CreatePlanRequest,
        current_user: Dict = Depends(get_current_user)
    ):
        """Crear nuevo plan"""
        try:
            return await plan_service.create_plan(request)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    
    @app.get("/plans/{plan_id}", response_model=PlanResponse)
    @require_permission("read:plans")
    async def get_plan(
        plan_id: str,
        current_user: Dict = Depends(get_current_user)
    ):
        """Obtener plan por ID"""
        plan = await plan_service.get_plan(plan_id)
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan no encontrado"
            )
        return plan
    
    @app.get("/users/{user_id}/plans", response_model=List[PlanResponse])
    @require_permission("read:plans")
    async def get_user_plans(
        user_id: str,
        current_user: Dict = Depends(get_current_user)
    ):
        """Obtener planes de un usuario"""
        # Implementar lógica para obtener planes del usuario
        pass
    
    return app

# =================== FUNCIONES AUXILIARES ===================

def decode_jwt_token(token: str) -> Dict[str, Any]:
    """Decodificar token JWT"""
    # Implementar decodificación de JWT
    # Esta es una implementación simplificada
    pass

def get_firestore_client():
    """Obtener cliente de Firestore"""
    # Implementar conexión a Firestore
    pass

def get_auth_service():
    """Obtener servicio de autenticación"""
    # Implementar servicio de autenticación
    pass

def get_payment_service():
    """Obtener servicio de pagos"""
    # Implementar servicio de pagos
    pass

# =================== CONFIGURACIÓN DE LOGGING ===================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# =================== MANEJO DE ERRORES ===================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Manejador global de excepciones"""
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor"}
    )

# =================== HEALTH CHECK ===================

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 