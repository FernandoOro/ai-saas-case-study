# 🏗️ Cómo Diseñar una Arquitectura Escalable con FastAPI

*Basado en la experiencia desarrollando FastStrat - Plataforma de Marketing Inteligente*

## Introducción

En este artículo comparto las lecciones aprendidas al diseñar e implementar una arquitectura escalable con FastAPI para una plataforma de marketing inteligente. El sistema maneja miles de usuarios, integra múltiples servicios de IA, y procesa pagos en tiempo real.

## 🎯 Principios de Diseño

### 1. Separación de Responsabilidades

La arquitectura se divide en capas bien definidas:

```python
# Ejemplo de estructura modular
app/
├── api/
│   ├── v1/
│   │   ├── users.py
│   │   ├── plans.py
│   │   └── payments.py
├── services/
│   ├── user_service.py
│   ├── plan_service.py
│   └── payment_service.py
├── models/
│   ├── user.py
│   ├── plan.py
│   └── payment.py
└── core/
    ├── config.py
    ├── security.py
    └── database.py
```

### 2. Patrón Repository

Implementamos el patrón Repository para abstraer la capa de datos:

```python
class UserRepository:
    def __init__(self, db_client):
        self.db = db_client
    
    async def create(self, user_data: Dict) -> str:
        user_ref = self.db.collection("users").document()
        user_id = user_ref.id
        await user_ref.set({**user_data, "user_id": user_id})
        return user_id
    
    async def get_by_id(self, user_id: str) -> Optional[Dict]:
        doc = await self.db.collection("users").document(user_id).get()
        return doc.to_dict() if doc.exists else None
    
    async def update(self, user_id: str, data: Dict) -> bool:
        await self.db.collection("users").document(user_id).update(data)
        return True
```

### 3. Service Layer Pattern

Los servicios encapsulan la lógica de negocio:

```python
class UserService:
    def __init__(self, user_repo: UserRepository, auth_service: AuthService):
        self.user_repo = user_repo
        self.auth_service = auth_service
    
    async def create_user(self, user_data: CreateUserRequest) -> UserResponse:
        # Validación de datos
        if await self._email_exists(user_data.email):
            raise ValueError("Email ya existe")
        
        # Crear usuario en base de datos
        user_id = await self.user_repo.create(user_data.dict())
        
        # Crear usuario en sistema de autenticación
        await self.auth_service.create_auth_user(user_data.email, user_data.password)
        
        return await self.get_user(user_id)
```

## 🔐 Sistema de Autenticación y Autorización

### Middleware de Autenticación

```python
async def get_current_user(token: str = Depends(HTTPBearer())) -> Dict[str, Any]:
    try:
        payload = decode_jwt_token(token.credentials)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Token inválido")
        
        return {"user_id": user_id, "role": payload.get("role")}
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido")
```

### Sistema de Permisos Granular

```python
def require_permission(permission: str):
    def decorator(func):
        async def wrapper(*args, current_user: Dict = Depends(get_current_user), **kwargs):
            if not has_permission(current_user, permission):
                raise HTTPException(status_code=403, detail="Permisos insuficientes")
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

def has_permission(user: Dict[str, Any], permission: str) -> bool:
    role_permissions = {
        "owner": ["*"],
        "admin": ["read:plans", "write:plans", "read:users", "write:users"],
        "user": ["read:plans", "read:content", "write:content"],
        "client": ["read:content"]
    }
    
    user_role = user.get("role", "user")
    permissions = role_permissions.get(user_role, [])
    
    return "*" in permissions or permission in permissions
```

## 🚀 Optimizaciones de Performance

### 1. Caching Inteligente

```python
class CacheManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.default_ttl = 3600
    
    async def get_or_set(self, key: str, getter_func: Callable, ttl: int = None) -> Any:
        cached_value = await self.redis.get(key)
        if cached_value:
            return json.loads(cached_value)
        
        fresh_value = await getter_func()
        await self.redis.setex(key, ttl or self.default_ttl, json.dumps(fresh_value))
        return fresh_value
```

### 2. Consultas Optimizadas

```python
async def get_user_with_plans(self, user_id: str) -> Dict:
    # Ejecutar consultas en paralelo
    user_doc, plans_docs = await asyncio.gather(
        self.db.collection('users').document(user_id).get(),
        self.db.collection('plans').where('user_ids', 'array-contains', user_id).get()
    )
    
    return {
        'user': user_doc.to_dict(),
        'plans': [doc.to_dict() for doc in plans_docs]
    }
```

### 3. Rate Limiting

```python
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(self, user_id: str, endpoint: str, limit: int = 100) -> bool:
        key = f"rate_limit:{user_id}:{endpoint}"
        current = await self.redis.incr(key)
        
        if current == 1:
            await self.redis.expire(key, 3600)  # 1 hora
        
        return current <= limit
```

## 🔄 Integración con Servicios Externos

### Patrón Adapter para Múltiples LLMs

```python
class LLMAdapter:
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "claude": ClaudeProvider(),
            "vertex": VertexAIProvider()
        }
    
    async def generate_content(self, prompt: str, provider: str = "openai") -> str:
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not supported")
        
        return await self.providers[provider].generate(prompt)
    
    async def generate_with_fallback(self, prompt: str) -> str:
        for provider in self.providers.values():
            try:
                return await provider.generate(prompt)
            except Exception:
                continue
        raise Exception("All providers failed")
```

### Webhooks con Validación

```python
class WebhookHandler:
    def __init__(self, stripe_client):
        self.stripe = stripe_client
    
    async def handle_stripe_webhook(self, payload: bytes, signature: str) -> bool:
        try:
            event = self.stripe.Webhook.construct_event(
                payload, signature, os.getenv("STRIPE_WEBHOOK_SECRET")
            )
            
            if event['type'] == 'checkout.session.completed':
                return await self.process_successful_payment(event)
            elif event['type'] == 'invoice.payment_failed':
                return await self.process_failed_payment(event)
            
            return True
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False
```

## 📊 Monitoreo y Observabilidad

### Logging Estructurado

```python
class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_api_request(self, endpoint: str, user_id: str, duration: float):
        self.logger.info("API Request", extra={
            "endpoint": endpoint,
            "user_id": user_id,
            "duration_ms": duration * 1000,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def log_ai_generation(self, model: str, tokens_used: int, cost: float):
        self.logger.info("AI Generation", extra={
            "model": model,
            "tokens_used": tokens_used,
            "cost_usd": cost,
            "timestamp": datetime.utcnow().isoformat()
        })
```

### Métricas de Performance

```python
class PerformanceMetrics:
    def __init__(self):
        self.request_times = []
        self.error_counts = defaultdict(int)
    
    def record_request_time(self, endpoint: str, duration: float):
        self.request_times.append((endpoint, duration))
        
        # Mantener solo las últimas 1000 mediciones
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
    
    def get_average_response_time(self, endpoint: str = None) -> float:
        if endpoint:
            times = [t for e, t in self.request_times if e == endpoint]
        else:
            times = [t for _, t in self.request_times]
        
        return sum(times) / len(times) if times else 0
```

## 🧪 Testing y Calidad

### Tests Unitarios

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestUserService:
    @pytest.fixture
    def user_service(self):
        mock_repo = AsyncMock()
        mock_auth = AsyncMock()
        return UserService(mock_repo, mock_auth)
    
    @pytest.mark.asyncio
    async def test_create_user_success(self, user_service):
        # Arrange
        user_data = CreateUserRequest(
            email="test@example.com",
            full_name="Test User",
            password="password123"
        )
        user_service.user_repo.create.return_value = "user_123"
        user_service._email_exists.return_value = False
        
        # Act
        result = await user_service.create_user(user_data)
        
        # Assert
        assert result.user_id == "user_123"
        user_service.user_repo.create.assert_called_once()
        user_service.auth_service.create_auth_user.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_user_email_exists(self, user_service):
        # Arrange
        user_data = CreateUserRequest(
            email="existing@example.com",
            full_name="Test User",
            password="password123"
        )
        user_service._email_exists.return_value = True
        
        # Act & Assert
        with pytest.raises(ValueError, match="Email ya existe"):
            await user_service.create_user(user_data)
```

### Tests de Integración

```python
class TestAPIIntegration:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)
    
    def test_create_user_endpoint(self, client):
        response = client.post("/users", json={
            "email": "test@example.com",
            "full_name": "Test User",
            "password": "password123"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert "user_id" in data
```

## 🚀 Deployment y DevOps

### Docker Multi-stage

```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### GitHub Actions CI/CD

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy faststrat-api \
            --image gcr.io/project/faststrat-api \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated
```

## 📈 Lecciones Aprendidas

### 1. Diseño de Base de Datos
- **Normalización vs Denormalización**: Para consultas frecuentes, la denormalización puede mejorar el performance significativamente
- **Índices Estratégicos**: Crear índices compuestos para consultas complejas
- **Particionamiento**: Considerar particionamiento por fecha para colecciones grandes

### 2. Manejo de Errores
- **Logging Estructurado**: Usar logs estructurados para facilitar el debugging
- **Circuit Breaker**: Implementar circuit breakers para servicios externos
- **Retry Logic**: Implementar retry con backoff exponencial

### 3. Seguridad
- **Validación de Entrada**: Usar Pydantic para validación estricta
- **Rate Limiting**: Implementar rate limiting por usuario y endpoint
- **Audit Logs**: Registrar todas las acciones críticas

### 4. Performance
- **Caching**: Cachear datos frecuentemente accedidos
- **Async/Await**: Usar async/await para operaciones I/O
- **Connection Pooling**: Reutilizar conexiones de base de datos

## 🎯 Conclusiones

Diseñar una arquitectura escalable con FastAPI requiere:

1. **Separación clara de responsabilidades** entre capas
2. **Patrones de diseño** como Repository y Service Layer
3. **Sistema de autenticación robusto** con permisos granulares
4. **Optimizaciones de performance** (caching, consultas optimizadas)
5. **Monitoreo completo** del sistema
6. **Testing exhaustivo** en todos los niveles
7. **Deployment automatizado** con CI/CD

La clave está en mantener el código modular, bien documentado y fácil de mantener, mientras se optimiza para el performance y la escalabilidad.

---

*Este artículo está basado en la experiencia real desarrollando FastStrat, una plataforma de marketing inteligente que maneja miles de usuarios y integra múltiples servicios de IA.* 