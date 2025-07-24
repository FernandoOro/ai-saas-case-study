# 🏗️ Diagramas de Arquitectura - FastStrat

## Arquitectura General del Sistema

```mermaid
graph TB
    subgraph "Frontend (Svelte)"
        UI[Interfaz de Usuario]
        Components[Componentes Reactivos]
        State[Estado Global]
    end
    
    subgraph "API Gateway (FastAPI)"
        Router[Router Principal]
        Middleware[Middleware de Auth]
        Validator[Validación Pydantic]
    end
    
    subgraph "Servicios de Negocio"
        UserService[Servicio de Usuarios]
        PlanService[Servicio de Planes]
        PaymentService[Servicio de Pagos]
        AIService[Servicio de IA]
        ContentService[Servicio de Contenido]
    end
    
    subgraph "Base de Datos"
        Firestore[(Firebase Firestore)]
        Auth[(Firebase Auth)]
        VectorDB[(ChromaDB/Qdrant)]
    end
    
    subgraph "Servicios Externos"
        Stripe[Stripe API]
        OpenAI[OpenAI API]
        Claude[Anthropic Claude]
        VertexAI[Google Vertex AI]
        GCP[Google Cloud Platform]
    end
    
    UI --> Router
    Router --> Middleware
    Middleware --> UserService
    Middleware --> PlanService
    Middleware --> PaymentService
    Middleware --> AIService
    Middleware --> ContentService
    
    UserService --> Firestore
    PlanService --> Firestore
    PaymentService --> Stripe
    AIService --> OpenAI
    AIService --> Claude
    AIService --> VertexAI
    AIService --> VectorDB
    ContentService --> Firestore
    ContentService --> GCP
```

## Flujo de Autenticación y Autorización

```mermaid
sequenceDiagram
    participant U as Usuario
    participant F as Frontend
    participant A as API Gateway
    participant M as Middleware Auth
    participant S as Servicios
    participant DB as Firestore
    
    U->>F: Login con credenciales
    F->>A: POST /auth/login
    A->>M: Validar credenciales
    M->>DB: Verificar usuario
    DB-->>M: Datos del usuario
    M->>M: Generar JWT
    M-->>A: Token + permisos
    A-->>F: Respuesta con token
    F->>U: Redirigir a dashboard
    
    Note over U,DB: Solicitudes posteriores
    U->>F: Acceder a recurso
    F->>A: GET /api/resource
    A->>M: Validar JWT
    M->>M: Verificar permisos
    M->>S: Ejecutar lógica de negocio
    S->>DB: Consultar datos
    DB-->>S: Resultados
    S-->>A: Respuesta
    A-->>F: Datos del recurso
```

## Sistema de Planes y Suscripciones

```mermaid
graph LR
    subgraph "Gestión de Planes"
        PlanModel[Modelo de Plan]
        PlanService[Servicio de Planes]
        PlanValidator[Validador de Planes]
    end
    
    subgraph "Sistema de Pagos"
        StripeClient[Cliente Stripe]
        WebhookHandler[Manejador de Webhooks]
        PaymentProcessor[Procesador de Pagos]
    end
    
    subgraph "Base de Datos"
        Plans[(Colección Planes)]
        Users[(Colección Usuarios)]
        Payments[(Colección Pagos)]
    end
    
    subgraph "Estados de Plan"
        Active[Activo]
        Expired[Expirado]
        Suspended[Suspendido]
        Pending[Pendiente]
    end
    
    PlanModel --> PlanService
    PlanService --> PlanValidator
    PlanService --> StripeClient
    StripeClient --> WebhookHandler
    WebhookHandler --> PaymentProcessor
    PaymentProcessor --> Plans
    PaymentProcessor --> Users
    Plans --> Active
    Plans --> Expired
    Plans --> Suspended
    Plans --> Pending
```

## Arquitectura de IA y Generación de Contenido

```mermaid
graph TB
    subgraph "Capa de Presentación"
        API[API Endpoints]
        WebSocket[WebSocket]
    end
    
    subgraph "Capa de IA"
        AIGateway[Gateway de IA]
        LLMAdapter[Adaptador Multi-LLM]
        CrewAIService[Servicio CrewAI]
        LangChainService[Servicio LangChain]
    end
    
    subgraph "Proveedores de IA"
        OpenAI[OpenAI GPT]
        Claude[Anthropic Claude]
        VertexAI[Google Vertex AI]
        Local[Ollama Local]
    end
    
    subgraph "Almacenamiento Vectorial"
        ChromaDB[ChromaDB]
        Qdrant[Qdrant]
        Embeddings[Embeddings Cache]
    end
    
    subgraph "Procesamiento"
        ContentProcessor[Procesador de Contenido]
        PDFGenerator[Generador de PDFs]
        ReportGenerator[Generador de Reportes]
    end
    
    API --> AIGateway
    WebSocket --> AIGateway
    AIGateway --> LLMAdapter
    LLMAdapter --> OpenAI
    LLMAdapter --> Claude
    LLMAdapter --> VertexAI
    LLMAdapter --> Local
    
    AIGateway --> CrewAIService
    AIGateway --> LangChainService
    
    CrewAIService --> ChromaDB
    LangChainService --> Qdrant
    LangChainService --> Embeddings
    
    AIGateway --> ContentProcessor
    ContentProcessor --> PDFGenerator
    ContentProcessor --> ReportGenerator
```

## Sistema de Seguridad y Permisos

```mermaid
graph TD
    subgraph "Autenticación"
        JWT[JWT Token]
        FirebaseAuth[Firebase Auth]
        SessionManager[Gestor de Sesiones]
    end
    
    subgraph "Autorización"
        PermissionMatrix[Matriz de Permisos]
        RoleValidator[Validador de Roles]
        ResourceGuard[Guardia de Recursos]
    end
    
    subgraph "Roles del Sistema"
        Owner[Owner]
        Admin[Admin]
        User[User]
        Client[Client]
    end
    
    subgraph "Recursos Protegidos"
        Plans[Planes]
        Content[Contenido]
        Analytics[Analytics]
        Settings[Configuración]
    end
    
    JWT --> FirebaseAuth
    FirebaseAuth --> SessionManager
    SessionManager --> PermissionMatrix
    PermissionMatrix --> RoleValidator
    RoleValidator --> ResourceGuard
    
    Owner --> Plans
    Owner --> Content
    Owner --> Analytics
    Owner --> Settings
    
    Admin --> Plans
    Admin --> Content
    Admin --> Analytics
    
    User --> Content
    User --> Analytics
    
    Client --> Content
```

## Flujo de Generación de Contenido

```mermaid
sequenceDiagram
    participant U as Usuario
    participant API as API Gateway
    participant AI as Servicio de IA
    participant LLM as LLM Provider
    participant Vector as Vector DB
    participant PDF as PDF Generator
    participant Storage as Cloud Storage
    
    U->>API: Solicitar generación de contenido
    API->>AI: Procesar solicitud
    AI->>Vector: Buscar contexto relevante
    Vector-->>AI: Contexto encontrado
    AI->>LLM: Generar contenido con contexto
    LLM-->>AI: Contenido generado
    AI->>AI: Aplicar guardrails
    AI->>PDF: Generar PDF del contenido
    PDF->>Storage: Guardar PDF
    Storage-->>PDF: URL del PDF
    AI-->>API: Contenido + PDF URL
    API-->>U: Respuesta completa
```

## Deployment y Infraestructura

```mermaid
graph TB
    subgraph "Desarrollo"
        LocalDev[Desarrollo Local]
        DockerDev[Docker Development]
    end
    
    subgraph "CI/CD"
        GitHub[GitHub Actions]
        Build[Build Process]
        Test[Test Suite]
    end
    
    subgraph "Cloud Platform"
        GCP[Google Cloud Platform]
        CloudRun[Cloud Run]
        CloudStorage[Cloud Storage]
        CloudSQL[Cloud SQL]
    end
    
    subgraph "Monitoreo"
        OpenTelemetry[OpenTelemetry]
        Sentry[Sentry]
        PostHog[PostHog]
        Logs[Cloud Logging]
    end
    
    LocalDev --> GitHub
    DockerDev --> GitHub
    GitHub --> Build
    Build --> Test
    Test --> GCP
    GCP --> CloudRun
    CloudRun --> CloudStorage
    CloudRun --> CloudSQL
    
    CloudRun --> OpenTelemetry
    CloudRun --> Sentry
    CloudRun --> PostHog
    CloudRun --> Logs
```

## Estructura de Base de Datos

```mermaid
erDiagram
    USERS {
        string user_id PK
        string email
        string full_name
        string role
        string status
        datetime created_date
        string plan_type
    }
    
    PLANS {
        string plan_id PK
        string plan_type
        string status
        datetime start_date
        datetime end_date
        int max_seats
        int unassigned_seats
        string agency_id FK
    }
    
    AGENCIES {
        string agency_id PK
        string master_user_id FK
        string agency_type
        string status
        int max_clients
        float monthly_cost
    }
    
    CONTENT_GENERATIONS {
        string generation_id PK
        string user_id FK
        string plan_id FK
        string content_type
        json metadata
        datetime created_at
    }
    
    CALENDAR_EVENTS {
        string event_id PK
        string plan_id FK
        string user_id FK
        string title
        datetime start_date
        datetime end_date
    }
    
    SAVED_CHATS {
        string chat_id PK
        string plan_id FK
        string user_id FK
        string title
        string category
        datetime created_at
    }
    
    USERS ||--o{ PLANS : "owns"
    USERS ||--o{ CONTENT_GENERATIONS : "creates"
    USERS ||--o{ CALENDAR_EVENTS : "creates"
    USERS ||--o{ SAVED_CHATS : "creates"
    PLANS ||--o{ CONTENT_GENERATIONS : "allows"
    PLANS ||--o{ CALENDAR_EVENTS : "contains"
    PLANS ||--o{ SAVED_CHATS : "contains"
    AGENCIES ||--o{ PLANS : "manages"
    USERS ||--o{ AGENCIES : "owns"
``` 