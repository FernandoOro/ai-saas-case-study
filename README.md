# 🧠 AI SaaS Case Study – Scalable Platform with FastAPI & Multi-LLM Integration

This repository documents the architecture, design decisions, and example code from a real-world AI-powered SaaS platform. The platform was built using **FastAPI**, **Firebase Firestore**, and integrates multiple **LLMs (OpenAI, Claude, Vertex AI)** with guardrails, caching, payments via Stripe, and more.

> ⚠️ This case study is based on real experience developing a production-grade system. All code here is generalized and does **not** expose any confidential logic.

---

## 🧱 Architecture Overview

- **Backend:** FastAPI, Firebase Firestore, Firebase Auth
- **Frontend:** Svelte (SPA), reactive store, dynamic routing
- **Authentication:** JWT + Firebase + granular permission matrix
- **AI Providers:** OpenAI, Claude, Vertex AI (via adapters + fallback)
- **Vector DBs:** ChromaDB, Qdrant
- **Payments:** Stripe + Webhooks
- **DevOps:** Docker, GitHub Actions, GCP Cloud Run
- **Observability:** OpenTelemetry, Sentry, PostHog
- **PDF Generation:** ReportLab, Markdown

---

## 📁 Structure
.
├── README.md # This file
├── docs/ # Technical documentation and architecture explanations
│ ├── arquitectura_escalable_fastapi.md
│ ├── technical_architecture.md
│ └── architecture.md
├── code_examples/ # Generalized code for backend and AI services
│ ├── fastapi_architecture.py
│ └── ai_integration_example.py
└── diagrams/ # Optional: PNG renders of architecture diagrams


---

## 📘 Documentation Highlights

### [`arquitectura_escalable_fastapi.md`](docs/arquitectura_escalable_fastapi.md)

In-depth explanation of:
- Modular folder structure
- Repository and Service Layer patterns
- Auth with role-based access control
- AI integration (adapters + fallback)
- Caching, rate limiting, and logging
- CI/CD and Docker pipeline

### [`technical_architecture.md`](docs/technical_architecture.md)

Covers:
- Design principles: scalability, security, modularity
- Database structure (Firestore) and indexes
- Stripe integration with webhooks
- Guardrails for AI content validation
- Performance metrics and logging strategy

### [`architecture.md`](docs/architecture.md)

Includes multiple Mermaid diagrams:
- System architecture
- Auth & permission flow
- Plan management & payment handling
- AI content generation pipeline
- Deployment & monitoring infrastructure

---

## 🧪 Code Examples

### [`fastapi_architecture.py`](code_examples/fastapi_architecture.py)

Example of a FastAPI application with:
- Pydantic models for input/output
- Service layer with validation and logic
- Middleware for auth and permissions
- Endpoints for user and plan management

### [`ai_integration_example.py`](code_examples/ai_integration_example.py)

Robust AI integration service:
- Adapters for OpenAI, Claude, and Vertex AI
- Guardrails for profanity, toxicity, and brand safety
- Fallback logic and cost tracking
- Strategy and blog content generation
- Caching layer and endpoint simulation

---

## 🚀 Highlights

- ✅ Scalable architecture with real-world patterns
- ✅ AI features with multi-LLM abstraction and validation
- ✅ Payment automation via Stripe Webhooks
- ✅ Testing, monitoring, and CI/CD integration
- ✅ Clear documentation and code examples

---

## 🧠 Author

**Uriel Fernando Orozco Castillo**  
AI Engineer & System Architect  
🔗 [LinkedIn](https://www.linkedin.com/in/tu-perfil)  
📫 uriel@example.com

---

## 🛡️ Disclaimer

This project is a **generalized reconstruction** of a real-world SaaS platform. All proprietary code, data, and business logic have been abstracted or removed.

