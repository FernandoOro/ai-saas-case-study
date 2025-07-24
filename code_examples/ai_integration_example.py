# 🤖 Ejemplo de Integración Multi-LLM - FastStrat

"""
Este archivo demuestra cómo implementar una integración robusta con múltiples
proveedores de IA (OpenAI, Claude, Vertex AI) para generar contenido de marketing.
Basado en la experiencia desarrollando FastStrat.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json

# =================== MODELOS DE DATOS ===================

class ContentType(str, Enum):
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL_CAMPAIGN = "email_campaign"
    STRATEGY_REPORT = "strategy_report"
    MARKETING_PLAN = "marketing_plan"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    VERTEX_AI = "vertex_ai"
    OLLAMA = "ollama"

@dataclass
class GenerationRequest:
    content_type: ContentType
    prompt: str
    context: Dict[str, Any]
    provider: LLMProvider = LLMProvider.OPENAI
    max_tokens: int = 2000
    temperature: float = 0.7

@dataclass
class GenerationResponse:
    content: str
    provider_used: LLMProvider
    tokens_used: int
    cost_usd: float
    generation_time: float
    metadata: Dict[str, Any]

# =================== ADAPTADORES DE LLM ===================

class BaseLLMProvider:
    """Clase base para proveedores de LLM"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generar contenido con el LLM"""
        raise NotImplementedError
    
    def calculate_cost(self, tokens_used: int) -> float:
        """Calcular costo de la generación"""
        raise NotImplementedError

class OpenAIProvider(BaseLLMProvider):
    """Adaptador para OpenAI GPT"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        self.cost_per_1k_tokens = 0.03  # Ejemplo de costo
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            # Simulación de llamada a OpenAI
            # En implementación real: openai.ChatCompletion.create()
            response = await self._call_openai_api(prompt, max_tokens, temperature)
            return response
        except Exception as e:
            self.logger.error(f"Error generando con OpenAI: {e}")
            raise
    
    async def _call_openai_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Implementación real de llamada a OpenAI
        # Esta es una simulación
        await asyncio.sleep(0.5)  # Simular latencia de red
        return f"Contenido generado por OpenAI GPT-4: {prompt[:50]}..."
    
    def calculate_cost(self, tokens_used: int) -> float:
        return (tokens_used / 1000) * self.cost_per_1k_tokens

class ClaudeProvider(BaseLLMProvider):
    """Adaptador para Anthropic Claude"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet"):
        super().__init__(api_key, model)
        self.cost_per_1k_tokens = 0.015  # Ejemplo de costo
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            # Simulación de llamada a Claude
            response = await self._call_claude_api(prompt, max_tokens, temperature)
            return response
        except Exception as e:
            self.logger.error(f"Error generando con Claude: {e}")
            raise
    
    async def _call_claude_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Implementación real de llamada a Claude
        await asyncio.sleep(0.6)  # Simular latencia de red
        return f"Contenido generado por Claude: {prompt[:50]}..."
    
    def calculate_cost(self, tokens_used: int) -> float:
        return (tokens_used / 1000) * self.cost_per_1k_tokens

class VertexAIProvider(BaseLLMProvider):
    """Adaptador para Google Vertex AI"""
    
    def __init__(self, project_id: str, model: str = "text-bison"):
        super().__init__(None, model)
        self.project_id = project_id
        self.cost_per_1k_tokens = 0.01  # Ejemplo de costo
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            response = await self._call_vertex_api(prompt, max_tokens, temperature)
            return response
        except Exception as e:
            self.logger.error(f"Error generando con Vertex AI: {e}")
            raise
    
    async def _call_vertex_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Implementación real de llamada a Vertex AI
        await asyncio.sleep(0.7)  # Simular latencia de red
        return f"Contenido generado por Vertex AI: {prompt[:50]}..."
    
    def calculate_cost(self, tokens_used: int) -> float:
        return (tokens_used / 1000) * self.cost_per_1k_tokens

# =================== SISTEMA DE GUARDRAILS ===================

class ContentGuardrails:
    """Sistema de guardrails para validar contenido generado"""
    
    def __init__(self):
        self.filters = [
            ProfanityFilter(),
            ToxicityFilter(),
            BrandSafetyFilter(),
            ContentQualityFilter()
        ]
    
    def validate_content(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        """Validar contenido generado"""
        results = {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }
        
        for filter in self.filters:
            filter_result = filter.validate(content, content_type)
            if not filter_result["is_valid"]:
                results["is_valid"] = False
                results["issues"].extend(filter_result["issues"])
            results["suggestions"].extend(filter_result.get("suggestions", []))
        
        return results

class ProfanityFilter:
    """Filtro para detectar lenguaje inapropiado"""
    
    def validate(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        # Implementación real de detección de profanidad
        # Esta es una simulación
        return {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }

class ToxicityFilter:
    """Filtro para detectar contenido tóxico"""
    
    def validate(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        # Implementación real de detección de toxicidad
        return {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }

class BrandSafetyFilter:
    """Filtro para seguridad de marca"""
    
    def validate(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        # Implementación real de filtros de marca
        return {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }

class ContentQualityFilter:
    """Filtro para calidad de contenido"""
    
    def validate(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        # Implementación real de validación de calidad
        return {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }

# =================== SERVICIO PRINCIPAL DE IA ===================

class AIService:
    """Servicio principal para generación de contenido con IA"""
    
    def __init__(self):
        self.providers = {
            LLMProvider.OPENAI: OpenAIProvider("openai_key", "gpt-4"),
            LLMProvider.CLAUDE: ClaudeProvider("claude_key", "claude-3-sonnet"),
            LLMProvider.VERTEX_AI: VertexAIProvider("project_id", "text-bison")
        }
        self.guardrails = ContentGuardrails()
        self.logger = logging.getLogger(__name__)
    
    async def generate_content(self, request: GenerationRequest) -> GenerationResponse:
        """Generar contenido con IA"""
        start_time = datetime.now()
        
        try:
            # Seleccionar proveedor
            provider = self.providers.get(request.provider)
            if not provider:
                raise ValueError(f"Provider {request.provider} not supported")
            
            # Generar contenido
            content = await provider.generate(
                request.prompt,
                request.max_tokens,
                request.temperature
            )
            
            # Validar contenido con guardrails
            validation_result = self.guardrails.validate_content(content, request.content_type)
            
            if not validation_result["is_valid"]:
                self.logger.warning(f"Content validation failed: {validation_result['issues']}")
                # En implementación real, regenerar o aplicar correcciones
            
            # Calcular métricas
            generation_time = (datetime.now() - start_time).total_seconds()
            tokens_used = len(content.split()) * 1.3  # Estimación aproximada
            cost = provider.calculate_cost(tokens_used)
            
            return GenerationResponse(
                content=content,
                provider_used=request.provider,
                tokens_used=int(tokens_used),
                cost_usd=cost,
                generation_time=generation_time,
                metadata={
                    "validation_result": validation_result,
                    "context": request.context
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise
    
    async def generate_with_fallback(self, request: GenerationRequest) -> GenerationResponse:
        """Generar contenido con fallback automático"""
        providers_to_try = [
            request.provider,
            LLMProvider.OPENAI,
            LLMProvider.CLAUDE,
            LLMProvider.VERTEX_AI
        ]
        
        for provider in providers_to_try:
            try:
                request.provider = provider
                return await self.generate_content(request)
            except Exception as e:
                self.logger.warning(f"Provider {provider} failed: {e}")
                continue
        
        raise Exception("All providers failed")
    
    async def generate_marketing_strategy(self, business_context: Dict[str, Any]) -> GenerationResponse:
        """Generar estrategia de marketing completa"""
        prompt = self._build_strategy_prompt(business_context)
        
        request = GenerationRequest(
            content_type=ContentType.MARKETING_PLAN,
            prompt=prompt,
            context=business_context,
            provider=LLMProvider.OPENAI,
            max_tokens=4000,
            temperature=0.7
        )
        
        return await self.generate_content(request)
    
    async def generate_blog_post(self, topic: str, target_audience: str) -> GenerationResponse:
        """Generar post de blog"""
        prompt = f"Escribe un post de blog sobre '{topic}' dirigido a {target_audience}"
        
        request = GenerationRequest(
            content_type=ContentType.BLOG_POST,
            prompt=prompt,
            context={"topic": topic, "target_audience": target_audience},
            provider=LLMProvider.CLAUDE,
            max_tokens=2000,
            temperature=0.8
        )
        
        return await self.generate_content(request)
    
    def _build_strategy_prompt(self, business_context: Dict[str, Any]) -> str:
        """Construir prompt para estrategia de marketing"""
        return f"""
        Analiza el siguiente contexto de negocio y genera una estrategia de marketing completa:
        
        Industria: {business_context.get('industry', 'N/A')}
        Público objetivo: {business_context.get('target_audience', 'N/A')}
        Presupuesto: {business_context.get('budget', 'N/A')}
        Objetivos: {business_context.get('objectives', 'N/A')}
        
        Incluye:
        1. Análisis de mercado
        2. Posicionamiento de marca
        3. Estrategia de contenido
        4. Canales de distribución
        5. Métricas de éxito
        """

# =================== CACHE Y OPTIMIZACIÓN ===================

class AICacheManager:
    """Gestor de cache para respuestas de IA"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hora
    
    async def get_cached_response(self, cache_key: str) -> Optional[GenerationResponse]:
        """Obtener respuesta cacheada"""
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return GenerationResponse(**data)
        except Exception as e:
            logging.error(f"Error getting cached response: {e}")
        return None
    
    async def cache_response(self, cache_key: str, response: GenerationResponse, ttl: int = None):
        """Cachear respuesta"""
        try:
            data = {
                "content": response.content,
                "provider_used": response.provider_used.value,
                "tokens_used": response.tokens_used,
                "cost_usd": response.cost_usd,
                "generation_time": response.generation_time,
                "metadata": response.metadata
            }
            await self.redis.setex(cache_key, ttl or self.default_ttl, json.dumps(data))
        except Exception as e:
            logging.error(f"Error caching response: {e}")
    
    def generate_cache_key(self, request: GenerationRequest) -> str:
        """Generar clave de cache"""
        # Crear hash del request para cache
        request_hash = hash(json.dumps({
            "content_type": request.content_type.value,
            "prompt": request.prompt,
            "provider": request.provider.value,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }, sort_keys=True))
        return f"ai_generation:{request_hash}"

# =================== ENDPOINTS DE API ===================

class AIEndpoints:
    """Endpoints de API para generación de contenido"""
    
    def __init__(self, ai_service: AIService, cache_manager: AICacheManager):
        self.ai_service = ai_service
        self.cache_manager = cache_manager
    
    async def generate_content_endpoint(self, request: GenerationRequest) -> Dict[str, Any]:
        """Endpoint para generar contenido"""
        try:
            # Verificar cache
            cache_key = self.cache_manager.generate_cache_key(request)
            cached_response = await self.cache_manager.get_cached_response(cache_key)
            
            if cached_response:
                return {
                    "success": True,
                    "cached": True,
                    "data": {
                        "content": cached_response.content,
                        "provider_used": cached_response.provider_used.value,
                        "tokens_used": cached_response.tokens_used,
                        "cost_usd": cached_response.cost_usd,
                        "generation_time": cached_response.generation_time
                    }
                }
            
            # Generar contenido
            response = await self.ai_service.generate_content(request)
            
            # Cachear respuesta
            await self.cache_manager.cache_response(cache_key, response)
            
            return {
                "success": True,
                "cached": False,
                "data": {
                    "content": response.content,
                    "provider_used": response.provider_used.value,
                    "tokens_used": response.tokens_used,
                    "cost_usd": response.cost_usd,
                    "generation_time": response.generation_time,
                    "metadata": response.metadata
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_strategy_endpoint(self, business_context: Dict[str, Any]) -> Dict[str, Any]:
        """Endpoint para generar estrategia de marketing"""
        try:
            response = await self.ai_service.generate_marketing_strategy(business_context)
            
            return {
                "success": True,
                "data": {
                    "strategy": response.content,
                    "provider_used": response.provider_used.value,
                    "tokens_used": response.tokens_used,
                    "cost_usd": response.cost_usd,
                    "generation_time": response.generation_time
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# =================== EJEMPLO DE USO ===================

async def main():
    """Ejemplo de uso del sistema de IA"""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Inicializar servicios
    ai_service = AIService()
    cache_manager = AICacheManager(None)  # Sin Redis para el ejemplo
    endpoints = AIEndpoints(ai_service, cache_manager)
    
    # Ejemplo 1: Generar post de blog
    blog_request = GenerationRequest(
        content_type=ContentType.BLOG_POST,
        prompt="Escribe un post sobre estrategias de marketing digital",
        context={"industry": "tech", "audience": "startups"},
        provider=LLMProvider.OPENAI
    )
    
    result = await endpoints.generate_content_endpoint(blog_request)
    print("Blog Post Generation:", json.dumps(result, indent=2))
    
    # Ejemplo 2: Generar estrategia de marketing
    business_context = {
        "industry": "SaaS",
        "target_audience": "Pequeñas empresas",
        "budget": "$10,000/mes",
        "objectives": "Aumentar leads en 50%"
    }
    
    strategy_result = await endpoints.generate_strategy_endpoint(business_context)
    print("Marketing Strategy:", json.dumps(strategy_result, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 