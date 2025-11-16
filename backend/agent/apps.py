from django.apps import AppConfig
import logging
import asyncio
import os

logger = logging.getLogger(__name__)


class AgentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'agent'
    
    # Глобальные клиенты и модели
    llm_client = None
    embedder_model = None
    reranker_model = None
    retriever = None
    
    def ready(self):
        """Инициализация при старте приложения"""
        import sys
        
        # Пропускаем инициализацию при миграциях и других командах
        skip_commands = ['migrate', 'makemigrations', 'createsuperuser', 'collectstatic', 'shell']
        if any(cmd in sys.argv for cmd in skip_commands):
            logger.info("⏭️ Пропуск инициализации RAG (команда управления)")
            return
        
        # Инициализируем только при запуске сервера
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv[0]:
            logger.info("=" * 80)
            logger.info("🚀 АВТОМАТИЧЕСКАЯ ИНИЦИАЛИЗАЦИЯ RAG СИСТЕМЫ")
            logger.info("=" * 80)
            
            try:
                # Инициализация RAG системы
                from .services.rag import initialize_rag
                
                # Запускаем асинхронную инициализацию
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._init_rag_system())
                loop.close()
                
                logger.info("=" * 80)
                logger.info("✅ RAG СИСТЕМА ГОТОВА К РАБОТЕ")
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error("=" * 80)
                logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА ИНИЦИАЛИЗАЦИИ RAG")
                logger.error("=" * 80)
                logger.error(f"Тип: {type(e).__name__}")
                logger.error(f"Сообщение: {str(e)}", exc_info=True)
                logger.error("=" * 80)
                
                # Не падаем, чтобы Django мог стартовать
                logger.warning("⚠️ Сервер запущен БЕЗ RAG системы!")
    
    async def _init_rag_system(self):
        """Асинхронная инициализация RAG системы"""
        from django.conf import settings
        from openai import AsyncOpenAI
        from .services.rag import (
            initialize_rag,
            LocalEmbedder,
            LocalReranker,
            KnowledgeBaseRetriever
        )
        
        logger.info("=" * 80)
        logger.info("🔧 ЭТАП 1: ИНИЦИАЛИЗАЦИЯ LLM КЛИЕНТА")
        logger.info("=" * 80)
        
        # 1. Инициализация LLM клиента
        try:
            self.__class__.llm_client = AsyncOpenAI(
                base_url=settings.LLM_CONFIG.get("base_url"),
                api_key=settings.LLM_CONFIG.get("api_key"),
            )
            logger.info(f"  ✅ LLM клиент создан")
            logger.info(f"     ├─ Base URL: {settings.LLM_CONFIG.get('base_url')}")
            logger.info(f"     └─ Model: {settings.LLM_CONFIG.get('model')}")
        except Exception as e:
            logger.error(f"  ❌ Ошибка создания LLM клиента: {e}")
            raise
        
        logger.info("=" * 80)
        logger.info("🔧 ЭТАП 2: ЗАГРУЗКА ЛОКАЛЬНЫХ МОДЕЛЕЙ")
        logger.info("=" * 80)
        
        # 2. Инициализация локального эмбеддера
        try:
            logger.info("  ⏳ Загрузка эмбеддера...")
            embedder_model_name = os.getenv(
                "EMBEDDER_MODEL", 
                "intfloat/multilingual-e5-large"
            )
            self.__class__.embedder_model = LocalEmbedder(model_name=embedder_model_name)
            logger.info(f"  ✅ Эмбеддер загружен: {embedder_model_name}")
        except Exception as e:
            logger.error(f"  ❌ Ошибка загрузки эмбеддера: {e}")
            raise
        
        # 3. Инициализация локального реранкера
        try:
            logger.info("  ⏳ Загрузка реранкера...")
            reranker_model_name = os.getenv(
                "RERANKER_MODEL", 
                "BAAI/bge-reranker-v2-m3"
            )
            self.__class__.reranker_model = LocalReranker(model_name=reranker_model_name)
            logger.info(f"  ✅ Реранкер загружен: {reranker_model_name}")
        except Exception as e:
            logger.error(f"  ❌ Ошибка загрузки реранкера: {e}")
            raise
        
        logger.info("=" * 80)
        logger.info("🔧 ЭТАП 3: ИНИЦИАЛИЗАЦИЯ БАЗЫ ЗНАНИЙ")
        logger.info("=" * 80)
        
        # 4. Инициализация базы знаний и retriever
        try:
            # Передаём загруженные модели в RAG систему
            import agent.services.rag as rag_module
            rag_module._local_embedder = self.__class__.embedder_model
            rag_module._local_reranker = self.__class__.reranker_model
            rag_module._llm_client = self.__class__.llm_client
            
            # Инициализируем базу знаний
            await initialize_rag(force_rebuild=False)
            
            # Сохраняем retriever в AppConfig
            self.__class__.retriever = rag_module._retriever
            
            logger.info(f"  ✅ База знаний инициализирована")
            logger.info(f"     └─ Чанков в индексе: {len(self.__class__.retriever.texts)}")
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка инициализации базы знаний: {e}")
            raise
        
        logger.info("=" * 80)
        logger.info("🎉 ВСЕ КОМПОНЕНТЫ УСПЕШНО ЗАГРУЖЕНЫ")
        logger.info("=" * 80)
        logger.info(f"  ├─ LLM клиент: ✅")
        logger.info(f"  ├─ Эмбеддер: ✅")
        logger.info(f"  ├─ Реранкер: ✅")
        logger.info(f"  └─ База знаний: ✅")
        logger.info("=" * 80)


# Глобальные функции-хелперы для доступа к ресурсам
def get_llm_client():
    """Получить LLM клиент"""
    if AgentConfig.llm_client is None:
        logger.warning("⚠️ LLM клиент не инициализирован!")
        from django.conf import settings
        from openai import AsyncOpenAI
        
        AgentConfig.llm_client = AsyncOpenAI(
            base_url=settings.LLM_CONFIG.get("base_url"),
            api_key=settings.LLM_CONFIG.get("api_key"),
        )
    return AgentConfig.llm_client


def get_embedder_model():
    """Получить локальный эмбеддер"""
    if AgentConfig.embedder_model is None:
        logger.warning("⚠️ Эмбеддер не инициализирован!")
        from .services.rag import LocalEmbedder
        AgentConfig.embedder_model = LocalEmbedder()
    return AgentConfig.embedder_model


def get_reranker_model():
    """Получить локальный реранкер"""
    if AgentConfig.reranker_model is None:
        logger.warning("⚠️ Реранкер не инициализирован!")
        from .services.rag import LocalReranker
        AgentConfig.reranker_model = LocalReranker()
    return AgentConfig.reranker_model


def get_retriever():
    """Получить retriever"""
    if AgentConfig.retriever is None:
        logger.warning("⚠️ Retriever не инициализирован!")
        raise RuntimeError("RAG система не инициализирована. Вызовите initialize_rag()")
    return AgentConfig.retriever


def is_rag_ready():
    """Проверить готовность RAG системы"""
    return all([
        AgentConfig.llm_client is not None,
        AgentConfig.embedder_model is not None,
        AgentConfig.reranker_model is not None,
        AgentConfig.retriever is not None,
    ])
