from adrf.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging
import uuid
import traceback

from .orchestrator import process_question, get_demo_user
from .services.memorybank import MemoryBank

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class ChatAPIView(APIView):
    """
    Основной endpoint для чата с агентом
    
    POST /api/chat/
    Body: {
        "question": "Что такое дивиденды?",
        "session_id": "optional-session-id"
    }
    """
    
    permission_classes = []
    
    async def post(self, request):
        logger.info("=" * 80)
        logger.info("🚀 НАЧАЛО ОБРАБОТКИ ЗАПРОСА /api/chat/")
        logger.info("=" * 80)
        
        try:
            question = request.data.get("question")
            session_id = request.data.get("session_id") or str(uuid.uuid4())
            
            logger.info(f"  ├─ question: {question[:100] if question else 'None'}...")
            logger.info(f"  └─ session_id: {session_id}")
            
            if not question or not question.strip():
                logger.warning("⚠️ Пустой вопрос!")
                return Response(
                    {"error": "Поле 'question' обязательно и не может быть пустым"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # ✅ ПОЛУЧАЕМ ДЕМО-ПОЛЬЗОВАТЕЛЯ ИЗ БД
            user = await get_demo_user()
            logger.info(f"👤 Пользователь: {user.username} (ID: {user.id})")
            
            logger.info(f"\n📨 Вопрос от {user.username}: '{question}'\n")
            
            logger.info("🎯 Вызов оркестратора...")
            result = await process_question(
                question=question.strip(),
                user=user,
                session_id=session_id
            )
            
            logger.info(f"✅ Оркестратор завершил работу")
            logger.info(f"  ├─ intent: {result.get('intent')}")
            logger.info(f"  └─ success: {result.get('success')}")
            
            response_data = {
                "session_id": session_id,
                "question": question,
                "answer": result.get("answer", "Произошла ошибка"),
                "intent": result.get("intent", "unknown"),
                "success": result.get("success", False),
                "metadata": {
                    "user_id": user.id,
                    "user_name": user.username,
                }
            }
            
            intent = result.get("intent")
            if intent == "trade":
                response_data["trade_result"] = result.get("trade_result")
            elif intent == "securities_info":
                response_data["securities_info"] = result.get("securities_info")
            
            logger.info("✅ ЗАПРОС УСПЕШНО ОБРАБОТАН\n")
            
            return Response(response_data, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА В ChatAPIView")
            logger.error(traceback.format_exc())
            
            return Response(
                {
                    "error": "Внутренняя ошибка сервера",
                    "detail": str(e),
                    "success": False
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class TokenManagementAPIView(APIView):
    """
    Управление Tinkoff токеном
    
    POST /api/token/save/ - Сохранение токена
    GET /api/token/status/ - Проверка статуса токена
    """
    
    permission_classes = []
    
    async def post(self, request):
        logger.info("🔑 POST /api/token/save/")
        
        try:
            token = request.data.get("token")
            
            if not token:
                return Response(
                    {"error": "Поле 'token' обязательно"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # ✅ ПОЛУЧАЕМ ДЕМО-ПОЛЬЗОВАТЕЛЯ ИЗ БД
            user = await get_demo_user()
            logger.info(f"👤 Сохранение токена для: {user.username}")
            
            memory = MemoryBank(user)
            await memory.save_tinkoff_token(token)
            
            logger.info(f"✅ Токен сохранён для {user.username}")
            
            return Response(
                {
                    "success": True,
                    "message": "Токен Tinkoff успешно сохранён",
                    "has_token": True
                },
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения токена: {e}")
            return Response(
                {"error": "Ошибка при сохранении токена"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    async def get(self, request):
        logger.info("🔍 GET /api/token/status/")
        
        try:
            # ✅ ПОЛУЧАЕМ ДЕМО-ПОЛЬЗОВАТЕЛЯ ИЗ БД
            user = await get_demo_user()
            
            memory = MemoryBank(user)
            has_token = await memory.has_token()
            
            return Response(
                {
                    "has_token": has_token,
                    "authenticated": True,
                    "user": user.username
                },
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(f"❌ Ошибка проверки токена: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class ConversationHistoryAPIView(APIView):
    """
    История диалогов
    
    GET /api/history/?limit=20
    """
    
    permission_classes = []
    
    async def get(self, request):
        logger.info("📜 GET /api/history/")
        
        try:
            # ✅ ПОЛУЧАЕМ ДЕМО-ПОЛЬЗОВАТЕЛЯ ИЗ БД
            user = await get_demo_user()
            
            limit = int(request.query_params.get('limit', 20))
            
            memory = MemoryBank(user)
            history = await memory.get_conversation_history(limit=limit)
            
            logger.info(f"✅ Найдено диалогов: {len(history)}")
            
            return Response(
                {
                    "count": len(history),
                    "history": history,
                    "user": user.username
                },
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(f"❌ Ошибка получения истории: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
