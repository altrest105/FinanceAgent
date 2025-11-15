# agent/views.py

from adrf.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tinkoff.invest import exceptions
from .orchestrator import execute_agent_query
import logging

logger = logging.getLogger(__name__)

class AgentOrchestratorAPIView(APIView):
    async def post(self, request):
        # 1. Получение токена и данных из запроса
        auth_header = request.headers.get("Authorization", "")
        tinkoff_token = auth_header.replace("Bearer ", "").strip()
        
        user_question = request.data.get("question")
        account_id = request.data.get("account_id")
        
        # 2. Валидация
        if not all([tinkoff_token, user_question, account_id]):
            return Response(
                {"error": "Требуются 'Authorization: Bearer <token>' в заголовке, а также 'question' и 'account_id' в теле запроса."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 3. АСИНХРОННЫЙ ВЫЗОВ ОРКЕСТРАТОРА
            result = await execute_agent_query(user_question, tinkoff_token, account_id)
            
            return Response(result, status=status.HTTP_200_OK)

        except exceptions.AioRequestError as e:
            # Обработка ошибок, связанных с Тинькофф API (неверный токен, лимиты)
            logger.error(f"Ошибка API Тинькофф: {e}")
            return Response(
                {"error": "Ошибка API Тинькофф Инвестиций", "detail": str(e), "code": getattr(e.code, 'name', 'UNKNOWN')},
                status=status.HTTP_401_UNAUTHORIZED
            )
        except Exception as e:
            logger.error(f"Критическая ошибка в AgentOrchestratorAPIView: {e}", exc_info=True)
            return Response(
                {"error": "Внутренняя ошибка сервера", "detail": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )