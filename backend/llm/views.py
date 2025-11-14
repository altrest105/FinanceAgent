from adrf.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import get_answer_with_retry 
from django.conf import settings

MAX_QUESTION_LENGTH = settings.LLM_CONFIG.get("max_question_length")

class InvestAIAgentAPIView(APIView):
    async def post(self, request, *args, **kwargs):
        data = request.data 
        
        if 'question' not in data:
            return Response(
                {"detail": "Поле 'question' обязательно в теле запроса."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        question = data['question']

        if not isinstance(question, str) or not (0 < len(question) <= MAX_QUESTION_LENGTH):
            return Response(
                {"detail": f"Поле 'question' должно быть непустой строкой длиной не более {MAX_QUESTION_LENGTH} символов."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        answer = await get_answer_with_retry(question)

        return Response(
            {"question": question, "answer": answer}, 
            status=status.HTTP_200_OK
        )