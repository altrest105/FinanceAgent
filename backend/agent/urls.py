from django.urls import path
from .views import (
    ChatAPIView,
    TokenManagementAPIView,
    ConversationHistoryAPIView,
)

urlpatterns = [
    # Основной чат
    path('chat/', ChatAPIView.as_view(), name='chat'),
    
    # Управление токеном
    path('token/save/', TokenManagementAPIView.as_view(), name='token-save'),
    path('token/status/', TokenManagementAPIView.as_view(), name='token-status'),
    
    # История диалогов
    path('history/', ConversationHistoryAPIView.as_view(), name='history'),
]