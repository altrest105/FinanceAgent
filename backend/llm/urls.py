from django.urls import path
from .views import InvestAIAgentAPIView

urlpatterns = [
    path('ask/', InvestAIAgentAPIView.as_view(), name='llm-ask'),
]