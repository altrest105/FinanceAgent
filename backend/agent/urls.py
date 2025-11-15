from django.urls import path
from .views import AgentOrchestratorAPIView

urlpatterns = [
    path('ask/', AgentOrchestratorAPIView.as_view(), name='agent_ask'),
]