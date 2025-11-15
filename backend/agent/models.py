from django.db import models
from django.contrib.auth.models import User

class RiskProfile(models.TextChoices):
    """Типы риск-профилей инвестора"""
    CONSERVATIVE = 'conservative', 'Консервативный'
    MODERATE = 'moderate', 'Умеренный'
    AGGRESSIVE = 'aggressive', 'Агрессивный'
    UNKNOWN = 'unknown', 'Не определён'

class InvestmentGoal(models.TextChoices):
    """Инвестиционные цели"""
    CAPITAL_PRESERVATION = 'capital_preservation', 'Сохранение капитала'
    STABLE_INCOME = 'stable_income', 'Стабильный доход'
    CAPITAL_GROWTH = 'capital_growth', 'Рост капитала'
    SPECULATION = 'speculation', 'Спекуляция'
    RETIREMENT = 'retirement', 'Пенсионные накопления'
    UNKNOWN = 'unknown', 'Не определена'

class Conversation(models.Model):
    """История разговоров с пользователем"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    session_id = models.CharField(max_length=100, db_index=True)
    question = models.TextField()
    answer = models.TextField()
    intent = models.CharField(max_length=50)
    extracted_data = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['session_id']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.question[:50]} ({self.timestamp})"

class InvestorProfile(models.Model):
    """Профиль инвестора"""
    RISK_CHOICES = [
        ('conservative', 'Консервативный'),
        ('moderate', 'Умеренный'),
        ('aggressive', 'Агрессивный'),
        ('unknown', 'Не определён'),
    ]
    
    GOAL_CHOICES = [
        ('capital_preservation', 'Сохранение капитала'),
        ('stable_income', 'Стабильный доход'),
        ('capital_growth', 'Рост капитала'),
        ('speculation', 'Спекуляция'),
        ('retirement', 'Пенсионные накопления'),
        ('unknown', 'Не определена'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='investor_profile')
    
    # Риск-профиль
    risk_tolerance = models.CharField(max_length=20, choices=RISK_CHOICES, default='unknown')
    risk_score = models.IntegerField(default=0)  # 0-100
    
    # Цели
    primary_goal = models.CharField(max_length=50, choices=GOAL_CHOICES, default='unknown')
    secondary_goals = models.JSONField(default=list, blank=True)
    
    # Финансовые параметры
    investment_horizon_months = models.IntegerField(default=12)
    monthly_investment = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_capital = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    
    # Опыт
    experience_years = models.IntegerField(default=0)
    knowledge_level = models.CharField(max_length=20, default='beginner')
    
    # Предпочтения
    preferences = models.JSONField(default=dict, blank=True)
    
    # Tinkoff данные (зашифрованные)
    tinkoff_token_encrypted = models.TextField(null=True, blank=True)
    tinkoff_account_id = models.CharField(max_length=100, null=True, blank=True)
    
    # Метаданные
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Профиль инвестора"
        verbose_name_plural = "Профили инвесторов"
    
    def __str__(self):
        return f"{self.user.username} - {self.get_risk_tolerance_display()}"
    
    def update_completeness(self):
        """Обновление процента заполненности профиля"""
        fields = [
            self.tinkoff_token_encrypted,
            self.risk_tolerance != 'unknown',
            self.primary_goal != 'unknown',
            self.investment_horizon_months > 0,
            self.experience_years >= 0,
            self.knowledge_level
        ]
        self.profile_completeness = int((sum(1 for f in fields if f) / len(fields)) * 100)
        self.save(update_fields=['profile_completeness'])
