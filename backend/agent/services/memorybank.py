import logging
import re
from django.conf import settings
from cryptography.fernet import Fernet
from asgiref.sync import sync_to_async

from ..models import InvestorProfile, Conversation, RiskProfile, InvestmentGoal

logger = logging.getLogger(__name__)

# Инициализация шифрования
ENCRYPTION_KEY = getattr(settings, 'ENCRYPTION_KEY', Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY)


# ============================================================================
# MEMORY BANK
# ============================================================================

class MemoryBank:
    """Банк памяти для хранения информации об инвесторе"""
    
    def __init__(self, user):
        self.user = user
        self.user_id = user.id if hasattr(user, 'id') else str(user)
        
        logger.info(f"💾 MemoryBank инициализирован для user_id={self.user_id}")
    
    # === ПОЛУЧЕНИЕ/СОЗДАНИЕ ПРОФИЛЯ ===
    
    async def get_or_create_profile(self):
        """Получение или создание профиля пользователя"""
        profile, created = await sync_to_async(InvestorProfile.objects.get_or_create)(
            user=self.user,
            defaults={
                'risk_tolerance': RiskProfile.UNKNOWN,
                'risk_score': 0,
                'primary_goal': InvestmentGoal.UNKNOWN,
                'secondary_goals': [],
                'investment_horizon_months': 12,
                'monthly_investment': 0,
                'total_capital': 0,
                'experience_years': 0,
                'knowledge_level': 'beginner',
                'preferences': {},
                'tinkoff_token_encrypted': None,
                'tinkoff_account_id': None,
            }
        )
        
        if created:
            logger.info(f"🆕 Создан новый профиль для user_id={self.user_id}")
        
        return profile
    
    # === РАБОТА С ТОКЕНОМ ===
    
    def encrypt_token(self, token):
        """Шифрование токена"""
        try:
            encrypted = cipher_suite.encrypt(token.encode()).decode()
            logger.info(f"🔐 Токен зашифрован (длина: {len(encrypted)})")
            return encrypted
        except Exception as e:
            logger.error(f"❌ Ошибка шифрования: {e}")
            return token
    
    def decrypt_token(self, encrypted_token):
        """Расшифровка токена"""
        try:
            decrypted = cipher_suite.decrypt(encrypted_token.encode()).decode()
            logger.info(f"🔓 Токен расшифрован")
            return decrypted
        except Exception as e:
            logger.error(f"❌ Ошибка расшифровки: {e}")
            return encrypted_token
    
    async def get_tinkoff_token(self):
        """Получение токена (расшифровка)"""
        logger.debug(f"🔑 Запрос Tinkoff токена для user_id={self.user_id}")
        
        profile = await self.get_or_create_profile()
        
        if not profile.tinkoff_token_encrypted:
            logger.warning(f"⚠️ Токен не найден для user_id={self.user_id}")
            return None
        
        decrypted = self.decrypt_token(profile.tinkoff_token_encrypted)
        logger.info(f"✅ Токен получен для user_id={self.user_id}")
        return decrypted
    
    async def save_tinkoff_token(self, token, account_id=None):
        """Сохранение токена (шифрование)"""
        logger.info(f"💳 Сохранение токена для user_id={self.user_id}")
        
        encrypted = self.encrypt_token(token)
        
        profile = await self.get_or_create_profile()
        profile.tinkoff_token_encrypted = encrypted
        
        if account_id:
            profile.tinkoff_account_id = account_id
        
        await sync_to_async(profile.save)()
        
        logger.info(f"✅ Токен сохранён для user_id={self.user_id}")
    
    def has_token(self):
        """Проверка наличия токена (синхронная)"""
        try:
            profile = InvestorProfile.objects.filter(user=self.user).first()
            has_token = bool(profile and profile.tinkoff_token_encrypted)
            logger.debug(f"🔍 Проверка токена для user_id={self.user_id}: {has_token}")
            return has_token
        except Exception as e:
            logger.error(f"❌ Ошибка проверки токена: {e}")
            return False
    
    # === УПРАВЛЕНИЕ ДИАЛОГАМИ ===
    
    async def save_conversation(
        self, 
        question, 
        answer, 
        intent = "",
        extracted_data = None,
        session_id = None
    ):
        """Сохранение диалога в базу данных"""
        logger.debug(f"💬 Сохранение диалога для user_id={self.user_id}")
        
        conversation = await sync_to_async(Conversation.objects.create)(
            user=self.user,
            question=question,
            answer=answer,
            intent=intent,
            extracted_data=extracted_data or {},
            session_id=session_id or ""
        )
        
        logger.info(f"✅ Диалог #{conversation.id} сохранён")
        return conversation
    
    async def get_conversation_history(
        self, 
        limit = 50,
        session_id = None
    ):
        """Получение истории диалогов"""
        logger.debug(f"📖 Получение истории диалогов (limit={limit})")
        
        queryset = Conversation.objects.filter(user=self.user)
        
        if session_id:
            queryset = queryset.filter(session_id=session_id)
        
        conversations = await sync_to_async(list)(
            queryset.order_by('-timestamp')[:limit]
        )
        
        result = [
            {
                'id': conv.id,
                'question': conv.question,
                'answer': conv.answer,
                'intent': conv.intent,
                'extracted_data': conv.extracted_data,
                'session_id': conv.session_id,
                'timestamp': conv.timestamp.isoformat()
            }
            for conv in reversed(conversations)  # Хронологический порядок
        ]
        
        logger.debug(f"✅ Получено {len(result)} диалогов")
        return result
    
    async def get_relevant_context(self, current_question, top_k = 5):
        """Получение релевантного контекста из истории диалогов"""
        logger.debug(f"🔍 Получение релевантного контекста (top_k={top_k})")
        
        recent_conversations = await self.get_conversation_history(limit=top_k)
        
        if not recent_conversations:
            logger.debug("⚠️ История пуста")
            return ""
        
        context = "\n\n".join([
            f"Ранее обсуждалось:\nВопрос: {conv['question']}\nОтвет: {conv['answer'][:300]}..."
            for conv in recent_conversations[-top_k:]
        ])
        
        logger.debug(f"✅ Контекст сформирован ({len(context)} символов)")
        return context
    
    async def get_conversation_count(self):
        """Получение общего количества диалогов"""
        count = await sync_to_async(
            Conversation.objects.filter(user=self.user).count
        )()
        
        logger.debug(f"📊 Всего диалогов для user_id={self.user_id}: {count}")
        return count
    
    # === ПОЛУЧЕНИЕ ПРОФИЛЯ ===
    
    async def get_profile_summary(self):
        """Получение сводки по профилю инвестора"""
        logger.debug(f"👤 Формирование сводки профиля для user_id={self.user_id}")
        
        profile = await self.get_or_create_profile()
        
        # Получаем количество диалогов
        conversation_count = await self.get_conversation_count()
        
        # Человекочитаемые названия из Enum
        risk_labels = {
            RiskProfile.CONSERVATIVE: 'Консервативный',
            RiskProfile.MODERATE: 'Умеренный',
            RiskProfile.AGGRESSIVE: 'Агрессивный',
            RiskProfile.UNKNOWN: 'Не определён'
        }
        
        goal_labels = {
            InvestmentGoal.CAPITAL_PRESERVATION: 'Сохранение капитала',
            InvestmentGoal.STABLE_INCOME: 'Стабильный доход',
            InvestmentGoal.CAPITAL_GROWTH: 'Рост капитала',
            InvestmentGoal.SPECULATION: 'Спекуляция',
            InvestmentGoal.RETIREMENT: 'Пенсионные накопления',
            InvestmentGoal.UNKNOWN: 'Не определена'
        }
        
        level_names = {
            'beginner': 'Начинающий',
            'intermediate': 'Средний',
            'advanced': 'Продвинутый',
            'expert': 'Эксперт'
        }
        
        summary = {
            'user_id': self.user_id,
            'risk_profile': {
                'type': risk_labels.get(profile.risk_tolerance, 'Не определён'),
                'type_key': profile.risk_tolerance,
                'score': profile.risk_score,
            },
            'goals': {
                'primary': goal_labels.get(profile.primary_goal, 'Не определена'),
                'primary_key': profile.primary_goal,
                'secondary': [
                    {'key': goal, 'label': goal_labels.get(goal, goal)}
                    for goal in profile.secondary_goals
                ],
            },
            'financial_params': {
                'investment_horizon_months': profile.investment_horizon_months,
                'monthly_investment': float(profile.monthly_investment),
                'total_capital': float(profile.total_capital),
            },
            'experience': {
                'years': profile.experience_years,
                'level': level_names.get(profile.knowledge_level, 'Начинающий'),
                'level_key': profile.knowledge_level,
            },
            'preferences': profile.preferences,
            'meta': {
                'has_tinkoff_token': bool(profile.tinkoff_token_encrypted),
                'tinkoff_account_id': profile.tinkoff_account_id,
                'total_conversations': conversation_count,
                'created_at': profile.created_at.isoformat(),
                'updated_at': profile.updated_at.isoformat(),
            }
        }
        
        logger.debug(f"✅ Сводка профиля сформирована")
        return summary
    
    # === ОБНОВЛЕНИЕ ПРОФИЛЯ ===
    
    async def update_profile(self, **kwargs):
        """Обновление произвольных полей профиля"""
        logger.info(f"🔄 Обновление профиля для user_id={self.user_id}")
        logger.debug(f"  └─ Поля: {list(kwargs.keys())}")
        
        profile = await self.get_or_create_profile()
        
        # Обновляем только валидные поля
        valid_fields = [
            'risk_tolerance', 'risk_score', 'primary_goal', 'secondary_goals',
            'investment_horizon_months', 'monthly_investment', 'total_capital',
            'experience_years', 'knowledge_level', 'preferences',
            'tinkoff_account_id'
        ]
        
        for key, value in kwargs.items():
            if key in valid_fields:
                # Валидация Enum полей
                if key == 'risk_tolerance' and value not in RiskProfile.values:
                    logger.warning(f"⚠️ Недопустимое значение risk_tolerance: {value}")
                    continue
                
                if key == 'primary_goal' and value not in InvestmentGoal.values:
                    logger.warning(f"⚠️ Недопустимое значение primary_goal: {value}")
                    continue
                
                if key == 'secondary_goals':
                    # Фильтруем только валидные цели
                    value = [g for g in value if g in InvestmentGoal.values]
                
                setattr(profile, key, value)
                logger.debug(f"  ├─ {key}: {value}")
        
        await sync_to_async(profile.save)()
        
        logger.info(f"✅ Профиль обновлён")
    
    # === СТАТИСТИКА ===
    
    async def get_stats(self):
        """Получение статистики по пользователю"""
        profile = await self.get_or_create_profile()
        conversation_count = await self.get_conversation_count()
        
        # Получаем количество диалогов по типам
        intent_stats = {}
        conversations = await sync_to_async(list)(
            Conversation.objects.filter(user=self.user).values('intent')
        )
        
        for conv in conversations:
            intent = conv['intent'] or 'unknown'
            intent_stats[intent] = intent_stats.get(intent, 0) + 1
        
        stats = {
            'total_conversations': conversation_count,
            'intent_distribution': intent_stats,
            'profile': {
                'risk_tolerance': profile.risk_tolerance,
                'risk_score': profile.risk_score,
                'primary_goal': profile.primary_goal,
                'has_token': bool(profile.tinkoff_token_encrypted),
            },
            'created_at': profile.created_at.isoformat(),
            'updated_at': profile.updated_at.isoformat(),
        }
        
        logger.debug(f"📊 Статистика для user_id={self.user_id}: {stats}")
        return stats


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

async def get_or_request_token(memory_bank):
    """Получение токена из памяти или возврат None"""
    logger.debug(f"🔐 Запрос токена для user_id={memory_bank.user_id}")
    
    token = await memory_bank.get_tinkoff_token()
    
    if not token:
        logger.warning(f"⚠️ Токен не найден для user_id={memory_bank.user_id}")
        return None
    
    logger.info(f"✅ Токен получен для user_id={memory_bank.user_id}")
    return token


async def extract_token_from_message(message):
    """Извлечение токена Tinkoff из сообщения пользователя"""
    logger.debug(f"🔍 Поиск токена в сообщении (длина: {len(message)})")
    
    # Паттерны для токенов Tinkoff
    patterns = [
        # Tinkoff токен с префиксом t. (основной паттерн)
        r't\.[A-Za-z0-9_\-]{80,}',
        
        # Токен без префикса (на случай если пользователь скопировал часть)
        r'[A-Za-z0-9_\-]{100,}',
        
        # Токен с явным указанием "токен:" или "token:"
        r'(?:токен|token)[:=\s]+([A-ZaZ0-9._\-]{80,})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        
        if match:
            token = match.group(1) if match.groups() else match.group(0)
            token = token.strip()
            
            # Tinkoff токены обычно длиной 100-150 символов
            if len(token) >= 80:
                logger.info(f"✅ Токен найден (длина: {len(token)}, паттерн: {pattern})")
                return token
    
    logger.debug("⚠️ Токен не найден")
    return None


async def clear_user_data(user):
    """Очистка всех данных пользователя (для тестирования)"""
    logger.warning(f"🗑️ Очистка данных для user_id={user.id}")
    
    # Удаляем профиль
    deleted_profiles = await sync_to_async(
        InvestorProfile.objects.filter(user=user).delete
    )()
    logger.debug(f"  ├─ Удалено профилей: {deleted_profiles[0]}")
    
    # Удаляем диалоги
    deleted_conversations = await sync_to_async(
        Conversation.objects.filter(user=user).delete
    )()
    logger.debug(f"  └─ Удалено диалогов: {deleted_conversations[0]}")
    
    logger.info(f"✅ Данные пользователя {user.username} очищены")


async def get_global_stats():
    """Получение глобальной статистики по всем пользователям"""
    logger.debug("📊 Получение глобальной статистики...")
    
    total_users = await sync_to_async(InvestorProfile.objects.count)()
    total_conversations = await sync_to_async(Conversation.objects.count)()
    users_with_tokens = await sync_to_async(
        InvestorProfile.objects.filter(
            tinkoff_token_encrypted__isnull=False
        ).count
    )()
    
    # Распределение по риск-профилям
    risk_distribution = {}
    profiles = await sync_to_async(list)(
        InvestorProfile.objects.values('risk_tolerance')
    )
    
    for profile in profiles:
        risk = profile['risk_tolerance']
        risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
    
    # Распределение по целям
    goal_distribution = {}
    goal_profiles = await sync_to_async(list)(
        InvestorProfile.objects.values('primary_goal')
    )
    
    for profile in goal_profiles:
        goal = profile['primary_goal']
        goal_distribution[goal] = goal_distribution.get(goal, 0) + 1
    
    stats = {
        'total_users': total_users,
        'total_conversations': total_conversations,
        'users_with_tokens': users_with_tokens,
        'risk_distribution': risk_distribution,
        'goal_distribution': goal_distribution,
    }
    
    logger.debug(f"✅ Глобальная статистика: {stats}")
    return stats


# ============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ С ENUM
# ============================================================================

def get_risk_profile_choices():
    """Получение списка доступных риск-профилей"""
    return [
        {'key': RiskProfile.CONSERVATIVE, 'label': 'Консервативный'},
        {'key': RiskProfile.MODERATE, 'label': 'Умеренный'},
        {'key': RiskProfile.AGGRESSIVE, 'label': 'Агрессивный'},
        {'key': RiskProfile.UNKNOWN, 'label': 'Не определён'},
    ]


def get_investment_goal_choices():
    """Получение списка доступных инвестиционных целей"""
    return [
        {'key': InvestmentGoal.CAPITAL_PRESERVATION, 'label': 'Сохранение капитала'},
        {'key': InvestmentGoal.STABLE_INCOME, 'label': 'Стабильный доход'},
        {'key': InvestmentGoal.CAPITAL_GROWTH, 'label': 'Рост капитала'},
        {'key': InvestmentGoal.SPECULATION, 'label': 'Спекуляция'},
        {'key': InvestmentGoal.RETIREMENT, 'label': 'Пенсионные накопления'},
        {'key': InvestmentGoal.UNKNOWN, 'label': 'Не определена'},
    ]


def validate_risk_profile(value):
    """Проверка валидности риск-профиля"""
    return value in RiskProfile.values


def validate_investment_goal(value):
    """Проверка валидности инвестиционной цели"""
    return value in InvestmentGoal.values