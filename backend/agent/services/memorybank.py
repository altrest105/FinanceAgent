import logging
import re
import json
from django.conf import settings
from cryptography.fernet import Fernet
from asgiref.sync import sync_to_async

from ..models import InvestorProfile, Conversation, RiskProfile, InvestmentGoal

logger = logging.getLogger(__name__)

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
            for conv in reversed(conversations)
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
        
        conversation_count = await self.get_conversation_count()
        
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
        
        valid_fields = [
            'risk_tolerance', 'risk_score', 'primary_goal', 'secondary_goals',
            'investment_horizon_months', 'monthly_investment', 'total_capital',
            'experience_years', 'knowledge_level', 'preferences',
            'tinkoff_account_id'
        ]
        
        for key, value in kwargs.items():
            if key in valid_fields:
                if key == 'risk_tolerance' and value not in RiskProfile.values:
                    logger.warning(f"⚠️ Недопустимое значение risk_tolerance: {value}")
                    continue
                
                if key == 'primary_goal' and value not in InvestmentGoal.values:
                    logger.warning(f"⚠️ Недопустимое значение primary_goal: {value}")
                    continue
                
                if key == 'secondary_goals':
                    value = [g for g in value if g in InvestmentGoal.values]
                
                setattr(profile, key, value)
                logger.debug(f"  ├─ {key}: {value}")
        
        await sync_to_async(profile.save)()
        
        logger.info(f"✅ Профиль обновлён")


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
        r't\.[A-Za-z0-9_\-]{80,}',
        
        # Токен без префикса
        r'[A-Za-z0-9_\-]{100,}',
        
        # Токен с явным указанием "токен:" или "token:"
        r'(?:токен|token)[:=\s]+([A-ZaZ0-9._\-]{80,})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        
        if match:
            token = match.group(1) if match.groups() else match.group(0)
            token = token.strip()
            
            if len(token) >= 80:
                logger.info(f"✅ Токен найден (длина: {len(token)}, паттерн: {pattern})")
                return token
    
    logger.debug("⚠️ Токен не найден")
    return None


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


# ============================================================================
# АНАЛИЗ И ОБНОВЛЕНИЕ ПРОФИЛЯ
# ============================================================================

async def analyze_and_update_profile(memory_bank, client):
    """Автоматический анализ диалогов и обновление инвестиционного профиля"""
    logger.info("=" * 80)
    logger.info("🧠 АВТОМАТИЧЕСКИЙ АНАЛИЗ ИНВЕСТИЦИОННОГО ПРОФИЛЯ")
    logger.info("=" * 80)
    logger.info(f"  └─ User ID: {memory_bank.user_id}")
    
    try:
        logger.debug("  ⏳ Получение истории диалогов...")
        conversations = await memory_bank.get_conversation_history(limit=10)
        
        if not conversations:
            logger.warning("  ⚠️ Нет диалогов для анализа")
            return None
        
        logger.info(f"  ✓ Получено диалогов: {len(conversations)}")
        
        conversation_text = "\n\n".join([
            f"Диалог #{i+1}:\nВопрос: {conv['question']}\nОтвет: {conv['answer']}"
            for i, conv in enumerate(conversations)
        ])
        
        current_profile = await memory_bank.get_profile_summary()
        
        analysis_prompt = f"""Ты — финансовый аналитик, который анализирует диалоги с инвестором для определения его инвестиционного профиля.

**ТЕКУЩИЙ ПРОФИЛЬ:**
- Склонность к риску: {current_profile['risk_profile']['type']} ({current_profile['risk_profile']['type_key']})
- Риск-скор: {current_profile['risk_profile']['score']}/10
- Основная цель: {current_profile['goals']['primary']} ({current_profile['goals']['primary_key']})
- Горизонт инвестирования: {current_profile['financial_params']['investment_horizon_months']} месяцев
- Опыт инвестирования: {current_profile['experience']['years']} лет
- Уровень знаний: {current_profile['experience']['level']} ({current_profile['experience']['level_key']})

**ПОСЛЕДНИЕ 10 ДИАЛОГОВ:**
{conversation_text}

---

**ЗАДАЧА:**
Проанализируй диалоги и определи инвестиционный профиль пользователя. Верни JSON с обновлёнными параметрами.

**ДОСТУПНЫЕ ЗНАЧЕНИЯ:**

**risk_tolerance** (склонность к риску):
- "CONSERVATIVE" — консервативный (избегает рисков, предпочитает облигации)
- "MODERATE" — умеренный (баланс риска/доходности, смешанный портфель)
- "AGGRESSIVE" — агрессивный (готов к высоким рискам ради доходности)
- "UNKNOWN" — не определён

**primary_goal** (основная цель):
- "CAPITAL_PRESERVATION" — сохранение капитала
- "STABLE_INCOME" — стабильный доход
- "CAPITAL_GROWTH" — рост капитала
- "SPECULATION" — спекуляция
- "RETIREMENT" — пенсионные накопления
- "UNKNOWN" — не определена

**knowledge_level** (уровень знаний):
- "beginner" — начинающий
- "intermediate" — средний
- "advanced" — продвинутый
- "expert" — эксперт

**investment_horizon** (горизонт инвестирования):
- "short" — краткосрочный (до 1 года)
- "medium" — среднесрочный (1-3 года)
- "long" — долгосрочный (3+ года)

---

**ФОРМАТ ОТВЕТА (СТРОГО JSON):**
{{
  "risk_appetite": "MODERATE",
  "risk_score": 6,
  "primary_goal": "CAPITAL_GROWTH",
  "investment_horizon": "long",
  "investment_horizon_months": 36,
  "financial_literacy": "intermediate",
  "reasoning": "Пользователь задаёт вопросы о долгосрочных инвестициях в акции роста, демонстрирует понимание базовых финансовых терминов, готов к умеренным рискам."
}}

**ВАЖНО:**
- Используй ТОЛЬКО значения из списка выше
- Поле `risk_score` должно быть от 0 до 10 (0 = минимальный риск, 10 = максимальный)
- Поле `investment_horizon_months` должно соответствовать `investment_horizon`
- Поле `reasoning` должно содержать краткое обоснование (1-2 предложения)
"""

        logger.debug("  ⏳ Запрос к LLM для анализа профиля...")
        
        from ..orchestrator import call_llm
        response = call_llm(client, analysis_prompt)
        
        analysis_text = response.content[0].text.strip()
        logger.debug(f"  ✓ Получен ответ от LLM ({len(analysis_text)} символов)")
        
        if "```json" in analysis_text:
            analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
        elif "```" in analysis_text:
            analysis_text = analysis_text.split("```")[1].split("```")[0].strip()
        
        analysis_result = json.loads(analysis_text)
        
        logger.info("=" * 80)
        logger.info("📊 РЕЗУЛЬТАТ АНАЛИЗА ПРОФИЛЯ")
        logger.info("=" * 80)
        logger.info(f"  ├─ Склонность к риску: {analysis_result.get('risk_appetite')}")
        logger.info(f"  ├─ Риск-скор: {analysis_result.get('risk_score')}/10")
        logger.info(f"  ├─ Основная цель: {analysis_result.get('primary_goal')}")
        logger.info(f"  ├─ Горизонт: {analysis_result.get('investment_horizon')} ({analysis_result.get('investment_horizon_months')} мес.)")
        logger.info(f"  ├─ Уровень знаний: {analysis_result.get('financial_literacy')}")
        logger.info(f"  └─ Обоснование: {analysis_result.get('reasoning', 'N/A')[:100]}...")
        logger.info("=" * 80)
        
        logger.debug("  ⏳ Обновление профиля в базе данных...")
        
        update_data = {}
        
        if 'risk_appetite' in analysis_result and validate_risk_profile(analysis_result['risk_appetite']):
            update_data['risk_tolerance'] = analysis_result['risk_appetite']
        
        if 'risk_score' in analysis_result:
            update_data['risk_score'] = min(10, max(0, int(analysis_result['risk_score'])))
        
        if 'primary_goal' in analysis_result and validate_investment_goal(analysis_result['primary_goal']):
            update_data['primary_goal'] = analysis_result['primary_goal']
        
        if 'investment_horizon_months' in analysis_result:
            update_data['investment_horizon_months'] = int(analysis_result['investment_horizon_months'])
        
        if 'financial_literacy' in analysis_result:
            valid_levels = ['beginner', 'intermediate', 'advanced', 'expert']
            if analysis_result['financial_literacy'] in valid_levels:
                update_data['knowledge_level'] = analysis_result['financial_literacy']
        
        if 'reasoning' in analysis_result:
            current_prefs = current_profile.get('preferences', {})
            current_prefs['last_analysis_reasoning'] = analysis_result['reasoning']
            current_prefs['last_analysis_date'] = logger.handlers[0].formatter.formatTime(
                logger.makeRecord('', 0, '', 0, '', (), None)
            ) if logger.handlers else None
            update_data['preferences'] = current_prefs
        
        if update_data:
            await memory_bank.update_profile(**update_data)
            logger.info(f"  ✅ Профиль обновлён ({len(update_data)} полей)")
        else:
            logger.warning("  ⚠️ Нет данных для обновления")
        
        logger.info("=" * 80)
        logger.info("✅ АНАЛИЗ ПРОФИЛЯ ЗАВЕРШЁН")
        logger.info("=" * 80)
        
        return analysis_result
        
    except json.JSONDecodeError as e:
        logger.error("=" * 80)
        logger.error("❌ ОШИБКА ПАРСИНГА JSON")
        logger.error("=" * 80)
        logger.error(f"  ├─ Ошибка: {e}")
        logger.error(f"  └─ Ответ LLM: {analysis_text[:200]}...")
        logger.error("=" * 80)
        return None
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ ОШИБКА АНАЛИЗА ПРОФИЛЯ")
        logger.error("=" * 80)
        logger.error(f"  ├─ Тип ошибки: {type(e).__name__}")
        logger.error(f"  └─ Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        return None