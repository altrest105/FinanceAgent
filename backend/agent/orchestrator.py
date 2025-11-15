import logging
import tiktoken
import asyncio
import json
from enum import Enum
from openai import AsyncOpenAI, RateLimitError, APIError
from django.conf import settings

# Импорт сервисов
from .services.rag import get_rag_context
from .services.tinvest import (
    search_instrument, 
    place_order, 
    get_token_from_settings,
    get_portfolio_summary,
    get_accounts
)
from .services.memorybank import MemoryBank, get_or_request_token, extract_token_from_message
from .apps import is_rag_ready, get_retriever, get_llm_client

logger = logging.getLogger(__name__)

ENCODING = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = settings.LLM_CONFIG.get("max_tokens")


# ============================================================================
# НАМЕРЕНИЯ ПОЛЬЗОВАТЕЛЯ (3 ОСНОВНЫХ СЦЕНАРИЯ)
# ============================================================================

class IntentType(Enum):
    """Типы намерений пользователя"""
    TRADE = "trade"                      # 1️⃣ Купить/продать ценные бумаги
    SECURITIES_INFO = "securities_info"  # 2️⃣ Вопрос ПРО ценные бумаги (новости, аналитика)
    GENERAL = "general"                  # 3️⃣ Вопрос НЕ про ценные бумаги (финансы, право, услуги)


class TradeAction(Enum):
    """Типы торговых операций"""
    BUY = "buy"
    SELL = "sell"


# ============================================================================
# ПРОМПТЫ ДЛЯ КЛАССИФИКАЦИИ И ИЗВЛЕЧЕНИЯ
# ============================================================================

def build_intent_classification_prompt(question):
    """Промпт для определения намерения пользователя (3 сценария)"""
    return f"""Определи намерение пользователя на основе его вопроса.

Возможные намерения:
1. TRADE - пользователь хочет КУПИТЬ или ПРОДАТЬ ценные бумаги (явное действие: "купи", "продай", "размести заявку")
2. SECURITIES_INFO - вопрос ПРО конкретные ценные бумаги (новости, котировки, аналитика, компании, вопросы про инвестиционный портфель пользователя)
3. GENERAL - общие вопросы НЕ про ценные бумаги (финансы, право, налоги, услуги)

Примеры TRADE:
- "Купи 10 акций Яндекса" -> TRADE
- "Продай Сбербанк по 280 рублей" -> TRADE
- "Размести заявку на покупку Газпрома" -> TRADE
- "Давай продадим LKOH" -> TRADE

Примеры SECURITIES_INFO:
- "Какие новости по Сбербанку?" -> SECURITIES_INFO
- "Что происходит с акциями GAZP?" -> SECURITIES_INFO
- "Какая цена акций Яндекса?" -> SECURITIES_INFO
- "Анализ компании Apple" -> SECURITIES_INFO
- "Почему упал LKOH?" -> SECURITIES_INFO


Примеры GENERAL:
- "Как взять кредит?" -> GENERAL
- "Что такое налог на доходы физических лиц?" -> GENERAL
- "Какие приемы помогут сократить импульсные покупки?" -> GENERAL
- "Как безопасно настроить автоплатеж?"
- "Как открыть ИИС?" -> GENERAL
- "Что такое облигации?" -> GENERAL

Вопрос: {question}

Ответь ТОЛЬКО одним словом: TRADE, SECURITIES_INFO или GENERAL"""


def build_trade_extraction_prompt(question):
    """Промпт для извлечения параметров торговой операции"""
    return f"""Извлеки параметры торговой операции из запроса пользователя.

Вопрос: {question}

Верни ответ СТРОГО в формате JSON:
{{
    "action": "buy" или "sell",
    "ticker": "тикер акции (SBER, GAZP, YNDX) или название компании",
    "quantity": число акций (целое число),
    "price": цена за акцию (число или null, если не указана)
}}

Если какой-то параметр не указан явно, уточни его у пользователя.

Примеры:
- "Купи 10 Сбербанка" -> {{"action": "buy", "ticker": "SBER", "quantity": 10, "price": null}}
- "Продай 5 акций Газпрома по 170 рублей" -> {{"action": "sell", "ticker": "GAZP", "quantity": 5, "price": 170}}"""


def build_securities_extraction_prompt(question):
    """Промпт для извлечения информации о ценных бумагах"""
    return f"""Извлеки все упомянутые тикеры или названия ценных бумаг из вопроса пользователя.

Вопрос: {question}

Верни ответ СТРОГО в формате JSON:
{{
    "tickers": ["SBER", "GAZP"],  // массив тикеров
    "companies": ["Сбербанк", "Газпром"]  // массив названий компаний
}}

Если ничего не найдено, верни пустые массивы.

Примеры:
- "Новости по Сбербанку и Газпрому" -> {{"tickers": ["SBER", "GAZP"], "companies": ["Сбербанк", "Газпром"]}}
- "Что с Apple?" -> {{"tickers": ["AAPL"], "companies": ["Apple"]}}
- "Общая ситуация на рынке" -> {{"tickers": [], "companies": []}}"""


# ============================================================================
# ПРОМПТЫ ДЛЯ ГЕНЕРАЦИИ ОТВЕТОВ (3 СЦЕНАРИЯ)
# ============================================================================

def build_general_answer_prompt(question, rag_context=""):
    """Промпт для ответа на ОБЩИЕ вопросы (НЕ про ценные бумаги)"""
    return f"""Ты — русскоязычный ИИ-ассистент, эксперт в области финансов, инвестиций, права и государственных услуг.

Твоя задача — дать подробный, структурированный и максимально полезный ответ на вопрос пользователя. Ты должен основываться на предоставленном тебе контексте.

Правила твоего ответа:
- Главный заголовок: Начинай ответ с основного заголовка, используя ###.
- Подзаголовки: Используй ####. ОБЯЗАТЕЛЬНО нумеруй логические разделы, начиная с 1. (например: #### 1. Название подраздела). Ключевые слова в подзаголовке выделяй жирным шрифтом.
- Структура: Если вопрос клиента состоит из нескольких явно выраженных аспектов (например, "Влияние на А и Б"), КАЖДЫЙ АСПЕКТ должен быть раскрыт в ОТДЕЛЬНОМ НУМЕРОВАННОМ ПОДЗАГОЛКЕ. (Пример: #### 1. Влияние на А, #### 2. Влияние на Б).
- Отступы: После каждого заголовка (### и ####) и между логическими блоками (например, после параграфа и перед списком) ВСЕГДА оставляй одну пустую строку для отступа.
- Маркированные списки: Используй дефис ( - ...) для перечислений, где порядок не важен.
- Нумерованные списки: Используй нумерацию 1. 2. 3. только для перечислений, где порядок имеет значение.
- Таблицы: Для вопросов, требующих сравнения или структурированного перечисления характеристик, ВСЕГДА используй таблицу в формате Markdown для наглядности.
- Выделение: Выделяй ключевые термины, определения, законы и важные цифры жирным шрифтом (**...**).
- Синтез и Полнота: Не копируй текст из контекста слово в слово. Твоя цель — синтезировать информацию в единый, связный и легко читаемый ответ. КРИТИЧЕСКИ ВАЖНО: Дважды проверь, что твой ответ полностью покрывает все аспекты и ИСПОЛЬЗУЕТ ВСЮ РЕЛЕВАНТНУЮ ИНФОРМАЦИЮ из контекста для максимальной полноты.
- Структурный вывод: Если уместно, завершай ответ практическим советом или кратким выводом, используя подзаголовок без нумерации (например, #### Рекомендации или #### Полезные советы).
- Если ответа нет, постарайся ответить сам. Если и в таком случае недостаточно информации для ответа на вопрос, вежливо сообщи: "Извините, но сейчас я не могу дать ответ на ваш вопрос."
- СОХРАНЯЙ все переносы строк и пустые строки, чтобы обеспечить требуемую Markdown-структуру (заголовки, списки, таблицы) в ответе.

Контекст:
{rag_context}

Вопрос клиента: {question}
"""


def build_securities_answer_prompt(question, news_context=""):
    """Промпт для ответа на вопросы ПРО ценные бумаги"""
    return f"""Ты — русскоязычный ИИ-ассистент, эксперт в области ценных бумаг, фондового рынка и финансовой аналитики.

Твоя задача — дать подробный, структурированный ответ на вопрос пользователя о ценных бумагах, основываясь на предоставленных новостях и данных.

Правила твоего ответа:
- Главный заголовок: Начинай ответ с основного заголовка, используя ###.
- Подзаголовки: Используй ####. ОБЯЗАТЕЛЬНО нумеруй логические разделы (#### 1. **Компания А**, #### 2. **Компания Б**).
- Актуальность: ОБЯЗАТЕЛЬНО упоминай даты новостей и событий.
- Источники: Если в новостях есть конкретные данные (цены, проценты), указывай их.
- Таблицы: Для сравнения котировок используй таблицы в формате Markdown.
- Выделение: Выделяй важные цифры, даты, изменения цен жирным шрифтом (**...**).
- Объективность: Представляй разные точки зрения, если они есть в новостях.
- Если новостей недостаточно, честно сообщи об этом и предложи, что ещё ты можешь сделать.

Новости и данные о портфеле:
{news_context}

Вопрос пользователя: {question}

Дай структурированный ответ с акцентом на актуальность и конкретные данные."""


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

async def call_llm(client, prompt, max_tokens=MAX_TOKENS):
    """Универсальная функция для вызова LLM с логированием"""
    logger.info("🤖 Вызов LLM...")
    logger.debug(f"  ├─ Промпт ({len(prompt)} символов): {prompt[:200]}...")
    logger.debug(f"  └─ max_tokens: {max_tokens}")
    
    if len(ENCODING.encode(prompt)) > max_tokens:
        logger.error("❌ Промпт слишком длинный!")
        raise ValueError("Промпт слишком длинный")

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"  ⏳ Попытка {attempt + 1}/{MAX_RETRIES}...")
            
            response = await client.chat.completions.create(
                model=settings.LLM_CONFIG.get("model"),
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                max_tokens=max_tokens
            )

            if response and response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info(f"  ✅ LLM ответил ({len(result)} символов)")
                logger.debug(f"  └─ Ответ: {result[:200]}...")
                return result
            
            logger.warning("  ⚠️ LLM вернул пустой ответ")
            return None

        except (RateLimitError, APIError) as e:
            logger.warning(f"  ⚠️ LLM API Error/Rate Limit (попытка {attempt+1}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(5)
            continue

        except Exception as e:
            logger.error(f"  ❌ Непредвиденная ошибка при вызове LLM: {e}")
            break

    logger.error("❌ Не удалось получить ответ от LLM после всех попыток")
    return None


async def classify_intent(question, client):
    """Определение намерения пользователя (3 сценария)"""
    logger.info("🎯 Классификация намерения...")
    
    prompt = build_intent_classification_prompt(question)
    response = await call_llm(client, prompt)
    
    if not response:
        logger.warning("⚠️ Не удалось классифицировать намерение, используем GENERAL")
        return IntentType.GENERAL
    
    response_upper = response.upper().strip()
    logger.info(f"  └─ Сырой ответ классификатора: '{response_upper}'")
    
    if "TRADE" in response_upper:
        intent = IntentType.TRADE
    elif "SECURITIES_INFO" in response_upper or "SECURITIES" in response_upper:
        intent = IntentType.SECURITIES_INFO
    else:
        intent = IntentType.GENERAL
    
    logger.info(f"✅ Определено намерение: {intent.value}")
    return intent


async def extract_trade_params(question, client):
    """Извлечение параметров торговой операции"""
    logger.info("💰 Извлечение параметров торговой операции...")
    
    prompt = build_trade_extraction_prompt(question)
    response = await call_llm(client, prompt)
    
    if not response:
        logger.warning("⚠️ Не удалось извлечь параметры торговли")
        return None
    
    try:
        # Очистка JSON из Markdown
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        params = json.loads(response)
        logger.info(f"✅ Параметры торговли:")
        logger.info(f"  ├─ Действие: {params.get('action')}")
        logger.info(f"  ├─ Тикер: {params.get('ticker')}")
        logger.info(f"  ├─ Количество: {params.get('quantity')}")
        logger.info(f"  └─ Цена: {params.get('price', 'рыночная')}")
        return params
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Ошибка парсинга JSON параметров торговли: {e}")
        logger.debug(f"  └─ Сырой ответ: {response}")
        return None


async def extract_securities_info(question, client):
    """Извлечение информации о ценных бумагах из вопроса"""
    logger.info("📊 Извлечение информации о ценных бумагах...")
    
    prompt = build_securities_extraction_prompt(question)
    response = await call_llm(client, prompt)
    
    if not response:
        logger.warning("⚠️ Не удалось извлечь информацию о ценных бумагах")
        return None
    
    try:
        # Очистка JSON из Markdown
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        info = json.loads(response)
        logger.info(f"✅ Извлечено:")
        logger.info(f"  ├─ Тикеры: {info.get('tickers', [])}")
        logger.info(f"  └─ Компании: {info.get('companies', [])}")
        return info
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Ошибка парсинга JSON информации о ценных бумагах: {e}")
        logger.debug(f"  └─ Сырой ответ: {response}")
        return None


async def fetch_news_for_securities(securities_info):
    """
    Получение новостей для ценных бумаг
    TODO: Интеграция с новостным API
    """
    logger.info(f"📰 Получение новостей для ЦБ...")
    logger.debug(f"  └─ Securities info: {securities_info}")
    
    tickers = securities_info.get("tickers", [])
    companies = securities_info.get("companies", [])
    
    logger.warning("⚠️ Используются MOCK данные новостей (требуется реальная интеграция)")
    
    # TODO: Здесь должна быть реальная интеграция с API новостей
    mock_news = f"""
📰 Последние новости по запрошенным компаниям ({', '.join(tickers if tickers else companies)}):

1. **Сбербанк (SBER)** - 15.11.2024
   - Цена закрытия: **285.50 руб** (+2.3%)
   - Объявлены квартальные результаты, прибыль выросла на 15%
   - Объём торгов: 12.3 млрд руб
   
2. **Газпром (GAZP)** - 15.11.2024
   - Цена закрытия: **168.20 руб** (-0.8%)
   - Компания объявила о новых контрактах на поставку газа
   - Дивиденды: планируется выплата 12 руб на акцию

⚠️ Данные приведены для примера. Требуется интеграция с реальным источником новостей.
"""
    
    return mock_news


async def execute_trade(params, token):
    """Выполнение торговой операции через Tinkoff API"""
    logger.info("💳 Выполнение торговой операции...")
    logger.info(f"  └─ Параметры: {params}")
    
    action = params.get("action")
    ticker = params.get("ticker")
    quantity = params.get("quantity")
    price = params.get("price")
    
    if not all([action, ticker, quantity]):
        logger.error("❌ Недостаточно параметров для торговли")
        return {
            "success": False,
            "message": "❌ Недостаточно параметров для выполнения операции.\n\n**Укажите:**\n- Действие (купить/продать)\n- Тикер или название акции\n- Количество акций\n\n*Пример: Купи 10 акций Сбербанка по 280 рублей*"
        }
    
    try:
        # Получаем первый счет
        logger.info("  ⏳ Получение списка счетов...")
        accounts = await get_accounts(token)
        if not accounts:
            logger.error("  ❌ Не найдено активных счетов")
            return {
                "success": False,
                "message": "❌ Не найдено активных счетов Tinkoff"
            }
        account_id = accounts[0]["id"]
        logger.info(f"  ✓ Используется счёт: {account_id}")
        
        # Ищем инструмент по тикеру
        logger.info(f"  ⏳ Поиск инструмента {ticker}...")
        instruments = await search_instrument(token, ticker)
        if not instruments:
            logger.error(f"  ❌ Инструмент {ticker} не найден")
            return {
                "success": False,
                "message": f"❌ Инструмент с тикером **{ticker}** не найден.\n\nПроверьте правильность написания тикера."
            }
        
        figi = instruments[0]["figi"]
        instrument_name = instruments[0].get("name", ticker)
        logger.info(f"  ✓ Найден: {instrument_name} (FIGI: {figi})")
        
        # Размещаем заявку
        order_type = "limit" if price else "market"
        logger.info(f"  ⏳ Размещение {order_type} заявки...")
        
        result = await place_order(
            token=token,
            account_id=account_id,
            figi=figi,
            quantity=quantity,
            direction=action,
            order_type=order_type,
            price=price
        )
        
        logger.info("✅ Заявка успешно размещена!")
        logger.info(f"  └─ Order ID: {result.get('order_id')}")
        
        price_text = f"по цене **{price} руб**" if price else "**по рыночной цене**"
        action_text = "покупку" if action == "buy" else "продажу"
        
        return {
            "success": True,
            "message": f"""✅ **Заявка успешно размещена!**

📋 **Детали операции:**
- Действие: {action_text.capitalize()}
- Инструмент: **{instrument_name}** ({ticker})
- Количество: **{quantity}** шт.
- Цена: {price_text}
- ID заявки: `{result.get('order_id')}`

Заявка отправлена в биржу и будет исполнена согласно условиям рынка.""",
            "order_id": result.get("order_id"),
            "params": params,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка при выполнении торговой операции: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"❌ **Ошибка при размещении заявки**\n\n```\n{str(e)}\n```\n\nПопробуйте ещё раз или обратитесь в поддержку."
        }


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ-ОРКЕСТРАТОР (3 СЦЕНАРИЯ)
# ============================================================================

async def process_question(question, user, account_id=None, session_id=None):
    """
    Главная функция-оркестратор для обработки вопроса пользователя
    
    3 ОСНОВНЫХ СЦЕНАРИЯ:
    1️⃣ TRADE - Купить/продать ценные бумаги
    2️⃣ SECURITIES_INFO - Вопрос ПРО ценные бумаги (новости, котировки)
    3️⃣ GENERAL - Вопрос НЕ про ценные бумаги (финансы, право, услуги)
    """
    logger.info("=" * 80)
    logger.info("🎬 ЗАПУСК ОРКЕСТРАТОРА")
    logger.info("=" * 80)
    logger.info(f"  ├─ Пользователь: {user.username}")
    logger.info(f"  ├─ Вопрос: {question[:100]}...")
    logger.info(f"  └─ Session ID: {session_id or 'N/A'}")
    
    # Проверяем готовность RAG системы
    if not is_rag_ready():
        logger.error("❌ RAG система не готова!")
        return {
            "intent": "error",
            "success": False,
            "answer": "Извините, система временно недоступна. Попробуйте позже."
        }
    
    # Инициализируем MemoryBank
    logger.info("💾 Инициализация MemoryBank...")
    memory = MemoryBank(user)
    logger.info(f"  └─ MemoryBank готов для {user.username}")
    
    # Получаем LLM клиент из AppConfig
    client = get_llm_client()
    logger.info("🔌 LLM клиент получен из AppConfig")
    
    try:
        # 🔑 ПРОВЕРКА НА ТОКЕН В СООБЩЕНИИ
        logger.info("🔍 Проверка на наличие токена в сообщении...")
        potential_token = await extract_token_from_message(question)
        
        if potential_token:
            logger.info("🔑 Обнаружен токен Tinkoff в сообщении!")
            logger.info(f"  └─ Токен: t.***{potential_token[-20:]}")
            
            await memory.save_tinkoff_token(potential_token)
            logger.info("✅ Токен успешно сохранён в MemoryBank")
            
            return {
                "intent": "system",
                "success": True,
                "answer": "✅ **Токен Tinkoff успешно сохранён!**\n\nТеперь вы можете:\n- Покупать и продавать ценные бумаги\n- Получать информацию о портфеле\n- Получать персонализированные рекомендации\n\nПопробуйте: *'Купи 10 акций Сбербанка'*"
            }
        
        logger.info("  └─ Токен не обнаружен, продолжаем обработку вопроса")
        
        # 1️⃣ ОПРЕДЕЛЯЕМ НАМЕРЕНИЕ (3 СЦЕНАРИЯ)
        intent = await classify_intent(question, client)
        
        # Получаем контекст из истории
        logger.info("📜 Получение контекста из истории диалогов...")
        conversation_context = await memory.get_relevant_context(question, top_k=3)
        logger.info(f"  └─ Контекст: {len(conversation_context)} символов")
        
        
        # ====================================================================
        # СЦЕНАРИЙ 1️⃣: ТОРГОВАЯ ОПЕРАЦИЯ (TRADE)
        # ====================================================================
        if intent == IntentType.TRADE:
            logger.info("=" * 80)
            logger.info("1️⃣ СЦЕНАРИЙ: ТОРГОВАЯ ОПЕРАЦИЯ")
            logger.info("=" * 80)
            
            # Извлекаем параметры торговли
            params = await extract_trade_params(question, client)
            
            if not params:
                logger.warning("⚠️ Не удалось извлечь параметры торговли")
                answer = "❌ Не удалось определить параметры торговой операции.\n\n**Пожалуйста, уточните:**\n- Какую акцию (тикер или название)?\n- Сколько акций?\n- По какой цене (или по рыночной)?\n\n*Пример: Купи 10 акций Сбербанка по 280 рублей*"
                await memory.save_conversation(question, answer, intent.value, session_id=session_id)
                return {"intent": intent.value, "success": False, "answer": answer}
            
            # Проверяем наличие токена
            logger.info("🔐 Проверка наличия токена для торговли...")
            token = await get_or_request_token(memory)
            
            if not token:
                logger.warning("⚠️ Токен не найден в MemoryBank")
                answer = "❌ **Для совершения торговых операций необходим API токен Tinkoff.**\n\n**Как получить токен:**\n1. Перейдите в [настройки профиля Т-Инвестиции](https://www.tbank.ru/invest/settings/) и авторизуйтесь в системе, если это требуется.\nФункция **Подтверждение сделок кодом** должна быть отключена\n2. Выпустите токен T-Invest API для биржи **с полным доступом**\n3. Скопируйте и отправьте мне токен в чат (он будет надёжно зашифрован)\n\n*Токен нужно отправить только один раз - я запомню его для всех операций.*"
                await memory.save_conversation(question, answer, intent.value, session_id=session_id)
                return {"intent": intent.value, "success": False, "answer": answer}
            
            logger.info("  ✓ Токен найден, выполняем торговлю")
            
            # Выполняем операцию
            trade_result = await execute_trade(params, token)
            
            await memory.save_conversation(
                question=question,
                answer=trade_result["message"],
                intent=intent.value,
                extracted_data={"trade_params": params, "trade_result": trade_result},
                session_id=session_id
            )
            
            logger.info("=" * 80)
            logger.info("✅ ТОРГОВАЯ ОПЕРАЦИЯ ЗАВЕРШЕНА")
            logger.info("=" * 80)
            
            return {
                "intent": intent.value,
                "success": trade_result["success"],
                "trade_result": trade_result,
                "answer": trade_result["message"]
            }
        
        
        # ====================================================================
        # СЦЕНАРИЙ 2️⃣: ВОПРОС ПРО ЦЕННЫЕ БУМАГИ (SECURITIES_INFO)
        # ====================================================================
        elif intent == IntentType.SECURITIES_INFO:
            logger.info("=" * 80)
            logger.info("2️⃣ СЦЕНАРИЙ: ВОПРОС ПРО ЦЕННЫЕ БУМАГИ")
            logger.info("=" * 80)
            
            # Извлекаем информацию о ценных бумагах
            securities_info = await extract_securities_info(question, client)
            
            # Получаем новости
            news_context = await fetch_news_for_securities(securities_info or {})
            
            # Опционально: добавляем данные портфеля, если есть токен
            portfolio_info = ""
            token = await get_or_request_token(memory)
            if token:
                try:
                    accounts = await get_accounts(token)
                    if accounts:
                        portfolio = await get_portfolio_summary(token, accounts[0]["id"])
                        portfolio_info = f"\n\nДанные портфеля пользователя:\n{json.dumps(portfolio, ensure_ascii=False, indent=2)}"
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось получить портфель: {e}")
            
            # Генерируем ответ с учётом портфеля
            full_context = news_context + portfolio_info
            prompt = build_securities_answer_prompt(question, full_context)
            answer = await call_llm(client, prompt)
            
            if not answer:
                logger.error("❌ Ошибка генерации ответа о ценных бумагах")
                answer = "Извините, произошла ошибка при генерации ответа о ценных бумагах."
            
            await memory.save_conversation(
                question=question,
                answer=answer,
                intent=intent.value,
                extracted_data={"securities_info": securities_info},
                session_id=session_id
            )
            
            logger.info("=" * 80)
            logger.info("✅ ВОПРОС О ЦЕННЫХ БУМАГАХ ОБРАБОТАН")
            logger.info("=" * 80)
            
            return {
                "intent": intent.value,
                "success": True,
                "securities_info": securities_info,
                "answer": answer.strip()
            }
        
        
        # ====================================================================
        # СЦЕНАРИЙ 3️⃣: ОБЩИЙ ВОПРОС НЕ ПРО ЦЕННЫЕ БУМАГИ (GENERAL)
        # ====================================================================
        else:
            logger.info("=" * 80)
            logger.info("3️⃣ СЦЕНАРИЙ: ОБЩИЙ ФИНАНСОВЫЙ ВОПРОС (RAG)")
            logger.info("=" * 80)
            
            # Получаем контекст из RAG
            logger.info("🔍 Получение контекста из RAG...")
            try:
                rag_context = await get_rag_context(question, top_k=3)
                logger.info(f"  ✓ RAG контекст: {len(rag_context)} символов")
            except Exception as e:
                logger.error(f"  ❌ Ошибка получения контекста из RAG: {e}")
                rag_context = ""
            
            # Добавляем контекст из истории
            if conversation_context:
                logger.info("  ✓ Добавление контекста из истории диалогов")
                rag_context = f"{conversation_context}\n\n---\n\n{rag_context}"
            
            # Генерируем ответ
            prompt = build_general_answer_prompt(question, rag_context)
            answer = await call_llm(client, prompt)
            
            if not answer:
                logger.error("❌ Ошибка генерации ответа")
                answer = "Извините, произошла ошибка при генерации ответа."
            
            await memory.save_conversation(
                question=question,
                answer=answer,
                intent=intent.value,
                session_id=session_id
            )
            
            # Периодический анализ профиля каждые 10 диалогов
            history = await memory.get_conversation_history(limit=1)
            if len(history) % 10 == 0:
                logger.info("📊 Запуск автоматического анализа профиля (каждые 10 диалогов)...")
                from .services.memorybank import analyze_and_update_profile
                risk_analysis = await analyze_and_update_profile(memory, client)
                if risk_analysis:
                    logger.info(f"  ✓ Анализ: {risk_analysis.get('reasoning', 'N/A')}")
            
            logger.info("=" * 80)
            logger.info("✅ ОБЩИЙ ВОПРОС ОБРАБОТАН")
            logger.info("=" * 80)
            
            return {
                "intent": intent.value,
                "success": True,
                "answer": answer.strip(),
                "investor_profile": await memory.get_profile_summary()
            }
    
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА В ОРКЕСТРАТОРЕ")
        logger.error("=" * 80)
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        
        error_msg = "Извините, произошла ошибка при обработке вашего запроса."
        await memory.save_conversation(question, f"Ошибка: {str(e)}", "error", session_id=session_id)
        
        return {
            "intent": "unknown",
            "success": False,
            "answer": error_msg
        }