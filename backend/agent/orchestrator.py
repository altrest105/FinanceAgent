import logging
import tiktoken
import asyncio
from enum import Enum
from openai import AsyncOpenAI, RateLimitError, APIError
from django.conf import settings

logger = logging.getLogger(__name__)

ENCODING = tiktoken.get_encoding("cl100k_base")

class IntentType(Enum):
    """Типы намерений пользователя"""
    FINANCE_GENERAL = "finance_general"  # Общие вопросы про финансы (RAG)
    SECURITIES_NEWS = "securities_news"  # Вопросы про ценные бумаги (парсинг новостей)
    TRADE = "trade"  # Покупка/продажа акций

class TradeAction(Enum):
    """Типы торговых операций"""
    BUY = "buy"
    SELL = "sell"

def build_intent_classification_prompt(question):
    """Промпт для определения намерения пользователя"""
    return f"""Определи намерение пользователя на основе его вопроса.

Возможные намерения:
1. FINANCE_GENERAL - общие вопросы про финансы, инвестиции, право, государственные услуги (НЕ про конкретные акции/облигации)
2. SECURITIES_NEWS - вопросы про конкретные ценные бумаги, акции, компании, их котировки, новости, аналитику
3. TRADE - пользователь хочет купить или продать ценные бумаги (явное намерение совершить сделку)

Примеры:
- "Что такое дивиденды?" -> FINANCE_GENERAL
- "Как платить налоги с инвестиций?" -> FINANCE_GENERAL
- "Какие новости по Сбербанку?" -> SECURITIES_NEWS
- "Что происходит с акциями Газпрома?" -> SECURITIES_NEWS
- "Купи 10 акций Яндекса" -> TRADE
- "Продай Сбер по 280 рублей" -> TRADE

Вопрос: {question}

Ответь ТОЛЬКО одним словом: FINANCE_GENERAL, SECURITIES_NEWS или TRADE"""

def build_trade_extraction_prompt(question):
    """Промпт для извлечения параметров торговой операции"""
    return f"""Извлеки параметры торговой операции из запроса пользователя.

Вопрос: {question}

Верни ответ СТРОГО в формате JSON:
{{
    "action": "buy" или "sell",
    "ticker": "тикер акции (например, SBER, GAZP)",
    "quantity": число акций (целое число),
    "price": цена (число или null, если не указана)
}}

Если какой-то параметр не указан явно, используй null."""

def build_securities_extraction_prompt(question):
    """Промпт для извлечения тикеров ценных бумаг из вопроса"""
    return f"""Извлеки все упомянутые тикеры или названия ценных бумаг из вопроса пользователя.

Вопрос: {question}

Верни ответ СТРОГО в формате JSON:
{{
    "tickers": ["SBER", "GAZP"],  // массив тикеров
    "companies": ["Сбербанк", "Газпром"]  // массив названий компаний
}}

Если ничего не найдено, верни пустые массивы."""

def build_answer_prompt(question, context_text=""):
    """Промпт для генерации аналитического ответа"""
    text = f"""Ты — русскоязычный ИИ-ассистент, эксперт в области финансов, инвестиций, права и государственных услуг.

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
    - Если недостаточно информации для ответа на вопрос, вежливо сообщи: "Извините, но сейчас я не могу дать ответ на ваш вопрос."
    - СОХРАНЯЙ все переносы строк и пустые строки, чтобы обеспечить требуемую Markdown-структуру (заголовки, списки, таблицы) в ответе.

    Контекст:
    {context_text}

    Вопрос клиента: {question}
    """
    return text

def build_securities_answer_prompt(question, news_context=""):
    """Промпт для ответа на вопросы о ценных бумагах с новостями"""
    text = f"""Ты — русскоязычный ИИ-ассистент, эксперт в области ценных бумаг, фондового рынка и финансовой аналитики.

    Твоя задача — дать подробный, структурированный ответ на вопрос пользователя о ценных бумагах, основываясь на предоставленных новостях и данных.

    Правила твоего ответа:
    - Главный заголовок: Начинай ответ с основного заголовка, используя ###.
    - Подзаголовки: Используй ####. ОБЯЗАТЕЛЬНО нумеруй логические разделы.
    - Актуальность: Упоминай даты новостей и событий.
    - Источники: Если в новостях есть конкретные данные (цены, проценты), указывай их.
    - Таблицы: Для сравнения котировок используй таблицы в формате Markdown.
    - Выделение: Выделяй важные цифры, даты, изменения цен жирным шрифтом (**...**).
    - Объективность: Представляй разные точки зрения, если они есть в новостях.
    - Если новостей нет или недостаточно информации, честно сообщи об этом.

    Новости и данные:
    {news_context}

    Вопрос пользователя: {question}
    """
    return text

async def call_llm(client, prompt, max_tokens=None):
    """Универсальная функция для вызова LLM"""
    if len(ENCODING.encode(prompt)) > settings.LLM_CONFIG.get("max_tokens", 8000):
        raise ValueError("Промпт слишком длинный")

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=settings.LLM_CONFIG.get("model"),
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ],
                max_tokens=max_tokens
            )

            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            
            return None

        except (RateLimitError, APIError) as e:
            logger.warning(f"LLM API Error/Rate Limit (попытка {attempt+1}/{MAX_RETRIES}): %s", e)
            await asyncio.sleep(5)
            continue

        except Exception as e:
            logger.error(f"Непредвиденная ошибка при вызове LLM: {e}")
            break

    return None

async def classify_intent(question, client):
    """Определение намерения пользователя"""
    prompt = build_intent_classification_prompt(question)
    response = await call_llm(client, prompt, max_tokens=20)
    
    if not response:
        return IntentType.FINANCE_GENERAL  # По умолчанию
    
    response_upper = response.upper().strip()
    if "TRADE" in response_upper:
        return IntentType.TRADE
    elif "SECURITIES_NEWS" in response_upper or "SECURITIES" in response_upper:
        return IntentType.SECURITIES_NEWS
    return IntentType.FINANCE_GENERAL

async def extract_securities_info(question, client):
    """Извлечение информации о ценных бумагах из вопроса"""
    import json
    
    prompt = build_securities_extraction_prompt(question)
    response = await call_llm(client, prompt, max_tokens=300)
    
    if not response:
        return None
    
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        info = json.loads(response)
        return info
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON информации о ценных бумагах: {e}")
        return None

async def extract_trade_params(question, client):
    """Извлечение параметров торговой операции"""
    import json
    
    prompt = build_trade_extraction_prompt(question)
    response = await call_llm(client, prompt, max_tokens=200)
    
    if not response:
        return None
    
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        params = json.loads(response)
        return params
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON параметров торговли: {e}")
        return None

async def fetch_news_for_securities(securities_info):
    """
    Получение новостей для ценных бумаг
    TODO: Интеграция с новостным API
    """
    logger.info(f"Получение новостей для: {securities_info}")
    
    # Заглушка - здесь должна быть интеграция с парсером новостей
    tickers = securities_info.get("tickers", [])
    companies = securities_info.get("companies", [])
    
    # Пример структуры данных
    mock_news = f"""
    Последние новости по запрошенным компаниям ({', '.join(tickers if tickers else companies)}):
    
    1. Сбербанк (SBER) - 15.11.2024
       - Цена закрытия: 285.50 руб (+2.3%)
       - Объявлены квартальные результаты, прибыль выросла на 15%
       
    2. Газпром (GAZP) - 15.11.2024
       - Цена закрытия: 168.20 руб (-0.8%)
       - Компания объявила о новых контрактах на поставку газа
    
    (Данные приведены для примера, требуется интеграция с реальным источником новостей)
    """
    
    return mock_news

async def fetch_rag_context(question):
    """
    Получение контекста из RAG системы
    TODO: Интеграция с векторной БД и retriever
    """
    logger.info(f"Получение контекста из RAG для вопроса: {question}")
    
    # Заглушка - здесь должна быть интеграция с RAG системой
    # Например, поиск в векторной БД (Pinecone, Weaviate, ChromaDB и т.д.)
    
    mock_context = """
    Релевантная информация из базы знаний:
    
    - Дивиденды - это часть прибыли компании, которая распределяется между акционерами.
    - Налог на доходы от инвестиций составляет 13% для резидентов РФ.
    - ИИС (Индивидуальный Инвестиционный Счет) позволяет получить налоговые вычеты.
    
    (Требуется интеграция с реальной RAG системой)
    """
    
    return mock_context

async def execute_trade(params):
    """Выполнение торговой операции (заглушка для интеграции с брокером)"""
    logger.info(f"Выполнение торговой операции: {params}")
    
    action = params.get("action")
    ticker = params.get("ticker")
    quantity = params.get("quantity")
    price = params.get("price")
    
    if not all([action, ticker, quantity]):
        return {
            "success": False,
            "message": "Недостаточно параметров для выполнения операции. Укажите действие (купить/продать), тикер и количество акций."
        }
    
    # TODO: Интеграция с брокерским API
    return {
        "success": True,
        "message": f"Заявка на {'покупку' if action == 'buy' else 'продажу'} {quantity} акций {ticker} {'по цене ' + str(price) if price else 'по рыночной цене'} успешно размещена.",
        "order_id": "ORDER_12345",
        "params": params
    }

async def generate_finance_answer(question, client, rag_context=""):
    """Генерация ответа на общие финансовые вопросы с использованием RAG"""
    prompt = build_answer_prompt(question, rag_context)
    answer = await call_llm(client, prompt)
    
    if not answer:
        return "Извините, произошла ошибка при генерации ответа."
    
    return answer

async def generate_securities_answer(question, client, news_context=""):
    """Генерация ответа на вопросы о ценных бумагах с использованием новостей"""
    prompt = build_securities_answer_prompt(question, news_context)
    answer = await call_llm(client, prompt)
    
    if not answer:
        return "Извините, произошла ошибка при генерации ответа о ценных бумагах."
    
    return answer

async def process_question(question, rag_context=None):
    """
    Главная функция-оркестратор для обработки вопроса пользователя
    
    Args:
        question: вопрос пользователя
        rag_context: предзагруженный контекст из RAG (опционально)
    
    Returns:
        dict: {
            "intent": "finance_general" | "securities_news" | "trade",
            "answer": str,
            "trade_result": dict,  # только для trade
            "securities_info": dict,  # только для securities_news
            "success": bool
        }
    """
    client = AsyncOpenAI(
        base_url=settings.LLM_CONFIG.get("base_url"),
        api_key=settings.LLM_CONFIG.get("api_key"),
    )
    
    try:
        # 1. Определяем намерение
        intent = await classify_intent(question, client)
        logger.info(f"Определено намерение: {intent.value}")
        
        # 2. Обрабатываем в зависимости от намерения
        if intent == IntentType.TRADE:
            # === ТОРГОВАЯ ОПЕРАЦИЯ ===
            params = await extract_trade_params(question, client)
            
            if not params:
                return {
                    "intent": intent.value,
                    "success": False,
                    "answer": "Не удалось определить параметры торговой операции. Пожалуйста, уточните: какую акцию, сколько и по какой цене вы хотите купить или продать?"
                }
            
            trade_result = await execute_trade(params)
            
            return {
                "intent": intent.value,
                "success": trade_result["success"],
                "trade_result": trade_result,
                "answer": trade_result["message"]
            }
        
        elif intent == IntentType.SECURITIES_NEWS:
            # === ВОПРОС О ЦЕННЫХ БУМАГАХ (парсинг новостей) ===
            securities_info = await extract_securities_info(question, client)
            
            # Получаем новости
            news_context = await fetch_news_for_securities(securities_info or {})
            
            # Генерируем ответ
            answer = await generate_securities_answer(question, client, news_context)
            
            # Повторная попытка при неудовлетворительном ответе
            attempts = 1
            while ("Извините" in answer) and (attempts <= 3):
                logger.warning(f"⚠️ Удовлетворительный ответ не получен (попытка {attempts}/3)")
                answer = await generate_securities_answer(question, client, news_context)
                attempts += 1
            
            return {
                "intent": intent.value,
                "success": True,
                "securities_info": securities_info,
                "answer": answer.strip()
            }
        
        else:
            # === ОБЩИЙ ФИНАНСОВЫЙ ВОПРОС (RAG) ===
            # Если контекст не передан, получаем его
            if rag_context is None:
                rag_context = await fetch_rag_context(question)
            
            # Генерируем ответ
            answer = await generate_finance_answer(question, client, rag_context)
            
            # Повторная попытка при неудовлетворительном ответе
            attempts = 1
            while ("Извините" in answer) and (attempts <= 3):
                logger.warning(f"⚠️ Удовлетворительный ответ не получен (попытка {attempts}/3)")
                answer = await generate_finance_answer(question, client, rag_context)
                attempts += 1
            
            return {
                "intent": intent.value,
                "success": True,
                "answer": answer.strip()
            }
    
    except Exception as e:
        logger.error(f"Ошибка в оркестраторе: {e}")
        return {
            "intent": "unknown",
            "success": False,
            "answer": "Извините, произошла ошибка при обработке вашего запроса."
        }

# Обратная совместимость
async def get_answer_with_retry(question, context_text=""):
    """Упрощенная функция для обратной совместимости"""
    result = await process_question(question, rag_context=context_text if context_text else None)
    return result.get("answer", "Ошибка обработки запроса")