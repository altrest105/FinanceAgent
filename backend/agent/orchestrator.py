import logging
import tiktoken
import asyncio
from openai import AsyncOpenAI, RateLimitError, APIError
from django.conf import settings

logger = logging.getLogger(__name__)

ENCODING = tiktoken.get_encoding("cl100k_base")

def build_answer(question, context_text=""):
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
    - Если ответа нет, постарайся ответить сам. Если и в таком случае недостаточно информации для ответа на вопрос, вежливо сообщи: "Извините, но сейчас я не могу дать ответ на ваш вопрос."
    - СОХРАНЯЙ все переносы строк и пустые строки, чтобы обеспечить требуемую Markdown-структуру (заголовки, списки, таблицы) в ответе.

    Контекст:
    {context_text}

    Вопрос клиента: {question}
    """

    return text

async def answer_generation(question, client_ans, context_text=""):
    question_text = build_answer(question, context_text=context_text)

    if (len(ENCODING.encode(question_text)) > settings.LLM_CONFIG.get("max_tokens", 8000)):
        return "Извините, но сейчас я не могу дать ответ на ваш вопрос"

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = await client_ans.chat.completions.create(
                model=settings.LLM_CONFIG.get("model"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question_text}
                        ]
                    }
                ]
            )

            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                return "Извините, но сейчас я не могу дать ответ на ваш вопрос"

        except (RateLimitError, APIError) as e:
            logger.warning("LLM API Error/Rate Limit, retrying: %s", e)
            await asyncio.sleep(5)
            continue

        except Exception as e:
            logger.error(f"Непредвиденная ошибка: {e}")
            break

    return "Извините, произошла ошибка при генерации ответа."

async def get_answer_with_retry(question):
    client_ans = AsyncOpenAI(
        base_url=settings.LLM_CONFIG.get("base_url"),
        api_key=settings.LLM_CONFIG.get("api_key"),
    )
    
    answer = await answer_generation(question=question, client_ans=client_ans)
    attempts = 1
    
    while ("Извините" in answer) and (attempts <= 3):
        logger.warning(f"⚠️ Удовлетворительный ответ не получен (попытка {attempts}/3)")
        # В реальной RAG-логике здесь менялся бы TOP_K. Пока это просто повтор.
        # TOP_K_GLOBAL = RANKS.get(str(attempts), 4) 
        
        answer = await answer_generation(question=question, client_ans=client_ans)
        attempts += 1

    return answer.strip()