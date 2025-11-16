import pandas as pd
import numpy as np
import faiss
import asyncio
import os
from openai import AsyncOpenAI
from django.conf import settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import tiktoken
import logging

logger = logging.getLogger(__name__)

# Конфигурация
TOP_K = 3
TOP_K_BEFORE_RERANK = 40
BATCH_SIZE = 100

# Глобальные переменные
_retriever = None
_llm_client = None
_local_embedder = None
_local_reranker = None

# Утилиты
def num_tokens_from_string(string, encoding_name = "cl100k_base"):
    """Подсчет токенов в строке"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def split_by_sections(text):
    """Разбиение текста на чанки по ##"""
    if pd.isna(text) or not text.strip():
        return []
    
    parts = [t.strip() for t in text.split("##") if t.strip()]
    return parts if parts else [text]

# Эмбеддер
class LocalEmbedder:
    def __init__(self, model_name = "BAAI/bge-m3"):
        logger.info("=" * 80)
        logger.info("🔮 ИНИЦИАЛИЗАЦИЯ ЛОКАЛЬНОГО ЭМБЕДДЕРА")
        logger.info("=" * 80)
        logger.info(f"  ├─ Модель: {model_name}")
        
        try:
            logger.debug("  ⏳ Загрузка модели SentenceTransformer...")
            self.model = SentenceTransformer(model_name)
            
            logger.info("=" * 80)
            logger.info("✅ ЛОКАЛЬНЫЙ ЭМБЕДДЕР ЗАГРУЖЕН")
            logger.info("=" * 80)
            logger.info(f"  └─ Размерность эмбеддингов: {self.model.get_sentence_embedding_dimension()}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}", exc_info=True)
            raise
    
    def encode(self, texts, batch_size = 32, show_progress = False):
        logger.debug(f"🔮 Получение эмбеддингов")
        logger.debug(f"  ├─ Количество текстов: {len(texts)}")
        logger.debug(f"  └─ Batch size: {batch_size}")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            logger.info(f"  ✅ Получено эмбеддингов: {len(embeddings)}")
            logger.debug(f"     └─ Форма: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения эмбеддингов: {e}", exc_info=True)
            raise

# Реранкер
class LocalReranker:
    def __init__(self, model_name = "BAAI/bge-reranker-v2-m3"):
        logger.info("=" * 80)
        logger.info("🎯 ИНИЦИАЛИЗАЦИЯ ЛОКАЛЬНОГО РЕРАНКЕРА")
        logger.info("=" * 80)
        logger.info(f"  ├─ Модель: {model_name}")
        
        try:
            logger.debug("  ⏳ Загрузка модели CrossEncoder...")
            self.model = CrossEncoder(model_name, max_length=512)
            
            logger.info("=" * 80)
            logger.info("✅ ЛОКАЛЬНЫЙ РЕРАНКЕР ЗАГРУЖЕН")
            logger.info("=" * 80)
            logger.info(f"  └─ Max length: 512")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки реранкера: {e}", exc_info=True)
            raise
    
    def rerank(self, query, documents, top_k = TOP_K):
        logger.debug(f"🔄 Локальный реранкинг")
        logger.debug(f"  ├─ Запрос: {query[:50]}...")
        logger.debug(f"  ├─ Документов: {len(documents)}")
        logger.debug(f"  └─ Top-K: {top_k}")
        
        if not documents:
            logger.warning("  ⚠️ Пустой список документов")
            return []
        
        try:
            pairs = [[query, doc] for doc in documents]
            
            logger.debug("  ⏳ Вычисление similarity scores...")
            scores = self.model.predict(pairs)
            
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            
            results = [
                (int(idx), float(scores[idx])) 
                for idx in ranked_indices
            ]
            
            logger.info(f"  ✅ Реранкинг завершён")
            logger.debug(f"     ├─ Top-3 scores: {[score for _, score in results[:3]]}")
            logger.debug(f"     └─ Возвращено: {len(results)} результатов")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка реранкинга: {e}", exc_info=True)
            return [(i, 1.0 / (i + 1)) for i in range(min(top_k, len(documents)))]

# Обработка базы знаний
async def process_knowledge_base(input_file, output_csv, embeddings_file):
    global _local_embedder
    
    logger.info("=" * 80)
    logger.info("📚 ОБРАБОТКА БАЗЫ ЗНАНИЙ")
    logger.info("=" * 80)
    logger.info(f"  ├─ Входной файл: {input_file}")
    logger.info(f"  ├─ Выходной CSV: {output_csv}")
    logger.info(f"  └─ Файл эмбеддингов: {embeddings_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"❌ Файл не найден: {input_file}")
        raise FileNotFoundError(f"Файл не найден: {input_file}")
    
    logger.debug(f"  ⏳ Загрузка данных из {input_file}...")
    df = pd.read_csv(input_file)
    
    logger.info(f"  ✓ Загружено документов: {len(df)}")
    logger.debug(f"     └─ Колонки: {list(df.columns)}")
    
    logger.info("  ⏳ Разбиение документов на чанки...")
    rows = []
    global_chunk_id = 0
    
    for doc_idx, (_, row) in enumerate(df.iterrows(), 1):
        chunks = split_by_sections(row["text"])
        
        if doc_idx <= 3:
            logger.debug(f"     Документ {doc_idx}: {len(chunks)} чанков")
        
        for chunk_idx, chunk in enumerate(chunks):
            rows.append({
                "doc_id": row["id"],
                "annotation": row["annotation"],
                "tags": row["tags"],
                "text_chunk": chunk,
                "chunk_id": global_chunk_id,
                "chunk_idx_in_doc": chunk_idx
            })
            global_chunk_id += 1
    
    df_chunks = pd.DataFrame(rows)
    logger.info(f"  ✅ Создано чанков: {len(df_chunks)} из {len(df)} документов")
    logger.debug(f"     └─ Среднее чанков на документ: {len(df_chunks) / len(df):.2f}")
    
    if _local_embedder is None:
        logger.info("  ⏳ Инициализация локального эмбеддера...")
        _local_embedder = LocalEmbedder()
    
    texts = df_chunks["text_chunk"].tolist()
    logger.info(f"  ⏳ Получение эмбеддингов для {len(texts)} чанков...")
    
    embeddings = _local_embedder.encode(texts, batch_size=BATCH_SIZE, show_progress=True)
    
    logger.info("  ⏳ Сохранение результатов...")
    
    os.makedirs(os.path.dirname(embeddings_file) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    
    np.save(embeddings_file, embeddings)
    logger.debug(f"     ├─ Эмбеддинги сохранены: {embeddings_file}")
    
    df_chunks.to_csv(output_csv, index=False, encoding="utf-8")
    logger.debug(f"     └─ CSV сохранён: {output_csv}")
    
    logger.info("=" * 80)
    logger.info("✅ БАЗА ЗНАНИЙ УСПЕШНО ОБРАБОТАНА")
    logger.info("=" * 80)
    logger.info(f"  ├─ Документов: {len(df)}")
    logger.info(f"  ├─ Чанков: {len(df_chunks)}")
    logger.info(f"  └─ Размерность эмбеддингов: {embeddings.shape[1]}")
    logger.info("=" * 80)

# Ретривер
class KnowledgeBaseRetriever:
    def __init__(self, db_csv, embeddings_file):
        logger.info("=" * 80)
        logger.info("🗄️ ИНИЦИАЛИЗАЦИЯ RETRIEVER")
        logger.info("=" * 80)
        logger.info(f"  ├─ CSV файл: {db_csv}")
        logger.info(f"  └─ Эмбеддинги: {embeddings_file}")
        
        if not os.path.exists(db_csv):
            logger.error(f"❌ CSV файл не найден: {db_csv}")
            raise FileNotFoundError(f"CSV файл не найден: {db_csv}")
        
        if not os.path.exists(embeddings_file):
            logger.error(f"❌ Файл эмбеддингов не найден: {embeddings_file}")
            raise FileNotFoundError(f"Файл эмбеддингов не найден: {embeddings_file}")
        
        logger.debug("  ⏳ Загрузка CSV...")
        self.df = pd.read_csv(db_csv)
        logger.info(f"  ✓ Загружено записей: {len(self.df)}")
        
        self.texts = self.df["text_chunk"].fillna("").tolist()
        logger.debug(f"     └─ Извлечено текстов: {len(self.texts)}")
        
        logger.debug("  ⏳ Загрузка эмбеддингов...")
        embeddings = np.load(embeddings_file).astype('float32')
        logger.info(f"  ✓ Форма эмбеддингов: {embeddings.shape}")
        
        logger.debug("  ⏳ Нормализация эмбеддингов...")
        faiss.normalize_L2(embeddings)
        
        logger.debug("  ⏳ Создание FAISS индекса...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        global _local_reranker
        if _local_reranker is None:
            logger.debug("  ⏳ Инициализация локального реранкера...")
            _local_reranker = LocalReranker()
        
        self.reranker = _local_reranker
        
        logger.info("=" * 80)
        logger.info("✅ RETRIEVER ГОТОВ")
        logger.info("=" * 80)
        logger.info(f"  ├─ Чанков в базе: {len(self.texts)}")
        logger.info(f"  ├─ Размерность: {dimension}")
        logger.info(f"  ├─ FAISS индекс: IndexFlatIP")
        logger.info(f"  └─ Реранкер: Локальный (CrossEncoder)")
        logger.info("=" * 80)
    
    def compute_query_embedding(self, query):
        global _local_embedder
        
        logger.debug(f"🔍 Вычисление эмбеддинга запроса")
        logger.debug(f"  ├─ Длина запроса: {len(query)} символов")
        
        try:
            if _local_embedder is None:
                logger.info("  ⏳ Инициализация локального эмбеддера...")
                _local_embedder = LocalEmbedder()
            
            embedding = _local_embedder.encode([query])[0]
            embedding = embedding.astype('float32').reshape(1, -1)
            
            logger.debug(f"  ├─ Форма эмбеддинга: {embedding.shape}")
            
            faiss.normalize_L2(embedding)
            logger.debug(f"  └─ Эмбеддинг нормализован")
            
            logger.info(f"  ✅ Эмбеддинг запроса получен")
            return embedding
            
        except Exception as e:
            logger.error("❌ Ошибка вычисления эмбеддинга запроса")
            logger.error(f"  ├─ Запрос: {query[:100]}...")
            logger.error(f"  └─ Ошибка: {e}", exc_info=True)
            raise
    
    def get_topk_contexts(self, query, top_k = TOP_K):
        logger.info("=" * 80)
        logger.info("🔎 ПОИСК РЕЛЕВАНТНЫХ КОНТЕКСТОВ")
        logger.info("=" * 80)
        logger.info(f"  ├─ Запрос: {query[:100]}...")
        logger.info(f"  ├─ Top-K: {top_k}")
        logger.info(f"  └─ Кандидатов для реранкинга: {TOP_K_BEFORE_RERANK}")
        
        logger.debug("  ⏳ Этап 1: Получение эмбеддинга запроса...")
        query_emb = self.compute_query_embedding(query)
        
        logger.debug(f"  ⏳ Этап 2: FAISS поиск (top-{TOP_K_BEFORE_RERANK})...")
        similarities, indices = self.index.search(query_emb, TOP_K_BEFORE_RERANK)
        
        logger.debug(f"  ├─ Найдено индексов: {len(indices[0])}")
        logger.debug(f"  └─ Similarity scores: {similarities[0][:5].tolist()}")
        
        candidate_texts = [self.texts[idx] for idx in indices[0]]
        logger.info(f"  ✓ Получено кандидатов из FAISS: {len(candidate_texts)}")
        
        logger.debug(f"  ⏳ Этап 3: Локальный реранкинг...")
        reranked_results = self.reranker.rerank(query, candidate_texts, top_k)
        
        final_texts = [candidate_texts[idx] for idx, _ in reranked_results]
        
        logger.info("=" * 80)
        logger.info("✅ КОНТЕКСТЫ ПОЛУЧЕНЫ (С ЛОКАЛЬНЫМ РЕРАНКИНГОМ)")
        logger.info("=" * 80)
        logger.info(f"  ├─ Возвращено контекстов: {len(final_texts)}")
        logger.info(f"  └─ Top-3 rerank scores: {[score for _, score in reranked_results[:3]]}")
        logger.info("=" * 80)
        
        return final_texts

# Генерация ответов
def build_answer_prompt(question, context_text):
    logger.debug(f"📝 Построение промпта")
    logger.debug(f"  ├─ Длина вопроса: {len(question)} символов")
    logger.debug(f"  └─ Длина контекста: {len(context_text)} символов")
    
    prompt = f"""Ты — русскоязычный ИИ-ассистент, эксперт в области финансов, инвестиций, права и государственных услуг.

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
    
    token_count = num_tokens_from_string(prompt)
    logger.debug(f"  └─ Итоговый промпт: {token_count} токенов")
    
    return prompt

# async def generate_answer(question, top_k = TOP_K):
#     global _retriever, _llm_client
    
#     logger.info("=" * 80)
#     logger.info("🤖 ГЕНЕРАЦИЯ ОТВЕТА (RAG)")
#     logger.info("=" * 80)
#     logger.info(f"  ├─ Вопрос: {question[:100]}...")
#     logger.info(f"  └─ Top-K контекстов: {top_k}")
    
#     if not _retriever:
#         logger.error("❌ RAG система не инициализирована!")
#         raise RuntimeError("RAG система не инициализирована. Вызовите initialize_rag() сначала.")
    
#     try:
#         logger.debug("  ⏳ Этап 1: Получение релевантных контекстов...")
#         contexts = _retriever.get_topk_contexts(question, top_k)
#         context_text = "\n\n---\n\n".join(contexts)
        
#         logger.info(f"  ✓ Получено контекстов: {len(contexts)}")
#         logger.debug(f"     └─ Общая длина контекста: {len(context_text)} символов")
        
#         logger.debug("  ⏳ Этап 2: Построение промпта...")
#         prompt = build_answer_prompt(question, context_text)
        
#         token_count = num_tokens_from_string(prompt)
#         max_tokens = 8000
        
#         logger.debug(f"  ├─ Токенов в промпте: {token_count}")
#         logger.debug(f"  └─ Максимум токенов: {max_tokens}")
        
#         if token_count > max_tokens:
#             logger.warning(f"  ⚠️ Промпт слишком длинный ({token_count} > {max_tokens})")
#             logger.info("=" * 80)
#             return "Извините, но сейчас я не могу дать ответ на ваш вопрос (слишком длинный контекст)."
        
#         logger.debug("  ⏳ Этап 3: Генерация ответа через LLM...")
#         logger.debug(f"     ├─ Модель: {settings.LLM_CONFIG.get('model')}")
#         logger.debug(f"     └─ Max retries: 3")
        
#         MAX_RETRIES = 3
#         for attempt in range(1, MAX_RETRIES + 1):
#             try:
#                 logger.debug(f"     ⏳ Попытка {attempt}/{MAX_RETRIES}...")
                
#                 response = await _llm_client.chat.completions.create(
#                     model=settings.LLM_CONFIG.get("model"),
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": prompt
#                         }
#                     ]
#                 )
                
#                 if response and response.choices and response.choices[0].message.content:
#                     answer = response.choices[0].message.content.strip()
                    
#                     logger.info("=" * 80)
#                     logger.info("✅ ОТВЕТ УСПЕШНО СГЕНЕРИРОВАН")
#                     logger.info("=" * 80)
#                     logger.info(f"  ├─ Длина ответа: {len(answer)} символов")
#                     logger.info(f"  ├─ Токенов (примерно): {num_tokens_from_string(answer)}")
#                     logger.info(f"  └─ Попытка: {attempt}/{MAX_RETRIES}")
#                     logger.info("=" * 80)
                    
#                     return answer
#                 else:
#                     logger.warning(f"     ⚠️ Пустой ответ от LLM")
#                     logger.info("=" * 80)
#                     return "Извините, но сейчас я не могу дать ответ на ваш вопрос."
                    
#             except Exception as e:
#                 logger.warning(f"     ⚠️ Ошибка (попытка {attempt}/{MAX_RETRIES}): {e}")
                
#                 if attempt < MAX_RETRIES:
#                     logger.debug(f"     ⏳ Ждём 5 секунд перед следующей попыткой...")
#                     await asyncio.sleep(5)
#                 else:
#                     logger.error("     ❌ Все попытки исчерпаны")
        
#         logger.info("=" * 80)
#         return "Извините, но сейчас я не могу дать ответ на ваш вопрос."
        
#     except Exception as e:
#         logger.error("=" * 80)
#         logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА В GENERATE_ANSWER")
#         logger.error("=" * 80)
#         logger.error(f"Вопрос: {question[:100]}...")
#         logger.error(f"Тип ошибки: {type(e).__name__}")
#         logger.error(f"Сообщение: {str(e)}", exc_info=True)
#         logger.error("=" * 80)
        
#         return "Извините, произошла ошибка при обработке вашего запроса."

async def initialize_rag(force_rebuild = False):
    global _retriever, _llm_client, _local_embedder, _local_reranker
    from ..apps import AgentConfig
    
    input_file = os.path.join(settings.BASE_DIR, "files", "train_data.csv")
    output_csv = os.path.join(settings.BASE_DIR, "files", "new_train_data.csv")
    embeddings_file = os.path.join(settings.BASE_DIR, "files", "embeddings.npy")
    
    logger.info("=" * 80)
    logger.info("🚀 ИНИЦИАЛИЗАЦИЯ RAG СИСТЕМЫ")
    logger.info("=" * 80)
    logger.info(f"  ├─ Входной файл: {input_file}")
    logger.info(f"  ├─ DB CSV: {output_csv}")
    logger.info(f"  └─ Embeddings: {embeddings_file}")
    
    if AgentConfig.llm_client:
        _llm_client = AgentConfig.llm_client
        logger.info("  ✓ Используем LLM клиент из AppConfig")
    else:
        logger.debug("  ⏳ Создание LLM клиента...")
        _llm_client = AsyncOpenAI(
            base_url=settings.LLM_CONFIG.get("base_url"),
            api_key=settings.LLM_CONFIG.get("api_key"),
        )
        logger.info(f"  ✓ LLM клиент создан")
    
    if AgentConfig.embedder_model:
        _local_embedder = AgentConfig.embedder_model
        logger.info("  ✓ Используем эмбеддер из AppConfig")
    
    if AgentConfig.reranker_model:
        _local_reranker = AgentConfig.reranker_model
        logger.info("  ✓ Используем реранкер из AppConfig")
    
    if force_rebuild or not os.path.exists(output_csv) or not os.path.exists(embeddings_file):
        logger.info("🔄 Создание/обновление базы знаний...")
        await process_knowledge_base(input_file, output_csv, embeddings_file)
    else:
        logger.info("✅ База знаний уже существует")
    
    logger.debug("  ⏳ Загрузка Retriever...")
    _retriever = KnowledgeBaseRetriever(output_csv, embeddings_file)
    
    logger.info("=" * 80)
    logger.info("✅ RAG СИСТЕМА УСПЕШНО ИНИЦИАЛИЗИРОВАНА")
    logger.info("=" * 80)
    logger.info(f"  ├─ Чанков в базе: {len(_retriever.texts)}")
    logger.info(f"  ├─ LLM: готов")
    logger.info(f"  ├─ Эмбеддер: {'из AppConfig' if _local_embedder else 'локальный'}")
    logger.info(f"  └─ Реранкер: {'из AppConfig' if _local_reranker else 'локальный'}")
    logger.info("=" * 80)

async def get_rag_context(question, top_k = TOP_K):
    global _retriever
    
    logger.debug(f"📚 Запрос RAG контекста (без генерации ответа)")
    logger.debug(f"  ├─ Вопрос: {question[:100]}...")
    logger.debug(f"  └─ Top-K: {top_k}")
    
    if not _retriever:
        logger.error("❌ RAG система не инициализирована!")
        raise RuntimeError("RAG система не инициализирована.")
    
    contexts = _retriever.get_topk_contexts(question, top_k)
    context = "\n\n---\n\n".join(contexts)
    
    logger.info(f"  ✅ Контекст получен ({len(context)} символов)")
    return context