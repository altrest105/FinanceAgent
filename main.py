import pandas as pd
import numpy as np
import os
import requests
import faiss
import time
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

TOP_K = 8 # Количество контекстов для генерации ответа
TOP_K_GLOBAL = 16 # Количество контекстов для повторной генерации ответа
TOP_K_BEFORE_RERANK = 40 # Количество кандидатов для реранкера
LLM_MODEL = "openrouter/mistralai/mistral-small-3.2-24b-instruct" # Модель для генерации ответов openrouter/google/gemma-3-27b-it openrouter/mistralai/mistral-small-3.2-24b-instruct
EMBEDDING_MODEL = "text-embedding-3-small" # Модель для эмбеддингов
RERANK_MODEL = "deepinfra/Qwen/Qwen3-Reranker-4B" # Модель для реранкинга
BATCH_SIZE = 100  # Размер батча для получения эмбеддингов

# --- Тест числа токенов ---
def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens

# --- Эмбеддинг ---
# Функция разбиения текста на чанки по ##
def split_by_sections(text):
    if pd.isna(text) or not text.strip():
        return []
    parts = [t.strip() for t in text.split("##") if t.strip()]
    return parts if parts else [text]

# Функция получения эмбеддингов батчами
def get_embeddings_batch(client, texts):
    """Получение эмбеддингов для батча текстов"""
    if not texts:
        return []
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"⚠️ Ошибка при получении эмбеддингов: {e}")
        return [None] * len(texts)

# Функция подготовки эмбеддингов
def process_knowledge_base(input_file, output_csv, embeddings_file):
    print(f"🔄 Загрузка данных из {input_file}...")
    df = pd.read_csv(input_file)
    
    required_cols = {"id", "annotation", "tags", "text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV должен содержать колонки: {required_cols}")
    
    print(f"✅ Загружено {len(df)} документов")
    
    rows = []
    global_chunk_id = 0
    
    print("🔄 Разбиение на чанки по ##...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Разбиение"):
        chunks = split_by_sections(row["text"])
        
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
    print(f"✅ Создано {len(df_chunks)} чанков из {len(df)} документов")
    
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )
    
    all_embeddings = []
    texts = df_chunks["text_chunk"].tolist()
    
    print(f"🔄 Получение эмбеддингов для {len(texts)} чанков батчами по {BATCH_SIZE}...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Эмбеддинги"):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = get_embeddings_batch(client, batch)
        all_embeddings.extend(embeddings)
    
    df_chunks["embedding"] = all_embeddings
    
    df_valid = df_chunks[df_chunks["embedding"].notnull()].reset_index(drop=True)
    
    if len(df_valid) < len(df_chunks):
        print(f"⚠️ Предупреждение: {len(df_chunks) - len(df_valid)} чанков не получили эмбеддинги")

    np.save(embeddings_file, np.array(df_valid["embedding"].tolist()))
    df_valid.drop(columns=["embedding"]).to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ Всего строк: {len(df_valid)}")

# --- Retriever с FAISS и реранкером ---
class KnowledgeBaseRetriever:
    def __init__(self, db_csv, embeddings_file):
        print("🔄 Загрузка базы знаний в память...")
        self.df = pd.read_csv(db_csv)
        self.texts = self.df["text_chunk"].fillna("").tolist()
        
        # Загружаем эмбеддинги и создаем FAISS индекс
        embeddings = np.load(embeddings_file).astype('float32')
        
        # Нормализуем для косинусного расстояния
        faiss.normalize_L2(embeddings)
        
        # Создаем FAISS индекс для быстрого поиска
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        self.client = OpenAI(
            base_url="https://ai-for-finance-hack.up.railway.app/",
            api_key=EMBEDDER_API_KEY,
        )
        print(f"✅ База загружена: {len(self.texts)} чанков, FAISS индекс создан")
    
    def compute_query_embedding(self, query):
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL, 
            input=query
        )
        embedding = np.array(response.data[0].embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(embedding)
        return embedding
    
    def rerank_docs(self, query, documents):
        url = "https://ai-for-finance-hack.up.railway.app/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {EMBEDDER_API_KEY}"
        }
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️ Ошибка при реранжировании: {e}")
            return None
    
    def get_topk_contexts(self, query, repeat, max_retries=3, retry_delay=5):
        query_emb = self.compute_query_embedding(query)
        similarities, indices = self.index.search(query_emb, TOP_K_BEFORE_RERANK)
        
        candidate_texts = [self.texts[idx] for idx in indices[0]]

        # --- Попытки реранкера ---
        rerank_result = None
        for attempt in range(1, max_retries + 1):
            rerank_result = self.rerank_docs(query, candidate_texts)

            if rerank_result and "results" in rerank_result:
                break  # Успешно → выходим

            print(f"⚠️ Реранкер недоступен (попытка {attempt}/{max_retries})")
            time.sleep(retry_delay * attempt)  # экспоненциальная задержка

        if (repeat):
            # --- Если реранкер сработал ---
            if rerank_result and "results" in rerank_result:
                sorted_results = sorted(
                    rerank_result["results"],
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True
                )[:TOP_K]

                top_contexts = [candidate_texts[item["index"]] for item in sorted_results]
            
            else:
                print("⚠️ Реранкер не сработал после всех попыток → используем FAISS поиск")
                top_contexts = candidate_texts[:TOP_K]
        else:
            # --- Если реранкер сработал ---
            if rerank_result and "results" in rerank_result:
                sorted_results = sorted(
                    rerank_result["results"],
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True
                )[:TOP_K_GLOBAL]

                top_contexts = [candidate_texts[item["index"]] for item in sorted_results]
            
            else:
                print("⚠️ Реранкер не сработал после всех попыток → используем FAISS поиск")
                top_contexts = candidate_texts[:TOP_K_GLOBAL]

        return top_contexts

# --- Генерация ответа ---
# Функция построения запроса к LLM для генерации ответа
def build_answer(question, repeat):
    contexts = retriever.get_topk_contexts(question, repeat = repeat)

    context_text = "\n\n---\n\n".join(contexts)
    
    text = f"""Ты — русскоязычный ИИ-ассистент, эксперт в области финансов, права и государственных услуг.

    Твоя задача — дать подробный, структурированный и максимально полезный ответ на вопрос пользователя. Ты должен основываться строго на предоставленном тебе контексте (документах из базы знаний).

    ## Правила твоего ответа:
    -  Главный заголовок: Начинай ответ с основного заголовка, используя ###.
    -  Подзаголовки: Разбивай сложные ответы на логические блоки с помощью подзаголовков ####.
    -  Списки и Таблицы: Используй маркированные списки (- или *) для перечислений. **ВАЖНО! Для вопросов, требующих сравнения, ВСЕГДА используй таблицу (Markdown) для наглядности — это предпочтительнее, чем блоки с подзаголовками.**
    -  Выделение: Выделяй ключевые термины, определения и важные цифры жирным шрифтом (**...**).
    -  Синтез, не копирование: Не копируй текст из контекста слово в слово. Твоя цель — синтезировать информацию из разных частей контекста и изложить ее в виде единого, связного и легко читаемого ответа.
    -  Полнота ответа: Дважды проверь, что твой ответ полностью покрывает все аспекты и под-вопросы в вопросе клиента.
    -  Полезный вывод: Если это уместно и информация есть в контексте, завершай ответ практическим советом или кратким выводом (например, "На что обратить внимание", "Когда какой вариант выбрать").
    -  Ограничения: Отвечай только на основе предоставленного контекста. Не придумывай информацию и не используй свои общие знания.
    -  Если ответа нет: Если в контексте нет релевантной информации для ответа на вопрос (или для полноценного сравнения), вежливо сообщи: "Извините, но сейчас я не могу дать ответ на ваш вопрос."

    Контекст:
    {context_text}

    Вопрос клиента: {question}
    """

    return text

# Функция генерации ответа на вопрос
def answer_generation(question, client_ans, repeat = True):
    question_text = build_answer(question, repeat)

    if (num_tokens_from_string(question_text) > 8000):
        return "Извините, но сейчас я не могу дать ответ на ваш вопрос"

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = client_ans.chat.completions.create(
                model=LLM_MODEL,
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

        except RateLimitError:
            print(f"Повтор через 5 секунд")
            time.sleep(5)
            continue

        except APIError as e:
            print(f"Ошибка API: {e}")
            time.sleep(5)
            continue

        except Exception as e:
            print(f"Ошибка: {e}")
            break

    return "Извините, но сейчас я не могу дать ответ на ваш вопрос"


# --- Основной запуск ---
if __name__ == "__main__":   
    # Создание эмбеддинга
    if not os.path.exists("new_train_data.csv") or not os.path.exists("embeddings.npy"):
        # print("🔄 Создание базы знаний...")
        process_knowledge_base(
            input_file="train_data.csv",
            output_csv="new_train_data.csv",
            embeddings_file="embeddings.npy"
        )
    else:
        print("⚠️ База знаний уже существует, пропускаем создание")

    # Инициализация ретривера
    retriever = KnowledgeBaseRetriever(
        db_csv="new_train_data.csv",
        embeddings_file="embeddings.npy"
    )

    # Генерация и сохранение ответов на вопросы из questions.csv
    questions = pd.read_csv('./questions.csv', encoding="utf-8")
    questions_list = questions['Вопрос'].tolist()
    answer_list = []

    client_ans = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY,
    )

    for current_question in tqdm(questions_list, desc="Генерация ответов"):
        answer = answer_generation(question=current_question, client_ans=client_ans)
        while ("Извините" in answer):
            print("Новый запрос")
            answer = answer_generation(question=current_question, client_ans=client_ans, repeat = False)
            if (TOP_K_GLOBAL == 32):
                TOP_K_GLOBAL = 16
                break
            TOP_K_GLOBAL *= 2

        answer_list.append(answer.replace('\n', ' ').strip())
    questions['Ответы на вопрос'] = answer_list
    questions.to_csv('submission.csv', index=False, encoding="utf-8")
    print("✅ Ответы сохранены в submission.csv")