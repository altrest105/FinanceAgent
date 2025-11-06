import os
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from main import answer_generation
import tiktoken

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
TOP_K = 5
MAX_TOKENS_PROMT = 7000


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def compute_query_embedding(query: str) -> np.ndarray:
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return np.array(response.data[0].embedding)

def get_topk_texts(query: str, db_csv: str, embeddings_file: str, top_k: int = TOP_K) -> str:
    df = pd.read_csv(db_csv)

    result_texts = df["text_chunk"].fillna("").tolist()

    db_embeddings = np.load(embeddings_file)
    query_emb = compute_query_embedding(query)

    similarities = cosine_similarity([query_emb], db_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    top_texts = [result_texts[i] for i in top_indices]

    context_text = " ## ".join(top_texts)

    tokens_n = num_tokens_from_string(context_text)
    if tokens_n > MAX_TOKENS_PROMT:
        l = len(context_text) / (tokens_n / MAX_TOKENS_PROMT)
        context_text = context_text[:int(l)]

    return context_text

def build_prompt(query, csv_f="new_train_data.csv", npy_f="embeddings.npy"):
    context_text = get_topk_texts(query, csv_f, npy_f, top_k=TOP_K)

    return f"""
    Отвечай как профессиональный сотрудник банка, важно отвечать на основе контекста из базы знаний.
    Если у тебя недостаточно информации и речь идёт не о общеизвестных вещах, то вежливо откажись от ответа.

    Инструкции:
    - Отвечай строго на поставленный вопрос, без вводных слов
    - Ответ должен быть монолитным, без пунктов
    - Избегай жаргона
    - Приводи примеры, если тема сложная (1–2 максимум)
    - Если не можешь ответить: "Извините, но сейчас я не могу дать ответ на ваш вопрос"

    Контекст из базы знаний:
    {context_text}

    Вопрос клиента: {query}
    """



if __name__ == "__main__":
    # query = "Как изменение ключевой ставки влияет на цену облигации с фиксированной ставкой?"
    # query = "Как опционные стратегии утилизируют деривативный лимит?"
    query = "Какие лимиты действуют на выплаты клиентам МФО по сравнению с гарантиями для клиентов НПФ?"

    sys_prompt = build_prompt(query)
    print(answer_generation(sys_prompt))
