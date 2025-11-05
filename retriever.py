import os
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

import tiktoken

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
TOP_K = 5
MAX_TOKENS_PROMT = 7000


# ------Checker------
def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def compute_query_embedding(query: str) -> np.ndarray:
    client = client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return np.array(response.data[0].embedding)

def get_topk_texts(query: str, db_csv: str, embeddings_file: str, top_k: int = TOP_K) -> str:
    df = pd.read_csv(db_csv)
    db_texts = (df['annotation'].fillna('') + " " + 
                df['tags'].fillna('') + " " + 
                df['text'].fillna('')).tolist()

    result_texts = df['text'].fillna('').tolist()

    db_embeddings = np.load(embeddings_file)
    query_emb = compute_query_embedding(query)

    similarities = cosine_similarity([query_emb], db_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    top_texts = [result_texts[i] for i in top_indices]

    context_text = "\n\n".join(top_texts)
    l = len(context_text) / (num_tokens_from_string(context_text) / MAX_TOKENS_PROMT)
    return context_text[:int(l)]

def build_prompt(query, csv_f = "new_train_data.csv", npy_f = "embeddings.npy"):
    context_text = get_topk_texts(query, csv_f, npy_f, top_k=TOP_K)

    return  f"""
    Отвечай как профессиональный сотрудник банка, важно отвечать па основе контекста из базы знаний.
    Если у тебя недостаточно информации и речь идёт не о общеизвестных вещах, то тебе необходимо вежливо отказаться отвечать.

    Инструкции:
    - Отвечать нужно строго на поставленный вопрос, без вводных слов и эмоционального окраса
    - Ответ должен быть монолитным, без пунктов, только текст
    - Избегай жаргонных слов, максимально понятные слова
    - Если тема сложная для объяснения приводи примеры (1 или максимум 2)
    - Если не можешь ответить: "Извините, но сейчас я не могу дать ответ на ваш вопрос"

    Контекст из базы знаний: 
    {context_text}
    
    Вопрос клиента: {query}
    """


if __name__ == "__main__":
    # query = "Как изменение ключевой ставки влияет на цену облигации с фиксированной ставкой?"
    query = "Как опционные стратегии утилизируют деривативный лимит?"
    
    sys_prompt = build_prompt(query)
