import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from retriever import num_tokens_from_string

tqdm.pandas()
load_dotenv()

EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
SIMILARITY_THRESHOLD = 0.90
MAX_TOKENS = 8000

STOP_WORDS = [
    "в данной статье", "рассказываем", "например", "также", "кстати", 
    "следует отметить", "важно", "необходимо", "обратите внимание"
]

def shorten_text(text: str) -> str:
    for word in STOP_WORDS:
        text = text.replace(word, "")
    text = " ".join(text.split())
    tokens_n = num_tokens_from_string(text)
    if tokens_n > MAX_TOKENS:
        l = len(text) / (tokens_n / 8000)
        text = text[:int(l)]
    return text

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^а-яa-z0-9.,:;!?()\n\s-]", "", text)
    return text.strip()

def split_by_sections(text: str) -> list[str]:
    if pd.isna(text) or not text.strip():
        return []
    parts = [t.strip() for t in text.split("##") if t.strip()]
    return parts

def get_embedding(client: OpenAI, text: str):
    text = shorten_text(text)
    if not text:
        return None
    try:
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Ошибка при эмбеддинге: {e}")
        return None

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    required_cols = {"annotation", "tags", "text"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV должен содержать колонки 'annotation', 'tags', 'text'")
    df["annotation"] = df["annotation"].astype(str).progress_apply(clean_text)
    df["tags"] = df["tags"].astype(str).progress_apply(clean_text)
    df["text"] = df["text"].astype(str).progress_apply(clean_text)
    df = df[(df["annotation"].str.len() > 0) | (df["text"].str.len() > 0)].reset_index(drop=True)
    return df

def expand_into_chunks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        chunks = split_by_sections(row["text"])
        for i, chunk in enumerate(chunks):
            rows.append({
                "annotation": row["annotation"],
                "tags": row["tags"],
                "text_chunk": chunk,
                "chunk_id": i
            })
    return pd.DataFrame(rows)

def add_embeddings(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    df["embedding"] = df["text_chunk"].progress_apply(lambda t: get_embedding(client, t))
    return df[df["embedding"].notnull()].reset_index(drop=True)

def remove_similar_rows(df: pd.DataFrame, threshold: float = SIMILARITY_THRESHOLD) -> pd.DataFrame:
    embeddings = np.array(df["embedding"].tolist())
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    to_drop = set()
    for i in range(len(similarity_matrix)):
        if i in to_drop:
            continue
        duplicates = np.where(similarity_matrix[i] > threshold)[0]
        to_drop.update(duplicates.tolist())
    return df.drop(df.index[list(to_drop)]).reset_index(drop=True)

def save_data(df: pd.DataFrame, output_csv: str, embeddings_file: str):
    np.save(embeddings_file, np.array(df["embedding"].tolist()))
    df.drop(columns=["embedding"]).to_csv(output_csv, index=False, encoding="utf-8")
    print(f"База сохранена в {output_csv}, эмбеддинги — в {embeddings_file}. Строк осталось: {len(df)}")

def process_knowledge_base(input_file: str, output_csv: str, embeddings_file: str):
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )

    df = load_and_clean_data(input_file)
    df_chunks = expand_into_chunks(df)
    df_emb = add_embeddings(df_chunks, client)
    df_final = remove_similar_rows(df_emb)
    save_data(df_final, output_csv, embeddings_file)

if __name__ == "__main__":
    process_knowledge_base(
        input_file="train_data.csv",
        output_csv="new_train_data.csv",
        embeddings_file="embeddings.npy"
    )
