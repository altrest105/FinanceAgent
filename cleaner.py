import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

tqdm.pandas()
load_dotenv()

EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
SIMILARITY_THRESHOLD = 0.90
MAX_TOKENS = 7000

STOP_WORDS = [
    "в данной статье", "рассказываем", "например", "также", "кстати", 
    "следует отметить", "важно", "необходимо", "обратите внимание"
]


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.replace("##", "\n")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^а-яa-z0-9.,:;!?()\n\s-]", "", text)
    return text.strip()

def shorten_text(text: str) -> str:
    for word in STOP_WORDS:
        text = text.replace(word, "")
    text = " ".join(text.split())
    max_chars = MAX_TOKENS * 4
    if len(text) > max_chars:
        text = text[:max_chars]
    return text

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

def get_embedding(client: OpenAI, text: str):
    text = shorten_text(text)
    if not text:
        return None
    try:
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def add_embeddings(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    texts = (df["annotation"] + " " + df["tags"] + " " + df["text"]).tolist()
    df["embedding"] = [get_embedding(client, t) for t in tqdm(texts)]
    return df[df["embedding"].notnull()].reset_index(drop=True)

def remove_similar_rows(df: pd.DataFrame, threshold: float = SIMILARITY_THRESHOLD) -> pd.DataFrame:
    embeddings = np.array(df["embedding"].tolist())
    from sklearn.metrics.pairwise import cosine_similarity
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
    print(f"✅ База сохранена в {output_csv}, эмбеддинги в {embeddings_file}, строк осталось: {len(df)}")

def process_knowledge_base(input_file: str, output_csv: str, embeddings_file: str):
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )

    df = load_and_clean_data(input_file)
    df = add_embeddings(df, client)
    df = remove_similar_rows(df)
    save_data(df, output_csv, embeddings_file)

if __name__ == "__main__":
    process_knowledge_base(
    input_file="train_data.csv",
    output_csv="new_train_data.csv",
    embeddings_file="embeddings.npy"
)