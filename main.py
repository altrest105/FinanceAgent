import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")


def answer_generation(question):
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY,
    )
    response = client.chat.completions.create(
        model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Ответь на вопрос: {question}"
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":    
    questions = pd.read_csv('./questions.csv')
    questions_list = questions['Вопрос'].tolist()
    answer_list = []
    for current_question in tqdm(questions_list, desc="Генерация ответов"):
        answer = answer_generation(question=current_question)
        answer_list.append(answer)
    questions['Ответы на вопрос'] = answer_list
    questions.to_csv('submission.csv', index=False)