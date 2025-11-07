import pandas as pd
import numpy as np
import os
import requests
import faiss
import time
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

TOP_K = 2 # Количество контекстов для генерации ответа
TOP_K_LOCAL = 4 # Количество контекстов для повторной генерации ответа
TOP_K_BEFORE_RERANK = 20 # Количество кандидатов для реранкера
LLM_MODEL = "openrouter/mistralai/mistral-small-3.2-24b-instruct" # Модель для генерации ответов openrouter/google/gemma-3-27b-it openrouter/mistralai/mistral-small-3.2-24b-instruct
EMBEDDING_MODEL = "text-embedding-3-small" # Модель для эмбеддингов
RERANK_MODEL = "deepinfra/Qwen/Qwen3-Reranker-4B" # Модель для реранкинга
BATCH_SIZE = 100  # Размер батча для получения эмбеддингов

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
    
    def get_topk_contexts(self, query, repeat, max_retries=3, retry_delay=1.5):
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
                )[:TOP_K_LOCAL]

                top_contexts = [candidate_texts[item["index"]] for item in sorted_results]
            
            else:
                print("⚠️ Реранкер не сработал после всех попыток → используем FAISS поиск")
                top_contexts = candidate_texts[:TOP_K_LOCAL]

        return top_contexts

# --- Генерация ответа ---
# Функция построения запроса к LLM для генерации ответа
def build_answer(question, repeat):
    contexts = retriever.get_topk_contexts(question, repeat = repeat)

    context_text = "\n\n---\n\n".join(contexts)
    
    print(context_text)

    text = f"""Ты профессиональный банковский консультант. Отвечай строго и только по приведённому контексту.

    Контекст:
    {context_text}

    Вопрос клиента: {question}

    Правила ответа:
    - Запрещено добавлять предположения
    - Запрещено использовать внешние знания
    - Запрещено придумывать детали
    - Отвечай строго на вопрос, без вводных слов
    - Ответ должен быть монолитным текстом, без списков
    - Приводи 1-2 примера для сложных тем
    - Если информации в контексте недостаточно — скажи: "Извините, но сейчас я не могу дать ответ на ваш вопрос"
    - Если ты дал ответ, но тебе он кажется не полным не пиши "Извините, но сейчас я не могу дать ответ на ваш вопрос"
    """

    return text

# Функция генерации ответа на вопрос
def answer_generation(question, client_ans, repeat = True):
    question_text = build_answer(question, repeat)

    response = client_ans.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question_text
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content


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
        pass
        # print("⚠️ База знаний уже существует, пропускаем создание")

    # Инициализация ретривера
    retriever = KnowledgeBaseRetriever(
        db_csv="new_train_data.csv",
        embeddings_file="embeddings.npy"
    )

    # Генерация и сохранение ответов на вопросы из questions.csv
    questions = pd.read_csv('./questions.csv')
    questions_list = questions['Вопрос'].tolist()
    answer_list = []

    client_ans = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY,
    )

    # for current_question in tqdm(questions_list, desc="Генерация ответов"):
    #     answer = answer_generation(question=current_question, client_ans=client_ans)
    #     answer_list.append(answer.replace('\n', ' ').strip())
    # questions['Ответы на вопрос'] = answer_list
    # questions.to_csv('submission.csv', index=False)
    # print("✅ Ответы сохранены в submission.csv")

    
    # # Mistral
    q = '152,Чем отличается порог выручки для обязательного приема карт «Мир» от максимальной суммы «закупки с полки» по 223‑ФЗ?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # Решён # q = '198,Как закон ограничивает штрафы за просрочку по кредитной карте по сравнению с микрозаймом?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '310,Почему массовая рассылка заявок снижает шансы на одобрение?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # Решён # q = '396,Каковы сроки подачи заявлений при ликвидации МФО по сравнению с обращением за возмещением при отзыве лицензии у НПФ?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'

    # Решён # q = '11,"Каковы сроки блокировки перевода, а также решения финомбудсмена?","Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # Решён # q = '60,Как различается защита денег при отзыве лицензии у банка: депозит против счета эскроу?,"При отзыве лицензии у банка депозиты защищены системой страхования вкладов в размере до 1,4 млн рублей по всем счетам и вкладам в одном банке, включая начисленные проценты. Например, если у клиента на депозите лежало 1,2 млн рублей, а банк начислил 50 тыс. рублей процентов, то общая сумма компенсации составит 1,25 млн рублей.  Счета эскроу, открытые для сделок с недвижимостью или долевого строительства, имеют повышенную защиту — до 10 млн рублей. Например, если на эскроу-счете лежали 8 млн рублей, то при отзыве лицензии у банка клиент получит всю сумму. Однако если у клиента в том же банке был дополнительный вклад на 1,4 млн рублей, то он получит компенсацию по нему отдельно, но общая выплата не превысит 10 млн рублей.  Извините, но сейчас я не могу дать ответ на ваш вопрос относительно аккредитивов, так как в контексте нет информации о их защите при отзыве лицензии."'
    # Решён # q = '69,Какие лимиты действуют для бесплатных переводов по СБП по сравнению с кэшаутом?,"В контексте СБП бесплатные переводы по номеру телефона ограничены 100 000 рублей в месяц на чужие номера и 30 млн рублей в месяц на свои счета в других банках. Кэшаут в контексте не упомянут, поэтому сравнить лимиты невозможно. Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # Решён # q = '145,Как соотносится лимит суммы по кредитным каникулам для потребительских займов с максимальным размером кредита по семейной ипотеке?,"В контексте указаны только лимиты для кредитных каникул: 450 000 рублей для потребительских займов и 1,6 млн рублей для автокредитов. Сведения о максимальном размере кредита по семейной ипотеке отсутствуют. Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # Решён # q = '197,Как отличается доступ к деньгам в ИСЖ по сравнению с продажей паев биржевых фондов?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # Решён # q = '245,Куда обращаться при споре со страховщиком по ИСЖ и по ОСАГО?,"При споре со страховщиком по ОСАГО направьте письменную претензию в страховую компанию, затем обратитесь к финансовому омбудсмену или в Банк России. По ИСЖ (ипотечному страхованию жилья) контекст не содержит информации, поэтому извините, но сейчас я не могу дать ответ на ваш вопрос."'

    # # Gemma
    # q = '152,Чем отличается порог выручки для обязательного приема карт «Мир» от максимальной суммы «закупки с полки» по 223‑ФЗ?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '198,Как закон ограничивает штрафы за просрочку по кредитной карте по сравнению с микрозаймом?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '310,Почему массовая рассылка заявок снижает шансы на одобрение?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '396,Каковы сроки подачи заявлений при ликвидации МФО по сравнению с обращением за возмещением при отзыве лицензии у НПФ?,"В контексте представлена информация о сроках подачи заявления на компенсацию при отзыве лицензии у НПФ: необходимо обратиться в АСВ в течение пяти месяцев со дня публикации информации об отзыве лицензии или начале банкротства фонда. Информации о сроках подачи заявлений при ликвидации МФО в контексте нет. Извините, но сейчас я не могу дать ответ на ваш вопрос."'

    # q = '6,Почему при высокой долговой нагрузке банк может отказать в новом кредите?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '46,Кому доступен залог до продажи: жильцам ипотечной квартиры против владельца автомобиля в автоломбарде?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '90,Почему инвестиции в себя считаются выгодными?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '102,Сколько максимально вернут по соцвычету и по взносам на ИИС при ставке НДФЛ 22%,"Максимальный размер вычета на взносы составляет 88 000 рублей = 400 000 х 22%. Извините, но сейчас я не могу дать ответ на ваш вопрос о максимальной сумме возврата по соцвычету."'
    # q = '231,Как обезопасить себя на случай утери векселя?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '249,Почему новичку опасно спекулировать деривативами?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '322,В каких случаях действует госстрахование вкладов?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '375,"Каковы последствия отзыва лицензии банка для долга по кредитке, а НПФ — для пенсионных накоплений?","В случае отзыва лицензии у НПФ размер вашей пенсии не изменится, ее накопительную часть будет выплачивать СФР. При отзыве лицензии у банка, если у него почти не осталось собственных средств, он не выдает деньги с вкладов, не проводит платежи клиентов, не сдает вовремя отчетность или нарушает законы, это может привести к проблемам у вкладчиков. Извините, но сейчас я не могу дать ответ на ваш вопрос относительно последствий отзыва лицензии банка для долга по кредитке."'
    # q = '379,Как определяется доходность по векселю по сравнению с ценой лота на аукционе замороженных активов?,"Доход по векселю определяется как разница между номиналом векселя и ценой его погашения, или в виде процента, указанного на самом векселе. Если номинал векселя Ивана Никитича составляет 100 000 рублей, а он заплатил за него 80 000 рублей, то его доход составит 20 000 рублей. Извините, но сейчас я не могу дать ответ на ваш вопрос о ценах на аукционе замороженных активов."'
    # q = '423,Покроет ли страховка ответственности кражу из банковской ячейки?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '462,Как соотносится автоматическое начисление налога на АУСН с порядком списания долгов со счетов ИП?,"В контексте не указано, как автоматическое начисление налога на АУСН соотносится с порядком списания долгов со счетов ИП. Извините, но сейчас я не могу дать ответ на ваш вопрос."'
    # q = '479,Как соотносится переплата по займам под 0% при просрочке с законным лимитом по краткосрочным кредитам?,"Извините, но сейчас я не могу дать ответ на ваш вопрос."'

    answer = answer_generation(question=q, client_ans=client_ans)
    if ("Извините, но сейчас я не могу дать ответ на ваш вопрос" in answer):
        answer = answer_generation(question=q, client_ans=client_ans, repeat = False)
        
    print("\n\n\n", answer)