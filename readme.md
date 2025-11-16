# 📊 RaifRAG - AI-Powered Financial Advisor

> AI-агент по инвестициям с интеграцией Tinkoff Invest API, RAG-системой и персонализацией профиля инвестора

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Django](https://img.shields.io/badge/Django-5.2-green.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)

---

## 🎯 Возможности

### 🤖 AI-Агент
- **RAG (Retrieval-Augmented Generation)** — ответы на основе базы знаний о финансах, инвестициях, праве и госуслугах
- **Классификация намерений** — автоматическое определение типа запроса (торговля, информация о ценных бумагах, общие вопросы)
- **Контекстная память** — система запоминает историю диалогов и учитывает предыдущий контекст

### 💼 Tinkoff Invest API Integration
- **Управление портфелем** — просмотр позиций, стоимости активов, доходности
- **Торговые операции** — покупка/продажа ценных бумаг через команды
- **Рыночная информация** — актуальные котировки, свечные графики, стаканы заявок

### 👤 Персонализация
- **Автоматический анализ профиля** — определение риск-профиля на основе диалогов (каждые 10 сообщений)
- **Инвестиционные цели** — отслеживание целей (сохранение капитала, рост, спекуляция, пенсия)
- **Шифрование токенов** — безопасное хранение API-токенов

### 📚 База знаний
- **Векторный поиск** — FAISS для быстрого семантического поиска
- **Реранкинг** — улучшение релевантности результатов с помощью BGE-reranker
- **Эмбеддинги** — многоязычная модель BGE-M3

---

## 🏗️ Архитектура

### Backend (Django)
```
backend/
├── agent/                     # Основное приложение
│   ├── models.py              # БД модели (профиль, история диалогов)
│   ├── views.py               # API endpoints
│   ├── orchestrator.py        # Главный оркестратор запросов
│   ├── services/
│   │   ├── memorybank.py      # Управление профилем и историей
│   │   ├── rag.py             # RAG-система (эмбеддинги, поиск)
│   │   └── tinvest.py         # Tinkoff Invest API
│   └── management/
│       └── commands/
│           └── init_demo_user.py  # Создание демо-пользователя
├── files/
│   ├── train_data.csv         # Исходная база знаний
│   └── hf_cache/              # Кэш HuggingFace моделей
└── backend/
    └── settings.py            # Конфигурация Django
```

### Frontend (React + TypeScript)
```
frontend/
├── src/
│   ├── App.tsx               # Главный компонент чата
│   ├── App.css               # Стили
│   └── assets/               # Иконки и изображения
└── vite.config.ts            # Конфигурация Vite
```

### Поток обработки запроса

```
           ┌─────────────┐
           │   Вопрос    │
           │пользователя │
           └──────┬──────┘
                  │
                  ▼
       ┌─────────────────────────────┐
       │  1. Определение намерения   │
       │  (TRADE/SECURITIES/GENERAL) │
       └──────────┬──────────────────┘
                  │
    ┌──────┴───┴──┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌──────┐    ┌──────────┐   ┌─────────┐
│TRADE │    │SECURITIES│   │ GENERAL │
└──┬───┘    └────┬─────┘   └────┬────┘
   │             │              │
   │     ┌───────┴────────┐     │
   │     │ Tinkoff API    │     │
   │     │                │     │
   │     └───────┬────────┘     │
   │             │              │
   └─────────────┼──────────────┘
                 ▼
         ┌───────────────┐
         │      LLM      │
         │   Генерация   │
         └───────┬───────┘
                 ▼
         ┌───────────────┐
         │    Ответ +    │
         │  Сохранение   │
         │   в историю   │
         └───────────────┘
```

---

## 🛠️ Технологический стек

### Backend
- **Django** — веб-фреймворк
- **Django REST Framework** — API
- **ADRF** — асинхронные view-функции
- **SQLite** — база данных
- **Cryptography** — шифрование токенов

### AI/ML
- **OpenAI API** — генерация ответов (через OpenRouter)
- **Sentence Transformers** — эмбеддинги (BAAI/bge-m3)
- **CrossEncoder** — реранкинг (BAAI/bge-reranker-v2-m3)
- **FAISS** — векторный поиск

### Tinkoff Integration
- **tinkoff-investments** — T-Investments API

### Frontend
- **React 18** — UI библиотека
- **TypeScript** — типизация
- **Vite** — сборщик
- **Axios** — HTTP-клиент
- **React Markdown** — отображение Markdown
- **Framer Motion** — анимации

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/altrest105/FinanceAgent.git
cd FinanceAgent
```

### 2. Настройка Backend

#### Создайте `.env` файл в корне проекта:

```env
SECRET_KEY="django-insecure-$g&!jkxrz!z35)u$ip(8rvti!f%tk3x_(eb)gm)-x)frp)9e2b"
ENCRYPTION_KEY="zaizwddTH69_9txJ8i5XHj9rEZHWFnWLH27LNjYUfkk="
AGENT_LOG_LEVEL="DEBUG"
DEBUG="True"

LLM_API_KEY="sk-or-v1-********"
LLM_BASE_URL="https://openrouter.ai/api/v1"
LLM_MODEL="openai/gpt-5.1"
LLM_MAX_TOKENS="8000"

EMBEDDER_MODEL="BAAI/bge-m3"
RERANKER_MODEL="BAAI/bge-reranker-v2-m3"

TOP_K="3"
TOP_K_BEFORE_RERANK="20"
BATCH_SIZE="100"
USE_RAG="0"
```

#### Установите зависимости:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Сделайте миграции:

```bash
python manage.py makemigrations
python manage.py migrate
```

#### Запустите сервер:

```bash
python manage.py runserver --noreload
```

Backend будет доступен по адресу: `http://127.0.0.1:8000`

---

### 3️⃣ Настройка Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend будет доступен по адресу: `http://localhost:5173`

---

## 📝 Примеры использования

### Торговые операции

```
Пользователь: Купи 10 акций Сбербанка
Агент: ✅ Заявка на покупку SBER успешно размещена!
     - Лотов: 10
     - Цена исполнения: 280.50 ₽
     - Общая сумма: 2,805.00 ₽
```

### Информация о ценных бумагах

```
Пользователь: Что происходит с акциями Газпрома?
Агент: ### 📊 Газпром (GAZP)

#### 1. **Текущая цена**
- Последняя цена: 175.30 ₽ (-2.5% за день)

#### 2. **Объём торгов**
- Сегодня: 15.2 млн шт.

#### 3. **Рекомендации**
На основе технического анализа...
```

### Общие финансовые вопросы

```
Пользователь: Что такое дивиденды?
Агент: ### Дивиденды

#### 1. **Определение**
Дивиденды — это часть прибыли компании, которая распределяется
между акционерами...

#### 2. **Виды дивидендов**
- Денежные
- Акциями

#### Полезные советы
- Обращайте внимание на дивидендную доходность
- Учитывайте налогообложение (13%)
```

---

## 🧪 API Endpoints

### Chat
**POST** `/api/chat/`
```json
{
  "question": "Купи 5 акций YNDX",
  "session_id": "uuid-session-id" // опционально
}
```

**Response:**
```json
{
  "session_id": "uuid-session-id",
  "question": "Купи 5 акций YNDX",
  "answer": "✅ Заявка размещена...",
  "intent": "trade",
  "success": true
}
```

### Token Management
**POST** `/api/token/save/`
```json
{
  "token": "t.xxxxxxxxxxxxxx",
  "account_id": "optional-account-id"
}
```

**GET** `/api/token/status/`
```json
{
  "has_token": true,
  "account_id": "2000000000"
}
```

### History
**GET** `/api/history/?limit=20`
```json
{
  "count": 15,
  "history": [
    {
      "id": 1,
      "question": "Что такое облигации?",
      "answer": "### Облигации...",
      "intent": "general",
      "timestamp": "2025-01-16T12:00:00Z"
    }
  ],
  "user": "demo_user"
}
```

---

## 📊 База данных

### Модель: `InvestorProfile`
- **risk_tolerance** — риск-профиль (conservative/moderate/aggressive)
- **risk_score** — оценка риска (0-10)
- **primary_goal** — основная цель (capital_preservation/stable_income/capital_growth/speculation/retirement)
- **investment_horizon_months** — горизонт инвестирования (месяцы)
- **tinkoff_token_encrypted** — зашифрованный токен
- **preferences** — JSON с дополнительными данными

### Модель: `Conversation`
- **question** — вопрос пользователя
- **answer** — ответ бота
- **intent** — намерение (trade/securities_info/general)
- **extracted_data** — JSON с извлечёнными данными
- **session_id** — ID сессии
