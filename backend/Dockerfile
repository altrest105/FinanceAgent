FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "\
    python manage.py makemigrations agent && \
    python manage.py migrate && \
    python manage.py init_demo_user && \
    python manage.py runserver 0.0.0.0:8000 --noreload \
"]
