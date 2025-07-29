# Базовый образ
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY app/ .

# Создание необходимых директорий
RUN mkdir -p /app/temp /app/logs /app/screenshots

# Порт для Streamlit
EXPOSE 8501

# Команда запуска
CMD ["streamlit", "run", "main.py"]