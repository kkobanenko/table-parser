# -------------------------------
# 1. Базовый образ
# -------------------------------
FROM python:3.11-slim

# -------------------------------
# 2. Установка системных зависимостей
# -------------------------------
#   - build-essential: для компиляции некоторых Python-зависимостей
#   - poppler-utils: pdf2image (конвертация PDF в PNG)
#   - ghostscript: поддержка Camelot (lattice mode)
#   - tesseract-ocr: OCR (английский и русский)
#   - libgl1/libglib2.0-0: для OpenCV
#   - openjdk-17-jre-headless: для Tabula (требуется Java)
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    ghostscript \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    openjdk-17-jre-headless \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 3. Настройка переменных окружения
# -------------------------------
ENV PYTHONUNBUFFERED=1
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# -------------------------------
# 4. Настройка рабочей директории
# -------------------------------
WORKDIR /app

# -------------------------------
# 5. Копирование зависимостей и установка Python-пакетов
# -------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 6. Копирование проекта
# -------------------------------
COPY . /app

# -------------------------------
# 7. Создание рабочих папок
# -------------------------------
RUN mkdir -p /app/temp /app/logs /app/screenshots

# -------------------------------
# 8. Открываем порт для Streamlit
# -------------------------------
EXPOSE 8501

# -------------------------------
# 9. Команда запуска приложения
# -------------------------------
# После исправления docker-compose volumes main.py лежит в /app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
