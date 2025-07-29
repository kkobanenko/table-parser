import streamlit as st
import pandas as pd
from datetime import datetime
import time
import os

# Импорты из локальных модулей
from config.settings import settings
from processors.file_processor import FileProcessor
from utils.file_handler import FileHandler
from utils.error_handler import ErrorHandler
from config.pdf_parser_settings import PDFParserSettings


def main():
    """
    Главная функция Streamlit приложения
    """
    st.set_page_config(
        page_title="Парсер медицинских таблиц",
        page_icon="📊",
        layout="wide"
    )

    # Заголовок приложения
    st.title("🔬 Парсер медицинских таблиц")

    # Sidebar с информацией
    with st.sidebar:
        st.header("🛠 Настройки")

        # Секция PDF настроек
        with st.expander("PDF Парсер"):
            pdf_settings = PDFParserSettings.get_streamlit_settings_ui()

        # Общие настройки
        st.metric("Макс. размер файла", f"{settings.MAX_FILE_SIZE_MB} МБ")
        st.metric("Макс. файлов за раз", settings.MAX_FILES_PER_BATCH)

    # Основной контент
    st.header("📤 Загрузка файлов")

    # Загрузка файлов
    uploaded_files = st.file_uploader(
        "Выберите файлы для обработки",
        type=['pdf', 'docx', 'csv', 'txt'],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Валидация количества файлов
        if len(uploaded_files) > settings.MAX_FILES_PER_BATCH:
            st.error(f"Можно загрузить не более {settings.MAX_FILES_PER_BATCH} файлов")
            return

        # Прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Создание контейнеров для результатов
        results_container = st.container()
        download_container = st.container()

        # Обработка файлов
        file_processor = FileProcessor()
        processed_files = []
        errors = []

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Обновление прогресса
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Обработка: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")

                # Сохранение файла
                temp_file_path = FileHandler.save_uploaded_file(uploaded_file)

                # Обработка файла
                result = file_processor.process_file(uploaded_file)
                result['original_filename'] = uploaded_file.name
                processed_files.append(result)

            except Exception as e:
                error_details = ErrorHandler.handle_error(e, {
                    'filename': uploaded_file.name
                })
                errors.append(error_details)

        # Очистка статусов
        progress_bar.empty()
        status_text.empty()

        # Отображение результатов
        with results_container:
            st.header("📊 Результаты обработки")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Успешно обработано", len(processed_files))
            with col2:
                st.metric("❌ Ошибок", len(errors))
            with col3:
                st.metric("📄 Найдено таблиц",
                          sum(result.get('tables_count', 0) for result in processed_files)
                          )

        # Кнопки скачивания
        with download_container:
            st.header("📥 Результаты")
            for result in processed_files:
                st.download_button(
                    label=f"Скачать: {result['output_filename']}",
                    data=result['data'],
                    file_name=result['output_filename'],
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

        # Показ ошибок
        if errors:
            with st.expander("⚠️ Детали ошибок"):
                for error in errors:
                    st.error(f"Файл: {error.get('context', {}).get('filename')}")
                    st.json(error)


def error_handler(func):
    """
    Декоратор для обработки ошибок в Streamlit
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
            ErrorHandler.handle_error(e)

    return wrapper


@error_handler
def run():
    main()


if __name__ == "__main__":
    run()