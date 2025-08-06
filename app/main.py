import streamlit as st
from datetime import datetime
from processors.file_processor import FileProcessor
from utils.file_handler import FileHandler
from utils.error_handler import ErrorHandler
from config.settings import settings


def main() -> None:
    st.set_page_config(
        page_title="Парсер медицинских таблиц",
        page_icon="📊",
        layout="wide",
    )
    st.title("🔬 Парсер медицинских таблиц")

    # Загрузка файлов
    uploaded_files = st.file_uploader(
        "Выберите файлы для обработки",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    file_processor = FileProcessor()
    for uploaded_file in uploaded_files:
        try:
            # Сохраняем во временную папку
            _ = FileHandler.save_uploaded_file(uploaded_file)

            # Обрабатываем файл
            result = file_processor.process_file(uploaded_file)

            # Если таблиц нет
            if result["tables_count"] == 0:
                st.warning(f"{uploaded_file.name}: таблиц не найдено")
                continue

            # Успешный результат
            st.success(f"{uploaded_file.name}: извлечено {result['tables_count']} таблиц")

            # Вывод OCR-настроек, если применялся тюнинг
            if result.get("ocr_config"):
                st.info("🏷 Победная OCR-конфигурация:")
                st.code(result["ocr_config"])
                st.metric("OCR score", f"{result['ocr_score']:.2%}")

            # Кнопка для скачивания Excel
            st.download_button(
                label=f"Скачать {result['output_filename']}",
                data=result["data"],
                file_name=result["output_filename"],
                mime=(
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
            )

        except Exception as e:
            error_details = ErrorHandler.handle_error(
                e,
                {"filename": uploaded_file.name},
            )
            st.error(f"{uploaded_file.name}: ошибка {error_details['error_message']}")


if __name__ == "__main__":
    main()
