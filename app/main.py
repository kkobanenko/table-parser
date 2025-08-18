import streamlit as st
from datetime import datetime
from processors.file_processor import FileProcessor
from utils.file_handler import FileHandler
from utils.error_handler import ErrorHandler
from config.settings import settings

def main() -> None:
    st.set_page_config(page_title="Парсер медицинских таблиц", layout="wide")
    st.title("🔬 Парсер медицинских таблиц")

    # ─── боковая панель: опции PDF-парсинга ────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Параметры PDF-обработки")
        use_camelot = st.checkbox("Использовать Camelot", value=True)
        use_tabula = st.checkbox("Использовать Tabula", value=True)
        use_ocr = st.checkbox("Использовать OCR", value=True)
        use_cell = st.checkbox("Использовать Cell-segmentation", value=True)

        st.subheader("Страницы для обработки")
        pages_option = st.radio(
            "Выберите страницы",
            options=["Все", "Интервал"],
            index=0
        )
        if pages_option == "Все":
            pages = "all"
        else:
            start = st.number_input(
                "С", min_value=1, step=1, value=1, key="page_start"
            )
            end = st.number_input(
                "По", min_value=start, step=1, value=start, key="page_end"
            )
            pages = f"{int(start)}-{int(end)}"

    # загрузка файлов
    uploaded_files = st.file_uploader(
        "Выберите файлы для обработки",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        return

    file_processor = FileProcessor()
    pdf_options = {
        "use_camelot": use_camelot,
        "use_tabula": use_tabula,
        "use_ocr": use_ocr,
        "use_cell_table": use_cell,
        "pages": pages
    }

    for uploaded_file in uploaded_files:
        try:
            _ = FileHandler.save_uploaded_file(uploaded_file)
            # Передаём PDF-опции только для PDF
            result = file_processor.process_file(uploaded_file, pdf_options)

            if result["tables_count"] == 0:
                st.warning(f"{uploaded_file.name}: таблиц не найдено")
                continue

            st.success(f"{uploaded_file.name}: извлечено {result['tables_count']} таблиц")

            # Показать OCR-конфигурацию, если есть
            if result.get("ocr_config"):
                st.info("🏷 Победная OCR-конфигурация:")
                st.code(result["ocr_config"])
                st.metric("OCR score", f"{result['ocr_score']:.2%}")

            # Кнопка скачивания
            st.download_button(
                label=f"Скачать {result['output_filename']}",
                data=result["data"],
                file_name=result["output_filename"],
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

        except Exception as e:
            err = ErrorHandler.handle_error(e, {"filename": uploaded_file.name})
            st.error(f"{uploaded_file.name}: ошибка {err['error_message']}")

if __name__ == "__main__":
    main()
