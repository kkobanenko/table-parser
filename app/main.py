import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

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
        
        # Векторные методы
        st.subheader("🔍 Векторные методы")
        use_camelot = st.checkbox("Использовать Camelot", value=False, 
                                help="Извлечение таблиц на основе векторных данных PDF")
        use_tabula = st.checkbox("Использовать Tabula", value=False,
                               help="Извлечение таблиц с помощью Java-библиотеки")
        
        # OCR методы
        st.subheader("📝 OCR методы")
        use_ocr = st.checkbox("Использовать OCR", value=False,
                             help="Распознавание текста с помощью Tesseract")
        use_cell = st.checkbox("Использовать Cell-segmentation", value=False,
                              help="Поиск границ ячеек с помощью компьютерного зрения")
        
        # Новые методы парсинга
        st.subheader("🚀 Новые методы")
        use_text_structure = st.checkbox("Использовать Text Structure Parser", value=False,
                                        help="Анализ структуры текста для определения таблиц")
        use_spacing = st.checkbox("Использовать Spacing Parser", value=False,
                                 help="Анализ пробелов между словами")
        use_easyocr = st.checkbox("Использовать EasyOCR Parser", value=False,
                                 help="Распознавание текста с координатами с помощью EasyOCR")
        use_spacing_analysis = st.checkbox("Использовать Spacing Analysis Parser", value=False,
                                          help="Анализ выравнивания текста для определения колонок")
        use_markitdown = st.checkbox("Использовать MarkItDown Parser", value=False,
                                    help="Конвертация файлов в Markdown с сохранением структуры таблиц")
        use_paddleocr = st.checkbox("Использовать PaddleOCR PP-OCRv5", value=False,
                                   help="Распознавание текста с помощью PaddleOCR PP-OCRv5 (высокая точность)")
        use_doctr = st.checkbox("Использовать DocTR (Mindee)", value=False,
                               help="OCR и анализ документов с помощью DocTR от Mindee")
        use_layoutparser = st.checkbox("Использовать LayoutParser", value=False,
                                      help="Анализ структуры документов и извлечение таблиц с помощью LayoutParser")
        
        # Комплексный пайплайн
        st.subheader("🔄 Комплексный пайплайн")
        use_pipeline = st.checkbox("Использовать комплексный пайплайн", value=False,
                                  help="Полный пайплайн: PDF→Image→Preprocessing→Layout→OCR→Tables→Export")
        
        if use_pipeline:
            st.write("**Настройки пайплайна:**")
            
            # Предобработка
            st.write("📄 **Предобработка:**")
            enable_preprocessing = st.checkbox("Включить предобработку", value=True)
            if enable_preprocessing:
                enable_deskew = st.checkbox("Выравнивание (deskew)", value=True)
                enable_denoise = st.checkbox("Удаление шума (denoise)", value=True)
                enable_binarize = st.checkbox("Бинаризация", value=False)
            else:
                enable_deskew = enable_denoise = enable_binarize = False
            
            # Layout детекция
            st.write("🎯 **Layout детекция:**")
            enable_layout_detection = st.checkbox("Включить детекцию layout", value=True)
            if enable_layout_detection:
                layout_model = st.selectbox("Модель layout", 
                                          ["lp://EfficientDete/PubLayNet", 
                                           "lp://PubLayNet/faster_rcnn_R_50_FPN_3x"],
                                          index=0)
            else:
                layout_model = "lp://EfficientDete/PubLayNet"
            
            # OCR
            st.write("🔍 **OCR:**")
            enable_ocr = st.checkbox("Включить OCR", value=True)
            if enable_ocr:
                ocr_method = st.selectbox("OCR метод", 
                                       ["paddleocr", "doctr", "tesseract"],
                                       index=0,
                                       help="PaddleOCR PP-OCRv5 рекомендуется для русского языка")
            else:
                ocr_method = "paddleocr"
            
            # Детекция таблиц
            st.write("📊 **Детекция таблиц:**")
            enable_table_detection = st.checkbox("Включить детекцию таблиц", value=True)
            
            # Экспорт
            st.write("💾 **Экспорт:**")
            enable_export = st.checkbox("Включить экспорт", value=True)
            if enable_export:
                export_formats = st.multiselect("Форматы экспорта",
                                             ["json", "excel", "csv", "txt"],
                                             default=["json", "excel"],
                                             help="JSON - метаданные, Excel/CSV - таблицы, TXT - текст")
            else:
                export_formats = ["json"]
        else:
            # Если пайплайн не используется, устанавливаем значения по умолчанию
            enable_preprocessing = enable_deskew = enable_denoise = enable_binarize = False
            enable_layout_detection = False
            layout_model = "lp://EfficientDete/PubLayNet"
            enable_ocr = False
            ocr_method = "paddleocr"
            enable_table_detection = False
            enable_export = False
            export_formats = ["json"]
        
        # Общие настройки
        st.subheader("⚙️ Общие настройки")
        check_rotations = st.checkbox("Исследовать повороты документа", value=False, 
                                    help="Если отключено, документ обрабатывается только в исходной ориентации")

        st.subheader("📄 Страницы для обработки")
        pages_option = st.radio(
            "Выберите страницы",
            options=["Все", "Интервал", "Конкретные страницы"],
            index=None,
            help="Выберите способ указания страниц для обработки"
        )
        
        if pages_option is None:
            pages = "all"  # Значение по умолчанию
            st.info("📋 Выберите способ указания страниц")
        elif pages_option == "Все":
            pages = "all"
            st.info("📋 Будут обработаны все страницы документа")
        elif pages_option == "Интервал":
            col1, col2 = st.columns(2)
            with col1:
                start = st.number_input(
                    "Страница с", min_value=1, step=1, value=1, key="page_start",
                    help="Начальная страница интервала"
                )
            with col2:
                end = st.number_input(
                    "Страница по", min_value=start, step=1, value=start, key="page_end",
                    help="Конечная страница интервала"
                )
            if start == end:
                pages = str(int(start))
                st.info(f"📋 Будет обработана страница {int(start)}")
            else:
                pages = f"{int(start)}-{int(end)}"
                st.info(f"📋 Будут обработаны страницы с {int(start)} по {int(end)}")
        else:  # Конкретные страницы
            pages_input = st.text_input(
                "Номера страниц",
                value="1",
                help="Введите номера страниц через запятую (например: 1,3,5 или 1,3-5)",
                placeholder="1,3,5"
            )
            pages = pages_input.strip() if pages_input.strip() else "1"
            if pages_input.strip():
                st.info(f"📋 Будут обработаны страницы: {pages}")
            else:
                st.warning("⚠️ Укажите номера страниц")

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
        "use_text_structure": use_text_structure,
        "use_spacing": use_spacing,
        "use_easyocr": use_easyocr,
        "use_spacing_analysis": use_spacing_analysis,
        "use_markitdown": use_markitdown,
        "use_paddleocr": use_paddleocr,
        "use_doctr": use_doctr,
        "use_layoutparser": use_layoutparser,
        "check_rotations": check_rotations,
        "pages": pages,
        # Настройки комплексного пайплайна
        "use_pipeline": use_pipeline,
        "enable_preprocessing": enable_preprocessing,
        "enable_deskew": enable_deskew,
        "enable_denoise": enable_denoise,
        "enable_binarize": enable_binarize,
        "enable_layout_detection": enable_layout_detection,
        "layout_model": layout_model,
        "enable_ocr": enable_ocr,
        "ocr_method": ocr_method,
        "enable_table_detection": enable_table_detection,
        "enable_export": enable_export,
        "export_formats": export_formats
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
