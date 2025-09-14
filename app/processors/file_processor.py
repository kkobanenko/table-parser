from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import streamlit as st  # only for typing
from utils.logger import setup_logger
from utils.file_handler import FileHandler
from utils.error_handler import ErrorHandler

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from parsers.csv_parser import CSVParser

# Импорт пайплайна
try:
    from pipeline.document_pipeline import DocumentPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

logger = setup_logger("file_processor")


class FileProcessor:
    """
    Принимает Streamlit-объект UploadedFile, извлекает таблицы и
    возвращает словарь-результат. Если таблиц нет, Excel-файл не
    создаётся.
    """

    def __init__(self) -> None:
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": DOCXParser(),
            ".doc": DOCXParser(),
            ".csv": CSVParser(),
            ".txt": CSVParser(),
        }

    def process_file(
        self,
        uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",  # type: ignore
        pdf_options: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        file_name = getattr(uploaded_file, "name", None) or getattr(
            uploaded_file, "filename", None
        )
        if not file_name:
            raise AttributeError("Uploaded file has no .name/.filename")

        suffix = Path(file_name).suffix.lower()
        if suffix not in self.parsers:
            raise ValueError(f"Unsupported file type: {suffix}")

        # write to temp for parser
        tmp_path = Path("/tmp") / file_name
        tmp_path.write_bytes(uploaded_file.read())

        parser = self.parsers[suffix]
        try:
            # PDF gets passed options
            if suffix == ".pdf":
                # Проверяем, нужно ли использовать комплексный пайплайн
                if (pdf_options and pdf_options.get("use_pipeline", False) and 
                    PIPELINE_AVAILABLE):
                    
                    logger.info(f"🔄 Используем комплексный пайплайн для {file_name}")
                    
                    # Инициализируем пайплайн
                    pipeline = DocumentPipeline()
                    pipeline.configure_pipeline(pdf_options)
                    pipeline.initialize_components()
                    
                    # Обрабатываем документ через пайплайн
                    results = pipeline.process_document(str(tmp_path))
                    
                    # Конвертируем результаты в формат, ожидаемый UI
                    tables = results.get('tables', [])
                    
                    # Добавляем информацию о пайплайне в результат
                    pipeline_info = {
                        'pipeline_used': True,
                        'pages_processed': len(results.get('pages', [])),
                        'layout_regions_found': len(results.get('layout_regions', [])),
                        'export_files': results.get('export_files', [])
                    }
                    
                    logger.info(f"✅ Пайплайн завершен: {len(tables)} таблиц, {pipeline_info['pages_processed']} страниц")
                    
                else:
                    # Обычная обработка PDF
                    tables = parser.extract_tables(tmp_path, pdf_options or {})
            else:
                tables = parser.extract_tables(tmp_path)
        except Exception as e:
            err = ErrorHandler.handle_error(e, {"filename": file_name})
            logger.error("Error parsing %s: %s", file_name, err["error_message"])
            return {"tables_count": 0, "output_filename": None, "data": None}

        tables_cnt = len(tables)
        logger.info("Файл %s → извлечено %s таблиц", file_name, tables_cnt)

        if tables_cnt == 0:
            return {"tables_count": 0, "output_filename": None, "data": None}

        # build Excel
        excel_bytes = self._create_excel(tables, file_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{Path(file_name).stem}_tables_{ts}.xlsx"

        return {
            "tables_count": tables_cnt,
            "output_filename": output_name,
            "data": excel_bytes,
        }

    @staticmethod
    def _sanitize_sheet_name(name: str, used: set[str]) -> str:
        name = name[:31] or "Sheet"
        base = name
        i = 1
        while name in used:
            suffix = f"_{i}"
            name = (base[:31 - len(suffix)] + suffix)[:31]
            i += 1
        used.add(name)
        return name

    def _create_excel(self, tables: List[dict[str, Any]], src_name: str) -> bytes:
        buffer = BytesIO()
        used_names: set[str] = set()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for idx, tbl in enumerate(tables, start=1):
                df: pd.DataFrame = tbl["data"]
                sheet = self._sanitize_sheet_name(tbl.get("sheet_name") or f"Table_{idx}", used_names)
                df.to_excel(writer, sheet_name=sheet, index=False)
        buffer.seek(0)
        return buffer.getvalue()
