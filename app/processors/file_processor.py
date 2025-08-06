from __future__ import annotations
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, List

import pandas as pd
import streamlit as st  # только для типизации

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from parsers.csv_parser import CSVParser
from utils.logger import setup_logger

logger = setup_logger("file_processor")


class FileProcessor:
    """
    Принимает UploadedFile, извлекает таблицы и возвращает словарь с:
    - tables_count
    - output_filename
    - data (Excel bytes)
    - ocr_config (если применялся OCR-тюнинг)
    - ocr_score (если применялся OCR-тюнинг)
    """

    def __init__(self) -> None:
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": DOCXParser(),
            ".doc": DOCXParser(),
            ".csv": CSVParser(),
        }

    def process_file(
        self,
        uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    ) -> dict[str, Any]:
        file_name = getattr(uploaded_file, "name", None) \
                    or getattr(uploaded_file, "filename", None)
        if not file_name:
            raise AttributeError("Uploaded file has no name")

        suffix = Path(file_name).suffix.lower()
        if suffix not in self.parsers:
            raise ValueError(f"Unsupported file type: {suffix}")

        tmp_path = Path("/tmp") / file_name
        tmp_path.write_bytes(uploaded_file.read())

        parser = self.parsers[suffix]
        tables = parser.extract_tables(tmp_path)
        tables_cnt = len(tables)
        logger.info("Файл %s → извлечено %s таблиц", file_name, tables_cnt)

        # если нет таблиц
        if tables_cnt == 0:
            return {
                "tables_count": 0,
                "output_filename": None,
                "data": None,
            }

        # извлекаем OCR-метаданные из первой таблицы (если есть)
        ocr_config = tables[0].get("ocr_config")
        ocr_score  = tables[0].get("ocr_score")

        excel_bytes = self._create_excel(tables, file_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{Path(file_name).stem}_tables_{ts}.xlsx"

        return {
            "tables_count": tables_cnt,
            "output_filename": output_name,
            "data": excel_bytes,
            "ocr_config": ocr_config,
            "ocr_score": ocr_score,
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
