from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, List

import pandas as pd
import streamlit as st  # для типизации UploadedFile

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from parsers.csv_parser import CSVParser
from utils.logger import setup_logger

logger = setup_logger("file_processor")


class FileProcessor:
    """
    Координатор: принимает Streamlit‑объект UploadedFile,
    извлекает таблицы и возвращает словарь с результатами.
    """

    def __init__(self) -> None:
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": DOCXParser(),
            ".doc": DOCXParser(),
            ".csv": CSVParser(),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_file(
        self,
        uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",  # type: ignore
    ) -> dict[str, Any]:
        """
        Возвращает:
            {
                "data": <bytes Excel>,
                "output_filename": <имя файла для скачивания>,
                "tables_count": <int>
            }
        """
        file_name = getattr(uploaded_file, "name", None) or getattr(
            uploaded_file, "filename", None
        )
        if not file_name:
            raise AttributeError("Uploaded file has no .name/.filename")

        suffix = Path(file_name).suffix.lower()
        if suffix not in self.parsers:
            raise ValueError(f"Unsupported file type: {suffix}")

        # сохраняем во временный файл (нужно парсерам)
        tmp_path = Path("/tmp") / file_name
        tmp_path.write_bytes(uploaded_file.read())

        parser = self.parsers[suffix]
        tables = parser.extract_tables(tmp_path)

        logger.info("Файл %s → извлечено %s таблиц", file_name, len(tables))

        # Excel‑bytes
        excel_bytes = self._create_excel(tables, file_name)

        # имя выходного файла
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{Path(file_name).stem}_tables_{ts}.xlsx"

        return {
            "data": excel_bytes,
            "output_filename": output_name,
            "tables_count": len(tables),
        }

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sanitize_sheet_name(name: str, used: set[str]) -> str:
        """Делает имя листа уникальным и ≤ 31 символ (ограничение Excel)."""
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
            if tables:
                for idx, tbl in enumerate(tables, start=1):
                    df: pd.DataFrame = tbl["data"]
                    sheet = self._sanitize_sheet_name(
                        tbl.get("sheet_name") or f"Table_{idx}", used_names
                    )
                    df.to_excel(writer, sheet_name=sheet, index=False)
            else:
                msg = pd.DataFrame(
                    {
                        "Info": [
                            f"В файле «{src_name}» таблицы не обнаружены.",
                            "Проверьте тип документа и настройки парсера.",
                        ]
                    }
                )
                msg.to_excel(writer, sheet_name="Нет_таблиц", index=False)

        buffer.seek(0)
        return buffer.getvalue()
