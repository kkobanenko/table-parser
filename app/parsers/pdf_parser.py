"""
Расширенный PDF‑парсер с модульной стратегией:
1) Camelot (vector) → 2) pdfplumber (mixed) → 3) tabula (guess) →
4) OCR‑скан через pdf2image + ImageParser.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import camelot                       # pip install "camelot‑py[cv]"
import pdfplumber
import tabula
from pdf2image import convert_from_path
import pandas as pd

from config.pdf_parser_settings import PDFParserSettings
from parsers.image_parser import ImageParser
from utils.logger import setup_logger

__all__ = ["PDFParser"]

logger = setup_logger("pdf_parser")


class PDFParser:
    """Стратегический парсер PDF‑таблиц."""

    def __init__(self, settings: Optional[Dict] = None) -> None:
        self.settings = settings or PDFParserSettings.load_settings()
        self.settings.setdefault("dpi", 300)
        self.cache_dir = Path(self.settings["cache_dir"])
        self.image_parser = ImageParser(self.settings.get("ocr", {}))

    # -------------------------- public API ---------------------------------

    def extract_tables(self, file_path: str | Path) -> List[Dict]:
        strategies = (
            self._extract_with_camelot,
            self._extract_with_pdfplumber,
            self._extract_with_tabula,
        )

        tables: List[Dict] = []
        for strategy in strategies:
            try:
                tables.extend(strategy(file_path))
                if tables:
                    logger.info(
                        "📄 %s → %s таблиц (%s)",
                        Path(file_path).name,
                        len(tables),
                        strategy.__name__,
                    )
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s failed: %s", strategy.__name__, exc)

        if not tables:  # OCR fallback
            tables.extend(self._extract_scanned_pdf(file_path))

        return self._deduplicate_tables(tables)

    # ----------------------- strategy: Camelot -----------------------------

    def _extract_with_camelot(self, file_path: str | Path) -> List[Dict]:
        flavor = self.settings.get("camelot_flavor", "stream")
        pages = self.settings.get("pages", "all")

        # line_scale допустим только для lattice :contentReference[oaicite:0]{index=0}
        camelot_kwargs = {
            "pages": pages,
            "flavor": flavor,
            "strip_text": "\n",
        }
        if flavor == "lattice":
            camelot_kwargs["line_scale"] = self.settings.get("line_scale", 40)

        tables = camelot.read_pdf(str(file_path), **camelot_kwargs)

        return [
            {
                "data": table.df.replace("", pd.NA),
                "sheet_name": f"Camelot_{i+1}",
                "source": f"camelot_{flavor}",
            }
            for i, table in enumerate(tables)
            if table.shape[0] > 1 and table.shape[1] > 1
        ]

    # -------------------- strategy: pdfplumber -----------------------------

    def _extract_with_pdfplumber(self, file_path: str | Path) -> List[Dict]:
        stg = self.settings
        tables: List[Dict] = []

        table_settings = {
            "vertical_strategy": stg["vertical_strategy"],
            "horizontal_strategy": stg["horizontal_strategy"],
            "join_tolerance": stg["line_margin"],
            "snap_tolerance": stg["char_margin"],
        }  # параметры передаются через table_settings, а не kwargs :contentReference[oaicite:1]{index=1}

        with pdfplumber.open(file_path) as pdf, ThreadPoolExecutor() as pool:
            futures = [
                pool.submit(
                    self._plumber_page_tables,
                    page,
                    pno,
                    table_settings,
                )
                for pno, page in enumerate(pdf.pages, start=1)
            ]
            for fut in futures:
                tables.extend(fut.result())
        return tables

    @staticmethod
    def _plumber_page_tables(page, page_num: int, ts: Dict) -> List[Dict]:
        page_tables = page.extract_tables(table_settings=ts)
        return [
            {
                "data": pd.DataFrame(tbl[1:], columns=tbl[0]),
                "sheet_name": f"Plumber_p{page_num}_{i+1}",
                "source": "pdfplumber",
                "page": page_num,
            }
            for i, tbl in enumerate(page_tables)
            if tbl and len(tbl) > 1
        ]

    # ----------------------- strategy: tabula ------------------------------

    def _extract_with_tabula(self, file_path: str | Path) -> List[Dict]:
        try:
            dfs = tabula.read_pdf(
                str(file_path),
                pages="all",
                multiple_tables=True,
                guess=self.settings.get("guess", True),
            )
        except Exception as exc:   # JVM отсутствует
            logger.warning("Tabula skipped: %s", exc)
            return []

        return [
            {
                "data": df,
                "sheet_name": f"Tabula_{i+1}",
                "source": "tabula",
            }
            for i, df in enumerate(dfs)
            if not df.empty and df.notna().any().any()
        ]

    # ---------------------------- OCR fallback -----------------------------

    def _extract_scanned_pdf(self, file_path: str | Path) -> List[Dict]:
        dpi = self.settings["dpi"]
        images = convert_from_path(str(file_path), dpi=dpi)
        tables: List[Dict] = []

        for idx, img in enumerate(images):
            tmp_img = self.cache_dir / f"page_{idx}.png"
            img.save(tmp_img, "PNG")
            try:
                tables.extend(
                    [
                        t
                        for t in self.image_parser.extract_tables(tmp_img)
                        if not t["data"].empty
                    ]
                )
            finally:
                tmp_img.unlink(missing_ok=True)
        return tables

    # ----------------------------- utils -----------------------------------

    @staticmethod
    def _deduplicate_tables(tables: Sequence[Dict]) -> List[Dict]:
        unique, seen = [], set()
        for tbl in tables:
            h = hash(
                (tuple(tbl["data"].columns), tuple(map(tuple, tbl["data"].values)))
            )
            if h not in seen:
                seen.add(h)
                unique.append(tbl)
        logger.info("🧹 После дедупликации: %s таблиц", len(unique))
        return unique
