from __future__ import annotations
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd
import camelot
import pdfplumber
import tabula
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter

from config.pdf_parser_settings import PDFParserSettings
from config.settings import settings as app_settings
from utils.logger import setup_logger
from utils.ocr_tuner import DEFAULT_OCR_CONFIGS, OCRTuner
from parsers.cell_table_parser import CellTableParser

logger = setup_logger("pdf_parser")


class PDFParser:
    """
    PDF-парсер с векторными, OCR- и cell-сегментационными стратегиями,
    плюс полный OCR каждой повернутой страницы.
    """

    def __init__(self, settings: Dict[str, Any] = None) -> None:
        self.settings = settings or PDFParserSettings.load_settings()
        self.settings.setdefault("dpi", 300)
        self.rotation_angles = self.settings.get("rotation_angles", [0, 90, 180, 270])

        # OCR-тюнер и cell-сегментатор
        self.tuner = OCRTuner(DEFAULT_OCR_CONFIGS)
        self.cell_parser = CellTableParser(
            ocr_psm=self.settings.get("ocr", {}).get("psm", 6),
            ocr_lang=app_settings.OCR_LANGUAGES,
            clahe=True,
            denoise=True
        )

        # Директория для скриншотов ячеек и OCR
        self.screenshots_dir = Path(app_settings.SCREENSHOTS_DIR)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    def extract_tables(
        self,
        file_path: Path,
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        opts = options or {}
        use_camelot = opts.get("use_camelot", True)
        use_tabula = opts.get("use_tabula", True)
        use_ocr = opts.get("use_ocr", True)
        use_cell = opts.get("use_cell_table", True)
        pages_opt = str(opts.get("pages", "all"))

        file_name = file_path.name
        logger.info("📂 Начало обработки PDF: %s", file_name)
        final_tables: List[Dict[str, Any]] = []

        # 1) Векторные методы
        for angle in self.rotation_angles:
            rotated_pdf = self._rotate_pdf(file_path, angle) if angle else file_path
            logger.info("🔄 Vector @ %d°", angle)

            # Camelot
            if use_camelot:
                try:
                    tables = camelot.read_pdf(
                        str(rotated_pdf),
                        pages=pages_opt,
                        flavor=self.settings.get("camelot_flavor", "stream"),
                        strip_text="\n"
                    )
                    for idx, t in enumerate(tables, start=1):
                        if t.shape[0] > 1 and t.shape[1] > 1:
                            df = t.df.replace("", pd.NA)
                            sheet = f"Camelot_{idx}_r{angle}"
                            final_tables.append({
                                "data": df,
                                "sheet_name": sheet,
                                "source": "camelot",
                                "rotation": angle,
                            })
                            logger.info("✅ %s → %d×%d", sheet, *df.shape)
                except Exception as e:
                    logger.warning("Camelot failed @ %d°: %s", angle, e)

            # Tabula
            if use_tabula:
                try:
                    dfs = tabula.read_pdf(
                        str(rotated_pdf),
                        pages=pages_opt,
                        multiple_tables=True,
                        guess=self.settings.get("guess", True)
                    )
                    for idx, df in enumerate(dfs, start=1):
                        if not df.empty and df.notna().any().any():
                            sheet = f"Tabula_{idx}_r{angle}"
                            final_tables.append({
                                "data": df,
                                "sheet_name": sheet,
                                "source": "tabula",
                                "rotation": angle,
                            })
                            logger.info("✅ %s → %d×%d", sheet, *df.shape)
                except Exception as e:
                    logger.warning("Tabula failed @ %d°: %s", angle, e)

        # Если найдены табличные данные вектором, пропускаем остальное
        if final_tables and (use_camelot or use_tabula):
            logger.info("🏆 Vector methods succeeded, skipping OCR and cell segmentation")
            return final_tables

        # 2) OCR-­тюнинг
        if use_ocr:
            logger.info("🔍 OCR all configs on all angles")
            images = convert_from_path(str(file_path), dpi=self.settings["dpi"])
            for page_idx, img in enumerate(images, start=1):
                for angle in self.rotation_angles:
                    # поворот и сохранение
                    rotated = img.rotate(angle, expand=True)
                    # tuned table extraction
                    cfg, tables, score = self.tuner.tune_page(rotated)
                    sheet = f"OCR_tuned_p{page_idx}_r{angle}"
                    for tbl in tables:
                        final_tables.append({
                            "data": tbl,
                            "sheet_name": sheet,
                            "source": "ocr_tuned",
                            "rotation": angle,
                            "ocr_config": repr(cfg),
                            "ocr_score": score,
                        })
                    logger.info("✅ %s → score=%.3f, tables=%d", sheet, score, len(tables))

                    # --- новый: полный OCR каждой повернутой страницы --- #
                    # raw text extraction
                    ocr_png = self.screenshots_dir / f"{Path(file_name).stem}_fullocr_p{page_idx}_r{angle}.png"
                    rotated.save(ocr_png)
                    logger.info("💾 FullOCR image saved: %s", ocr_png)
                    raw_text = pytesseract.image_to_string(
                        rotated,
                        lang=app_settings.OCR_LANGUAGES,
                        config="--oem 3"
                    )
                    lines = [line for line in raw_text.splitlines() if line.strip()]
                    df_raw = pd.DataFrame({"text": lines})
                    sheet_raw = f"FullOCR_p{page_idx}_r{angle}"
                    final_tables.append({
                        "data": df_raw,
                        "sheet_name": sheet_raw,
                        "source": "full_ocr",
                        "rotation": angle,
                    })
                    logger.info("✅ %s → %d lines", sheet_raw, len(lines))

        # 3) CellTableParser
        if use_cell:
            logger.info("🔎 CellTableParser on all angles")
            images = convert_from_path(str(file_path), dpi=self.settings["dpi"])
            stem = Path(file_name).stem
            for page_idx, img in enumerate(images, start=1):
                for angle in self.rotation_angles:
                    rotated = img.rotate(angle, expand=True)
                    img_path = self.screenshots_dir / f"{stem}_cells_p{page_idx}_r{angle}.png"
                    rotated.save(img_path)
                    logger.info("💾 Cell image saved: %s", img_path)

                    cell_tables = self.cell_parser.extract_tables(str(img_path))
                    for ct in cell_tables:
                        rows, cols = ct["data"].shape
                        total_cells = rows * cols
                        # не больше 200 ячеек
                        if total_cells > 200:
                            logger.info(
                                "⛔ Skipping %s p%d r%d: %d cells > 200",
                                ct["sheet_name"], page_idx, angle, total_cells
                            )
                            continue
                        sheet = f"{ct['sheet_name']}_p{page_idx}_r{angle}"
                        final_tables.append({
                            "data": ct["data"],
                            "sheet_name": sheet,
                            "source": ct["source"],
                            "rotation": angle,
                        })
                        logger.info("✅ %s → %d×%d", sheet, rows, cols)

        logger.info("🏁 Полная обработка завершена, всего таблиц: %d", len(final_tables))
        return final_tables

    def _rotate_pdf(self, file_path: Path, angle: int) -> Path:
        """
        Создаёт временный PDF, где все страницы повернуты на заданный угол.
        """
        reader = PdfReader(str(file_path))
        writer = PdfWriter()
        for page in reader.pages:
            page.rotate(angle)
            writer.add_page(page)
        tmp = Path(tempfile.gettempdir()) / f"rot_{angle}_{file_path.name}"
        with open(tmp, "wb") as f:
            writer.write(f)
        return tmp
