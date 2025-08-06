"""
PDF-парсер с постраничным OCR-тюнингом и перебором поворотов,
а также пробным запуском всех трёх основных библиотек на каждом угле.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import tempfile

import pandas as pd
import camelot
import pdfplumber
import tabula
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter

from config.pdf_parser_settings import PDFParserSettings
from config.settings import settings as app_settings
from utils.ocr_tuner import DEFAULT_OCR_CONFIGS, OCRConfig, OCRTuner
from utils.logger import setup_logger

logger = setup_logger("pdf_parser")


class PDFParser:
    """PDF-парсер с интеграцией всех библиотек и OCR-тюнера + повороты."""

    def __init__(self, settings: Dict = None) -> None:
        self.settings = settings or PDFParserSettings.load_settings()
        self.settings.setdefault("dpi", 300)
        self.cache_dir = Path(self.settings["cache_dir"])
        self.rotation_angles = self.settings.get("rotation_angles", [0, 90, 180, 270])
        self.screenshots_dir = Path(app_settings.SCREENSHOTS_DIR)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.tuner = OCRTuner(DEFAULT_OCR_CONFIGS)

    def extract_tables(self, file_path: Path) -> List[Dict]:
        file_name = file_path.name
        logger.info("📂 Начало обработки PDF: %s", file_name)
        final_tables: List[Dict] = []

        # 1) Векторные библиотеки (Camelot, pdfplumber, Tabula) на всех углах
        for angle in self.rotation_angles:
            rotated_pdf = self._rotate_pdf(file_path, angle) if angle else file_path
            logger.info("🔄 Библиотеки на углу %d°", angle)

            # Camelot
            try:
                flavor = self.settings.get("camelot_flavor", "stream")
                tables = camelot.read_pdf(
                    str(rotated_pdf), pages="all", flavor=flavor, strip_text="\n"
                )
                for idx, t in enumerate(tables, start=1):
                    if t.shape[0] > 1 and t.shape[1] > 1:
                        df = t.df.replace("", pd.NA)
                        sheet = f"Camelot_{flavor}_r{angle}_{idx}"
                        final_tables.append({
                            "data": df,
                            "sheet_name": sheet,
                            "source": f"camelot_{flavor}",
                            "rotation": angle,
                        })
                        logger.info("✅ %s → %d×%d", sheet, *df.shape)
            except Exception as e:
                logger.warning("Camelot не сработал на %d°: %s", angle, e)

            # pdfplumber
            try:
                with pdfplumber.open(rotated_pdf) as pdf:
                    for pno, page in enumerate(pdf.pages, start=1):
                        rows = page.extract_tables(table_settings={
                            "vertical_strategy": self.settings["vertical_strategy"],
                            "horizontal_strategy": self.settings["horizontal_strategy"],
                            "join_tolerance": self.settings["line_margin"],
                            "snap_tolerance": self.settings["char_margin"],
                        })
                        for idx, tbl in enumerate(rows, start=1):
                            if tbl and len(tbl) > 1:
                                df = pd.DataFrame(tbl[1:], columns=tbl[0])
                                sheet = f"Plumber_r{angle}_p{pno}_{idx}"
                                final_tables.append({
                                    "data": df,
                                    "sheet_name": sheet,
                                    "source": "pdfplumber",
                                    "rotation": angle,
                                })
                                logger.info("✅ %s → %d×%d", sheet, *df.shape)
            except Exception as e:
                logger.warning("pdfplumber не сработал на %d°: %s", angle, e)

            # Tabula
            try:
                dfs = tabula.read_pdf(
                    str(rotated_pdf), pages="all", multiple_tables=True,
                    guess=self.settings.get("guess", True)
                )
                for idx, df in enumerate(dfs, start=1):
                    if not df.empty and df.notna().any().any():
                        sheet = f"Tabula_r{angle}_{idx}"
                        final_tables.append({
                            "data": df,
                            "sheet_name": sheet,
                            "source": "tabula",
                            "rotation": angle,
                        })
                        logger.info("✅ %s → %d×%d", sheet, *df.shape)
            except Exception as e:
                logger.warning("Tabula не сработал на %d°: %s", angle, e)

        # 2) OCR-обработка на изображениях: все конфигурации и углы
        logger.info("🔍 Запускаем OCR-тюнинг всех конфигураций на всех углах")
        images = convert_from_path(str(file_path), dpi=self.settings["dpi"])
        for page_idx, img in enumerate(images, start=1):
            for angle in self.rotation_angles:
                rotated = img.rotate(angle, expand=True)
                # сохраняем для проверки
                img_path = (
                    self.screenshots_dir
                    / f"{Path(file_name).stem}_ocr_p{page_idx}_rot{angle}.png"
                )
                rotated.save(img_path)
                logger.info("💾 Сохранено OCR-изображение: %s", img_path)

                for cfg in DEFAULT_OCR_CONFIGS:
                    local_tuner = OCRTuner([cfg])
                    cfg_used, tbls, score = local_tuner.tune_page(rotated)
                    sheet = f"OCR_psm{cfg.psm}_{cfg.binarization_method}_r{angle}_p{page_idx}"
                    for tbl in tbls:
                        final_tables.append({
                            "data": tbl,
                            "sheet_name": sheet,
                            "source": "ocr",
                            "rotation": angle,
                            "ocr_config": repr(cfg),
                            "ocr_score": score,
                        })
                    logger.info("%s → score=%.3f, tables=%d", sheet, score, len(tbls))

        logger.info("🏁 Полная обработка завершена, таблиц: %d", len(final_tables))
        return final_tables

    def _rotate_pdf(self, file_path: Path, angle: int) -> Path:
        """
        Создаёт временный PDF, в котором все страницы повернуты на заданный угол.
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
