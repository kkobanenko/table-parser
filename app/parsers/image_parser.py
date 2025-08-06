"""
Image parser for detecting and extracting tables from images (PNG, JPG, PDF pages).
Includes OCR preprocessing with rotation and binarization.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
from utils.logger import setup_logger

logger = setup_logger("image_parser")


class ImageParser:
    """Парсер таблиц из изображений с поддержкой OCR."""

    def __init__(self, ocr_settings: Dict = None) -> None:
        self.ocr_settings = ocr_settings or {}
        self.psm = self.ocr_settings.get("psm", 6)
        self.rotate_auto = self.ocr_settings.get("rotate_auto", True)
        self.enhance_contrast = self.ocr_settings.get("enhance_contrast", True)
        self.denoise = self.ocr_settings.get("denoise", True)

    def extract_tables(self, image_path: Path) -> List[Dict]:
        """
        Извлекает таблицы из изображения.
        Алгоритм:
        1. Определение потенциальных областей таблиц
        2. OCR каждой области или всей страницы
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("⚠️ Не удалось открыть изображение %s", image_path)
            return []

        # 1️⃣ Предобработка изображения
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.enhance_contrast:
            gray = cv2.equalizeHist(gray)
        if self.denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

        # 2️⃣ Поиск контуров для определения таблиц
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        table_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 20:  # отсекаем мелкие шумы
                table_regions.append((x, y, w, h))

        if table_regions:
            logger.info("Detected %d potential table regions", len(table_regions))
        else:
            logger.info("No table regions detected, trying full image OCR")

        tables = []
        if not table_regions:
            # 3️⃣ Если таблицы не найдены — OCR всей страницы
            text_df = self._ocr_to_dataframe(gray)
            if not text_df.empty:
                tables.append({"data": text_df, "source": "ocr_full_image"})
            return tables

        # 4️⃣ OCR по областям
        for idx, (x, y, w, h) in enumerate(sorted(table_regions, key=lambda r: r[1])):
            roi = gray[y:y+h, x:x+w]
            text_df = self._ocr_to_dataframe(roi)
            if not text_df.empty:
                tables.append({"data": text_df, "source": f"ocr_region_{idx+1}"})

        return tables

    def _ocr_to_dataframe(self, img) -> pd.DataFrame:
        """Преобразует изображение с текстом в DataFrame таблицы"""
        config = f"--psm {self.psm}"
        text = pytesseract.image_to_string(img, lang="rus+eng", config=config)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return pd.DataFrame()

        # Разделение по пробелам или табуляции
        data = [re.split(r"\s{2,}|\t", line) for line in lines]

        df = pd.DataFrame(data)
        # 🔹 Исправление: используем .map вместо applymap
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        return df
