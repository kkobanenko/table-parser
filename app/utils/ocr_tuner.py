import pytesseract
import numpy as np
import cv2
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any


@dataclass
class OCRConfig:
    psm: int
    binarization: str
    threshold: Optional[Tuple[int, int]] = None
    # любые другие параметры конфигурации OCR


# Пример наборов конфигураций — ваш список может отличаться
DEFAULT_OCR_CONFIGS: List[OCRConfig] = [
    OCRConfig(psm=3, binarization="otsu"),
    OCRConfig(psm=6, binarization="otsu"),
    OCRConfig(psm=11, binarization="adaptive_gaussian"),
    OCRConfig(psm=11, binarization="adaptive_mean"),
    # … всего не более 10–15 конфигов
]


class OCRTuner:
    """
    Подбирает среди множества конфигураций OCR ту,
    которая даёт наибольшее качество по метрике «количество непустых ячеек».
    """

    def __init__(self, configs: List[OCRConfig]):
        self.configs = configs

    def tune_page(
        self, img: Any
    ) -> Tuple[OCRConfig, List[pd.DataFrame], float]:
        """
        Протестировать все конфигурации на одном изображении.
        Возвращает:
          - победную конфигурацию,
          - список извлечённых таблиц (DataFrame),
          - score: доля непустых ячеек.
        """
        best_cfg = None
        best_score = -1
        best_tables: List[pd.DataFrame] = []

        for cfg in self.configs:
            tables = self._do_ocr(img, cfg)
            # объединяем все таблицы в один DataFrame, чтобы посчитать непустые ячейки
            if tables:
                combined = pd.concat(tables, ignore_index=True)
                # вместо applymap используем apply+map
                stripped = combined.astype(str).apply(
                    lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x)
                )
                nonempty = stripped.astype(bool).sum().sum()
                total = stripped.size
                score = nonempty / total if total else 0.0
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_cfg = cfg
                best_tables = tables

        # на случай, если ни по одной конфигурации таблиц не нашлось
        if best_cfg is None:
            best_cfg = self.configs[0]
        return best_cfg, best_tables, best_score

    def _do_ocr(self, img: Any, cfg: OCRConfig) -> List[pd.DataFrame]:
        """
        Запустить Tesseract для одной конфигурации: psm, метод binarization, пороги.
        Возвращает список DataFrame-ов, где каждая таблица — результат OCR разбора.
        """
        # 1) Бинаризация
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        if cfg.binarization == "otsu":
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif cfg.binarization == "adaptive_gaussian":
            bin_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, cfg.threshold[0], cfg.threshold[1]
            )
        elif cfg.binarization == "adaptive_mean":
            bin_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, cfg.threshold[0], cfg.threshold[1]
            )
        else:
            # по умолчанию — простая бинаризация
            _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # 2) Запуск tesseract
        custom_oem_psm = f"--oem 3 --psm {cfg.psm}"
        data = pytesseract.image_to_data(
            bin_img, config=custom_oem_psm, output_type=pytesseract.Output.DATAFRAME
        )

        # 3) Преобразование в таблицу pandas
        tables: List[pd.DataFrame] = []
        # Группируем по строкам tesseract
        if not data.empty:
            data = data.dropna(subset=["text"])
            # Собираем строки и столбцы на основе координат bbox
            # (ваша логика разбора таблицы здесь)
            # Для простоты возьмём текстовые строки как отдельные таблицы:
            for _, group in data.groupby("line_num"):
                df_line = group[["left", "top", "text"]]
                tables.append(df_line)

        return tables
