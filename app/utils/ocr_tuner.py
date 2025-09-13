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
    preprocess: str = "none"  # none, denoise, enhance, clahe
    oem: int = 3  # OCR Engine Mode
    # любые другие параметры конфигурации OCR


# Расширенный набор конфигураций для лучшего распознавания таблиц
DEFAULT_OCR_CONFIGS: List[OCRConfig] = [
    # Базовые конфигурации для таблиц
    OCRConfig(psm=6, binarization="otsu", preprocess="none"),  # Единый блок текста
    OCRConfig(psm=3, binarization="otsu", preprocess="none"),  # Автоматическая сегментация
    OCRConfig(psm=11, binarization="otsu", preprocess="none"),  # Разреженный текст
    
    # С улучшением контраста
    OCRConfig(psm=6, binarization="otsu", preprocess="clahe"),
    OCRConfig(psm=3, binarization="otsu", preprocess="clahe"),
    OCRConfig(psm=11, binarization="otsu", preprocess="clahe"),
    
    # С шумоподавлением
    OCRConfig(psm=6, binarization="otsu", preprocess="denoise"),
    OCRConfig(psm=3, binarization="otsu", preprocess="denoise"),
    
    # Адаптивная бинаризация
    OCRConfig(psm=6, binarization="adaptive_gaussian", threshold=(11, 2), preprocess="none"),
    OCRConfig(psm=6, binarization="adaptive_gaussian", threshold=(15, 2), preprocess="none"),
    OCRConfig(psm=6, binarization="adaptive_mean", threshold=(11, 2), preprocess="none"),
    
    # Комбинированная предобработка
    OCRConfig(psm=6, binarization="otsu", preprocess="enhance"),
    OCRConfig(psm=3, binarization="otsu", preprocess="enhance"),
    
    # Специальные режимы для таблиц
    OCRConfig(psm=4, binarization="otsu", preprocess="none"),  # Один столбец текста
    OCRConfig(psm=7, binarization="otsu", preprocess="none"),  # Одна строка текста
    OCRConfig(psm=8, binarization="otsu", preprocess="none"),  # Одно слово
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
        # 1) Предобработка изображения
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        
        # Применяем предобработку в зависимости от конфигурации
        if cfg.preprocess == "clahe":
            # Улучшение контраста с помощью CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        elif cfg.preprocess == "denoise":
            # Шумоподавление
            gray = cv2.fastNlMeansDenoising(gray)
        elif cfg.preprocess == "enhance":
            # Комбинированное улучшение
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.fastNlMeansDenoising(gray)
        
        # 2) Бинаризация
        if cfg.binarization == "otsu":
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif cfg.binarization == "adaptive_gaussian":
            threshold = cfg.threshold or (11, 2)
            bin_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, threshold[0], threshold[1]
            )
        elif cfg.binarization == "adaptive_mean":
            threshold = cfg.threshold or (11, 2)
            bin_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, threshold[0], threshold[1]
            )
        else:
            # по умолчанию — простая бинаризация
            _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # 3) Запуск tesseract с улучшенными параметрами
        custom_config = f"--oem {cfg.oem} --psm {cfg.psm}"
        data = pytesseract.image_to_data(
            bin_img, config=custom_config, output_type=pytesseract.Output.DATAFRAME
        )

        # 4) Преобразование в таблицу pandas
        tables: List[pd.DataFrame] = []
        # Группируем по строкам tesseract
        if not data.empty:
            data = data.dropna(subset=["text"])
            # Фильтруем пустые тексты - приводим к строковому типу
            data["text"] = data["text"].astype(str)
            data = data[data["text"].str.strip() != ""]
            
            if not data.empty:
                # Собираем строки и столбцы на основе координат bbox
                # Группируем по строкам (top координата)
                data_sorted = data.sort_values(["top", "left"])
                
                # Создаем таблицу на основе координат
                rows = []
                current_row = []
                last_top = None
                tolerance = 10  # Допуск для группировки в строки
                
                for _, word in data_sorted.iterrows():
                    if last_top is None or abs(word["top"] - last_top) <= tolerance:
                        current_row.append(word["text"])
                    else:
                        if current_row:
                            rows.append(current_row)
                        current_row = [word["text"]]
                    last_top = word["top"]
                
                if current_row:
                    rows.append(current_row)
                
                # Создаем DataFrame из строк
                if rows:
                    max_cols = max(len(row) for row in rows) if rows else 0
                    # Дополняем короткие строки пустыми ячейками
                    padded_rows = [row + [""] * (max_cols - len(row)) for row in rows]
                    df = pd.DataFrame(padded_rows)
                    tables.append(df)

        return tables
