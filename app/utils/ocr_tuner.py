"""
Модуль для автоматического подбора оптимальных настроек OCR
по качеству извлечения таблиц.
"""

from typing import List, Tuple, Dict, Any
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re


class OCRConfig:
    """
    Конфигурация пред- и пост-обработки для OCR.
    """
    def __init__(
        self,
        psm: int,
        binarization_method: str,
        use_clahe: bool,
        use_denoise: bool,
        morph_kernel: Tuple[int, int] = None,
    ):
        self.psm = psm
        self.binarization_method = binarization_method
        self.use_clahe = use_clahe
        self.use_denoise = use_denoise
        self.morph_kernel = morph_kernel

    def __repr__(self):
        return (
            f"OCRConfig(psm={self.psm}, binar={self.binarization_method}, "
            f"clahe={self.use_clahe}, denoise={self.use_denoise}, "
            f"morph={self.morph_kernel})"
        )


# Преднастроенные конфигурации (≤10)
DEFAULT_OCR_CONFIGS: List[OCRConfig] = [
    OCRConfig(psm=6, binarization_method="otsu", use_clahe=False, use_denoise=False),
    OCRConfig(psm=6, binarization_method="otsu", use_clahe=True, use_denoise=False),
    OCRConfig(psm=6, binarization_method="otsu", use_clahe=True, use_denoise=True),
    OCRConfig(psm=6, binarization_method="adaptive_mean", use_clahe=True, use_denoise=True),
    OCRConfig(psm=11, binarization_method="otsu", use_clahe=True, use_denoise=True),
    OCRConfig(psm=11, binarization_method="adaptive_gaussian", use_clahe=True, use_denoise=True),
    OCRConfig(psm=3, binarization_method="otsu", use_clahe=True, use_denoise=True, morph_kernel=(5,5)),
    OCRConfig(psm=3, binarization_method="adaptive_mean", use_clahe=True, use_denoise=True, morph_kernel=(5,5)),
]


class OCRTuner:
    """
    Класс для перебора наборов настроек OCR и выбора наилучшей по метрике.
    """
    def __init__(self, configs: List[OCRConfig]):
        self.configs = configs
        self._pattern = re.compile(r"[0-9A-Za-zА-Яа-я]")

    def tune_page(
        self, image: Image.Image
    ) -> Tuple[OCRConfig, List[pd.DataFrame], float]:
        """
        Перебирает все конфигурации для одной страницы,
        возвращает лучшую конфигурацию, список таблиц и значение метрики.
        """
        best_score = -1.0
        best_cfg: OCRConfig = None  # type: ignore
        best_tables: List[pd.DataFrame] = []

        # Преобразуем PIL.Image в OpenCV-матрицу
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        for cfg in self.configs:
            proc = gray.copy()
            if cfg.use_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                proc = clahe.apply(proc)
            if cfg.use_denoise:
                proc = cv2.fastNlMeansDenoising(proc, h=10)
            proc = self._binarize(proc, cfg.binarization_method)
            if cfg.morph_kernel:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.morph_kernel)
                proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)

            tables = self._ocr_to_tables(proc, cfg.psm)
            score = self._eval_tables(tables)
            if score > best_score:
                best_score, best_cfg, best_tables = score, cfg, tables

        return best_cfg, best_tables, best_score

    def tune_pdf(self, images: List[Image.Image]) -> Dict[int, Dict[str, Any]]:
        """
        Подбирает оптимальную конфигурацию для каждой страницы списка изображений.
        Возвращает словарь: {страница: {config, tables, score}}
        """
        results: Dict[int, Dict[str, Any]] = {}
        for idx, img in enumerate(images, start=1):
            cfg, tables, score = self.tune_page(img)
            results[idx] = {"config": cfg, "tables": tables, "score": score}
        return results

    def _binarize(self, img: np.ndarray, method: str) -> np.ndarray:
        if method == "otsu":
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th
        if method == "adaptive_mean":
            return cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
        if method == "adaptive_gaussian":
            return cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        # fallback to Otsu
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def _ocr_to_tables(self, img: np.ndarray, psm: int) -> List[pd.DataFrame]:
        """
        Выполняет OCR и собирает текст в таблицу.
        """
        cfg = f"--psm {psm}"
        data = pytesseract.image_to_data(
            img, config=cfg, output_type=pytesseract.Output.DICT
        )
        lines: Dict[int, List[str]] = {}
        for i, txt in enumerate(data["text"]):
            txt = txt.strip()
            if not txt:
                continue
            row = data["line_num"][i]
            lines.setdefault(row, []).append(txt)

        tables: List[pd.DataFrame] = []
        if lines:
            rows = [lines[r] for r in sorted(lines)]
            maxc = max(len(r) for r in rows)
            normalized = [r + [""] * (maxc - len(r)) for r in rows]
            tables.append(pd.DataFrame(normalized))
        return tables

    def _eval_tables(self, tables: List[pd.DataFrame]) -> float:
        """
        Вычисляет комбинированную метрику качества для списка таблиц.
        """
        if not tables:
            return 0.0
        frates, ccons, hpres = [], [], []
        for df in tables:
            total = df.size
            nonempty = df.astype(str).applymap(
                lambda x: bool(self._pattern.search(x))
            ).sum().sum()
            fr = nonempty / total if total else 0
            row_counts = df.count(axis=1)
            mode = row_counts.mode()[0] if not row_counts.mode().empty else 0
            cc = (row_counts == mode).mean()
            hp = df.iloc[0].astype(str).apply(
                lambda x: bool(self._pattern.search(x))
            ).mean()
            frates.append(fr)
            ccons.append(cc)
            hpres.append(hp)
        return 0.5 * np.mean(frates) + 0.3 * np.mean(ccons) + 0.2 * np.mean(hpres)


def tune_pdf_ocr(pdf_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Упрощённая обёртка: конвертирует PDF в изображения и запускает OCRTuner.
    """
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=300)
    tuner = OCRTuner(DEFAULT_OCR_CONFIGS)
    return tuner.tune_pdf(images)
