import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from utils.logger import setup_logger

logger = setup_logger('cell_table_parser')


class CellTableParser:
    """
    Попытка явно найти линии ячеек через морфологию, Hough и кластеризацию.
    """

    def __init__(
        self,
        ocr_psm: int = 6,
        ocr_lang: str = 'eng',
        clahe: bool = True,
        denoise: bool = True
    ):
        self.ocr_psm = ocr_psm
        self.ocr_lang = ocr_lang
        self.clahe = clahe
        self.denoise = denoise

    def extract_tables(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Найти на изображении таблицы по сетке.
        Возвращает список {'data': df, 'sheet_name': str, 'source': 'cell_table'}.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("Не удалось загрузить изображение: %s", image_path)
            return []

        # бинаризация
        _, bin_img = cv2.threshold(img, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # grid search over configs (omitted here for brevity)
        best_cfg = None
        best_score = -1
        best_xy = ([], [], 0)

        for cfg in self._generate_configs():
            try:
                x_lines, y_lines, intersections = self._detect_grid(bin_img, cfg)
            except Exception as e:
                logger.warning("Grid detect failed %s: %s", cfg, e)
                continue

            cell_count = len(x_lines) * len(y_lines)
            logger.info("cfg %s: intersections=%d, x_lines=%d, y_lines=%d, cells=%d",
                        cfg, intersections, len(x_lines), len(y_lines), cell_count)

            # choose by max cells but <= 200
            if cell_count <= 200 and cell_count > best_score:
                best_score = cell_count
                best_cfg = cfg
                best_xy = (x_lines, y_lines, intersections)

        if best_cfg is None:
            logger.info("❌ Нет подходящего grid-конфига")
            return []

        logger.info("🏆 Выбран лучший cfg %s с %d ячейками",
                    best_cfg, best_score)

        # финальное извлечение таблицы
        x_lines, y_lines, _ = best_xy
        tables: List[Dict[str, Any]] = []
        if x_lines and y_lines:
            cells = self._extract_cells(img, x_lines, y_lines)
            df = pd.DataFrame(cells)
            tables.append({
                "data": df,
                "sheet_name": "CellTable",
                "source": "cell_table"
            })
        return tables

    def _detect_grid(self, bin_img: np.ndarray, cfg: Dict[str, Any]
                     ) -> Tuple[List[int], List[int], int]:
        """
        Две попытки: морфология → HoughLinesP
        Возвращает (x_lines, y_lines, intersections)
        """
        # Морфологические линии
        hor = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg['length'], 1))
        ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cfg['length']))
        mask_h = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, hor)
        mask_v = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, ver)
        # пересечения
        inter = cv2.bitwise_and(mask_h, mask_v)
        intersections = int(np.count_nonzero(inter))

        # Поиск через HoughLinesP
        lines_h = cv2.HoughLinesP(
            mask_h,
            rho=1,
            theta=np.pi/180,
            threshold=cfg['hough_thresh'],
            minLineLength=cfg['min_len'],
            maxLineGap=cfg['max_gap']
        )
        lines_v = cv2.HoughLinesP(
            mask_v,
            rho=1,
            theta=np.pi/180,
            threshold=cfg['hough_thresh'],
            minLineLength=cfg['min_len'],
            maxLineGap=cfg['max_gap']
        )

        # Проверяем наличие результатов
        if lines_h is None or lines_v is None:
            return [], [], intersections

        # разворачиваем в координаты
        xs = [line[0][0] for line in lines_h] + [line[0][2] for line in lines_h]
        ys = [line[0][1] for line in lines_v] + [line[0][3] for line in lines_v]

        x_lines = self._cluster_coords(sorted(xs), cfg['cluster_dist'])
        y_lines = self._cluster_coords(sorted(ys), cfg['cluster_dist'])

        return x_lines, y_lines, intersections

    def _cluster_coords(self, coords: List[int], thresh: int) -> List[int]:
        """
        Кластеризация координат в группы по близости thresh
        """
        if not coords:
            return []

        groups: List[List[int]] = []
        current = [coords[0]]
        for c in coords[1:]:
            if abs(c - current[-1]) <= thresh:
                current.append(c)
            else:
                groups.append(current)
                current = [c]
        groups.append(current)
        # средние координаты
        return [int(sum(g)/len(g)) for g in groups]

    def _extract_cells(
        self,
        img: np.ndarray,
        x_lines: List[int],
        y_lines: List[int]
    ) -> List[List[str]]:
        """
        Разбивает по пересечениям и OCR-читает каждую ячейку.
        Возвращает матрицу строк.
        """
        cells: List[List[str]] = []
        for i in range(len(y_lines)-1):
            row: List[str] = []
            for j in range(len(x_lines)-1):
                x1, x2 = x_lines[j], x_lines[j+1]
                y1, y2 = y_lines[i], y_lines[i+1]
                roi = img[y1:y2, x1:x2]
                text = pytesseract.image_to_string(
                    roi,
                    config=f'--oem 3 --psm {self.ocr_psm}',
                    lang=self.ocr_lang
                ).strip()
                row.append(text)
            cells.append(row)
        return cells

    def _generate_configs(self) -> List[Dict[str, Any]]:
        """
        Генерирует до ~144 конфигураций:
        - length ∈ {50,100,150}
        - hough_thresh ∈ {50,100,150}
        - min_len ∈ {10,20}
        - max_gap ∈ {10,20}
        - cluster_dist ∈ {10,20}
        """
        configs: List[Dict[str, Any]] = []
        for length in (50, 100, 150):
            for hough_thresh in (50, 100, 150):
                for min_len in (10, 20):
                    for max_gap in (10, 20):
                        for cluster_dist in (10, 20):
                            configs.append({
                                'length': length,
                                'hough_thresh': hough_thresh,
                                'min_len': min_len,
                                'max_gap': max_gap,
                                'cluster_dist': cluster_dist
                            })
        return configs
