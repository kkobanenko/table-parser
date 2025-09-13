import cv2
import numpy as np
from typing import Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger('image_cleaner')


class ImageCleaner:
    """
    Класс для очистки изображений от цветных помех (печати, логотипы, водяные знаки).
    Оставляет только черный текст для лучшего OCR распознавания.
    """
    
    def __init__(self):
        self.black_threshold = 50  # Порог для определения черного цвета
        self.color_tolerance = 30  # Допуск для цветовых вариаций
    
    def clean_image(self, img: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Основной метод очистки изображения от помех.
        
        Args:
            img: Входное изображение (BGR)
            method: Метод очистки ("adaptive", "strict", "color_filter")
        
        Returns:
            Очищенное изображение в градациях серого
        """
        if method == "adaptive":
            return self._adaptive_clean(img)
        elif method == "strict":
            return self._strict_clean(img)
        elif method == "color_filter":
            return self._color_filter_clean(img)
        else:
            logger.warning(f"Неизвестный метод очистки: {method}, используем adaptive")
            return self._adaptive_clean(img)
    
    def _adaptive_clean(self, img: np.ndarray) -> np.ndarray:
        """
        Адаптивная очистка - комбинирует несколько методов.
        """
        # 1. Конвертируем в HSV для лучшей работы с цветами
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 2. Создаем маску для черного/темно-серого цвета
        # Черный цвет в HSV: V (яркость) близка к 0
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, self.black_threshold])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # 3. Создаем маску для темно-серого цвета
        lower_gray = np.array([0, 0, self.black_threshold])
        upper_gray = np.array([180, 50, 120])  # Низкая насыщенность, средняя яркость
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # 4. Объединяем маски
        combined_mask = cv2.bitwise_or(black_mask, gray_mask)
        
        # 5. Применяем маску к изображению
        cleaned = cv2.bitwise_and(img, img, mask=combined_mask)
        
        # 6. Конвертируем в градации серого
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        
        # 7. Улучшаем контраст
        gray = self._enhance_contrast(gray)
        
        return gray
    
    def _strict_clean(self, img: np.ndarray) -> np.ndarray:
        """
        Строгая очистка - оставляет только очень темные пиксели.
        """
        # Конвертируем в градации серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Создаем маску для очень темных пикселей
        _, mask = cv2.threshold(gray, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Применяем маску
        cleaned = cv2.bitwise_and(gray, mask)
        
        return cleaned
    
    def _color_filter_clean(self, img: np.ndarray) -> np.ndarray:
        """
        Фильтрация по цвету - удаляет цветные элементы.
        """
        # Конвертируем в LAB для лучшей работы с яркостью
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Извлекаем канал L (яркость)
        l_channel = lab[:, :, 0]
        
        # Создаем маску для темных областей
        _, mask = cv2.threshold(l_channel, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Применяем маску к оригинальному изображению
        cleaned = cv2.bitwise_and(img, img, mask=mask)
        
        # Конвертируем в градации серого
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        
        return gray
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Улучшение контраста изображения.
        """
        # Применяем CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        return enhanced
    
    def remove_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Удаление шума с изображения.
        """
        # Морфологические операции для удаления шума
        kernel = np.ones((2, 2), np.uint8)
        
        # Открытие (erosion + dilation) для удаления мелкого шума
        cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        # Закрытие (dilation + erosion) для заполнения мелких дыр
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_text_regions(self, img: np.ndarray) -> np.ndarray:
        """
        Детекция областей с текстом для фокусировки OCR.
        """
        # Используем MSER (Maximally Stable Extremal Regions) для детекции текста
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(img)
        
        # Создаем маску для найденных регионов
        mask = np.zeros(img.shape, dtype=np.uint8)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.fillPoly(mask, [hull], 255)
        
        return mask

