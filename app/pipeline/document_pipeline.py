"""
Комплексный пайплайн обработки документов с полным контролем через UI.

Рекомендуемый пайплайн:
pdf2image/Poppler → (deskew/denoise/binarize) → layoutparser (зоны) → 
OCR (PaddleOCR/docTR/Tesseract) → (опционально) детектор таблиц → 
экспорт (hOCR/ALTO/PDF/A/JSON)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)

# Импорты для различных компонентов пайплайна
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("⚠️ pdf2image не установлен")

try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False
    logger.warning("⚠️ LayoutParser не установлен")

try:
    from parsers.paddleocr_parser import PaddleOCRParser
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from parsers.doctr_parser import DocTRParser
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

try:
    from parsers.layoutparser_parser import LayoutParserParser
    LAYOUTPARSER_PARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_PARSER_AVAILABLE = False


class DocumentPipeline:
    """
    Комплексный пайплайн обработки документов с модульной архитектурой.
    
    Поддерживает все этапы рекомендуемого пайплайна:
    1. PDF → Image конвертация
    2. Предобработка изображений (deskew/denoise/binarize)
    3. Детекция layout зон
    4. OCR обработка
    5. Детекция таблиц
    6. Экспорт в различные форматы
    """
    
    def __init__(self):
        """Инициализация пайплайна."""
        self.preprocessing_enabled = False
        self.layout_detection_enabled = False
        self.ocr_enabled = False
        self.table_detection_enabled = False
        self.export_enabled = False
        
        # Компоненты пайплайна
        self.layout_model = None
        self.ocr_agent = None
        self.table_detector = None
        
        logger.info("✅ DocumentPipeline инициализирован")
    
    def configure_pipeline(self, config: Dict[str, Any]):
        """
        Конфигурирует пайплайн на основе настроек UI.
        
        Args:
            config: Словарь с настройками пайплайна
        """
        # Предобработка
        self.preprocessing_enabled = config.get('enable_preprocessing', False)
        self.stamp_removal_enabled = config.get('enable_stamp_removal', False)
        self.deskew_enabled = config.get('enable_deskew', False)
        self.denoise_enabled = config.get('enable_denoise', False)
        self.binarize_enabled = config.get('enable_binarize', False)
        
        # Настройки удаления печатей
        self.stamp_removal_method = config.get('stamp_removal_method', 'white_replacement')
        self.blue_hsv_thresholds = config.get('blue_hsv_thresholds', {'h_min': 100, 'h_max': 130, 's_min': 50, 'v_min': 50})
        self.red_hsv_thresholds = config.get('red_hsv_thresholds', {'h_min': 0, 'h_max': 10, 'h_min2': 170, 'h_max2': 180, 's_min': 50, 'v_min': 50})
        self.morphology_kernel_size = config.get('morphology_kernel_size', 5)
        
        # Layout детекция
        self.layout_detection_enabled = config.get('enable_layout_detection', False)
        self.layout_model_name = config.get('layout_model', 'lp://EfficientDete/PubLayNet')
        
        # OCR
        self.ocr_enabled = config.get('enable_ocr', False)
        self.ocr_method = config.get('ocr_method', 'paddleocr')  # paddleocr, doctr, tesseract
        
        # Детекция таблиц
        self.table_detection_enabled = config.get('enable_table_detection', False)
        
        # Экспорт
        self.export_enabled = config.get('enable_export', False)
        self.export_formats = config.get('export_formats', ['json'])
        
        logger.info(f"🔧 Пайплайн сконфигурирован:")
        logger.info(f"  📄 Предобработка: {self.preprocessing_enabled}")
        if self.preprocessing_enabled:
            logger.info(f"    🔴 Удаление печатей: {self.stamp_removal_enabled}")
            logger.info(f"    📐 Выравнивание: {self.deskew_enabled}")
            logger.info(f"    🔇 Удаление шума: {self.denoise_enabled}")
            logger.info(f"    ⚫ Бинаризация: {self.binarize_enabled}")
        logger.info(f"  🎯 Layout детекция: {self.layout_detection_enabled}")
        logger.info(f"  🔍 OCR: {self.ocr_enabled} ({self.ocr_method})")
        logger.info(f"  📊 Детекция таблиц: {self.table_detection_enabled}")
        logger.info(f"  💾 Экспорт: {self.export_enabled}")
    
    def initialize_components(self):
        """Инициализирует компоненты пайплайна."""
        try:
            # Инициализация LayoutParser модели
            if self.layout_detection_enabled and LAYOUTPARSER_AVAILABLE:
                self.layout_model = lp.AutoLayoutModel(self.layout_model_name)
                logger.info(f"✅ LayoutParser модель загружена: {self.layout_model_name}")
            
            # Инициализация OCR агента
            if self.ocr_enabled:
                if self.ocr_method == 'paddleocr' and PADDLEOCR_AVAILABLE:
                    self.ocr_agent = PaddleOCRParser(use_server_model=True, lang='ru')
                    logger.info("✅ PaddleOCR PP-OCRv5 инициализирован")
                elif self.ocr_method == 'doctr' and DOCTR_AVAILABLE:
                    self.ocr_agent = DocTRParser(
                        det_arch='db_resnet50',
                        reco_arch='crnn_vgg16_bn',
                        pretrained=True
                    )
                    logger.info("✅ DocTR инициализирован")
                elif self.ocr_method == 'tesseract' and LAYOUTPARSER_AVAILABLE:
                    self.ocr_agent = lp.TesseractAgent(languages='rus+eng')
                    logger.info("✅ Tesseract OCR инициализирован")
                else:
                    logger.warning(f"⚠️ OCR метод {self.ocr_method} недоступен")
            
            # Инициализация детектора таблиц
            if self.table_detection_enabled and LAYOUTPARSER_PARSER_AVAILABLE:
                self.table_detector = LayoutParserParser(
                    model_name=self.layout_model_name,
                    confidence_threshold=0.8,
                    ocr_agent='tesseract'
                )
                logger.info("✅ Детектор таблиц инициализирован")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации компонентов: {e}")
            raise
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Обрабатывает документ через полный пайплайн.
        
        Args:
            file_path: Путь к PDF файлу
            
        Returns:
            Словарь с результатами обработки
        """
        logger.info(f"🚀 Начинаем обработку документа: {Path(file_path).name}")
        
        results = {
            'file_path': file_path,
            'pages': [],
            'tables': [],
            'text': '',
            'layout_regions': [],
            'export_files': []
        }
        
        try:
            # Этап 1: PDF → Image конвертация
            images = self._convert_pdf_to_images(file_path)
            logger.info(f"📄 Конвертировано {len(images)} страниц")
            
            # Обработка каждой страницы
            for page_idx, image in enumerate(images):
                logger.info(f"🔄 Обрабатываем страницу {page_idx + 1}")
                
                page_result = self._process_page(image, page_idx)
                results['pages'].append(page_result)
                
                # Собираем результаты
                if 'tables' in page_result:
                    results['tables'].extend(page_result['tables'])
                if 'ocr_tables' in page_result:
                    results['tables'].extend(page_result['ocr_tables'])
                if 'text' in page_result:
                    results['text'] += page_result['text'] + '\n'
                if 'layout_regions' in page_result:
                    results['layout_regions'].extend(page_result['layout_regions'])
            
            # Этап 6: Экспорт результатов
            if self.export_enabled:
                export_files = self._export_results(results, file_path)
                results['export_files'] = export_files
            
            logger.info(f"✅ Обработка завершена: найдено {len(results['tables'])} таблиц")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки документа: {e}")
            raise
    
    def _convert_pdf_to_images(self, file_path: str) -> List[Image.Image]:
        """Конвертирует PDF в изображения."""
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image не установлен")
        
        try:
            images = convert_from_path(
                file_path,
                dpi=300,  # Высокое разрешение для лучшего качества OCR
                first_page=1,
                last_page=None  # Все страницы
            )
            return images
        except Exception as e:
            logger.error(f"❌ Ошибка конвертации PDF: {e}")
            raise
    
    def _process_page(self, image: Image.Image, page_idx: int) -> Dict[str, Any]:
        """Обрабатывает одну страницу через пайплайн."""
        page_result = {
            'page_idx': page_idx,
            'image': image,
            'processed_image': image,
            'layout_regions': [],
            'text': '',
            'tables': []
        }
        
        # Создаем папку temp для промежуточных файлов
        Path("temp").mkdir(exist_ok=True)
        
        # Конвертируем PIL Image в numpy array для OpenCV
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Сохраняем оригинальное изображение
        original_path = f"temp/page_{page_idx+1}_original.png"
        cv2.imwrite(original_path, img_array)
        logger.debug(f"💾 Сохранено оригинальное изображение: {original_path}")
        
        # Этап 2: Предобработка изображения
        if self.preprocessing_enabled:
            img_array = self._preprocess_image(img_array, page_idx)
            page_result['processed_image'] = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        
        # Этап 3: Детекция layout зон
        if self.layout_detection_enabled and self.layout_model:
            layout_regions = self._detect_layout_regions(img_array)
            page_result['layout_regions'] = layout_regions
        
        # Этап 4: OCR обработка
        if self.ocr_enabled and self.ocr_agent:
            if self.layout_detection_enabled and page_result['layout_regions']:
                # OCR по зонам
                text = self._ocr_by_regions(img_array, page_result['layout_regions'])
            else:
                # OCR всей страницы
                text = self._ocr_full_page(img_array)
            page_result['text'] = text
            
            # Если PaddleOCR уже извлек таблицы, сохраняем их
            if hasattr(self.ocr_agent, 'extract_tables'):
                try:
                    # Сохраняем временный файл для PaddleOCR
                    temp_path = f"temp/ocr_temp_{id(img_array)}.png"
                    cv2.imwrite(temp_path, img_array)
                    
                    try:
                        ocr_tables = self.ocr_agent.extract_tables(temp_path)
                        if ocr_tables:
                            page_result['ocr_tables'] = ocr_tables
                            logger.debug(f"📊 PaddleOCR извлек {len(ocr_tables)} таблиц")
                    finally:
                        if Path(temp_path).exists():
                            Path(temp_path).unlink()
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка извлечения таблиц PaddleOCR: {e}")
        
        # Этап 5: Детекция таблиц
        if self.table_detection_enabled and self.table_detector:
            tables = self._detect_tables(img_array, page_result)
            page_result['tables'] = tables
        
        return page_result
    
    def _preprocess_image(self, image: np.ndarray, page_idx: int) -> np.ndarray:
        """Предобработка изображения (stamp_removal/deskew/denoise/binarize)."""
        processed = image.copy()
        
        # Stamp Removal (удаление оттисков печатей)
        if self.stamp_removal_enabled:
            processed = self._remove_stamps(processed)
            stamp_path = f"temp/page_{page_idx+1}_stamp_removed.png"
            cv2.imwrite(stamp_path, processed)
            logger.debug(f"💾 Сохранено изображение после удаления печатей: {stamp_path}")
        
        # Deskew (выравнивание)
        if self.deskew_enabled:
            processed = self._deskew_image(processed)
            deskew_path = f"temp/page_{page_idx+1}_deskewed.png"
            cv2.imwrite(deskew_path, processed)
            logger.debug(f"💾 Сохранено изображение после выравнивания: {deskew_path}")
        
        # Denoise (удаление шума)
        if self.denoise_enabled:
            processed = self._denoise_image(processed)
            denoise_path = f"temp/page_{page_idx+1}_denoised.png"
            cv2.imwrite(denoise_path, processed)
            logger.debug(f"💾 Сохранено изображение после удаления шума: {denoise_path}")
        
        # Binarize (бинаризация)
        if self.binarize_enabled:
            processed = self._binarize_image(processed)
            binary_path = f"temp/page_{page_idx+1}_binarized.png"
            cv2.imwrite(binary_path, processed)
            logger.debug(f"💾 Сохранено бинаризованное изображение: {binary_path}")
        
        # Сохраняем финальное обработанное изображение
        final_path = f"temp/page_{page_idx+1}_final_processed.png"
        cv2.imwrite(final_path, processed)
        logger.debug(f"💾 Сохранено финальное обработанное изображение: {final_path}")
        
        return processed
    
    def _preprocess_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Применяет предобработку к списку изображений."""
        processed_pages = []
        
        for i, image in enumerate(images):
            # Конвертируем PIL Image в numpy array
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Применяем предобработку
            processed_array = self._preprocess_image(img_array)
            
            # Конвертируем обратно в PIL Image
            processed_image = Image.fromarray(cv2.cvtColor(processed_array, cv2.COLOR_BGR2RGB))
            
            processed_pages.append({
                'page_number': i + 1,
                'image': image,
                'processed_image': processed_image
            })
        
        return processed_pages
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Выравнивание изображения (deskew)."""
        try:
            # Конвертируем в grayscale для анализа
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Находим контуры текста
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Вычисляем средний угол наклона
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi - 90
                    angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Поворачиваем изображение
                    if abs(median_angle) > 0.5:  # Только если угол значительный
                        h, w = image.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            logger.debug(f"📐 Выравнивание изображения выполнено")
            return image
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка выравнивания: {e}")
            return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Удаление шума с изображения."""
        try:
            # Используем Non-local Means Denoising
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            logger.debug(f"🔇 Удаление шума выполнено")
            return denoised
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка удаления шума: {e}")
            return image
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Бинаризация изображения."""
        try:
            # Конвертируем в grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Адаптивная бинаризация
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Конвертируем обратно в BGR
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            logger.debug(f"⚫ Бинаризация выполнена")
            return binary_bgr
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка бинаризации: {e}")
            return image
    
    def _remove_stamps(self, image: np.ndarray) -> np.ndarray:
        """
        Удаление оттисков печатей с изображения.
        
        Использует цветовую сегментацию в HSV пространстве для выделения
        синих и красных печатей, затем удаляет их с помощью морфологических операций.
        
        Args:
            image: Входное изображение в формате BGR
            
        Returns:
            Обработанное изображение без печатей
        """
        try:
            # Конвертируем в HSV для лучшего выделения цветов
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Создаем маски для синих печатей
            blue_lower = np.array([
                self.blue_hsv_thresholds['h_min'],
                self.blue_hsv_thresholds['s_min'],
                self.blue_hsv_thresholds['v_min']
            ])
            blue_upper = np.array([
                self.blue_hsv_thresholds['h_max'],
                255, 255
            ])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            # Создаем маски для красных печатей (красный цвет в HSV имеет два диапазона)
            red_lower1 = np.array([
                self.red_hsv_thresholds['h_min'],
                self.red_hsv_thresholds['s_min'],
                self.red_hsv_thresholds['v_min']
            ])
            red_upper1 = np.array([
                self.red_hsv_thresholds['h_max'],
                255, 255
            ])
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([
                self.red_hsv_thresholds['h_min2'],
                self.red_hsv_thresholds['s_min'],
                self.red_hsv_thresholds['v_min']
            ])
            red_upper2 = np.array([
                self.red_hsv_thresholds['h_max2'],
                255, 255
            ])
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            # Объединяем маски красных печатей
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Объединяем все маски печатей
            stamp_mask = cv2.bitwise_or(blue_mask, red_mask)
            
            # Морфологические операции для очистки маски
            kernel_size = self.morphology_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Закрытие для заполнения отверстий в печатях
            stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_CLOSE, kernel)
            
            # Открытие для удаления шума
            stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_OPEN, kernel)
            
            # Применяем маску к изображению
            result = image.copy()
            
            if self.stamp_removal_method == 'white_replacement':
                # Простая замена белым цветом
                result[stamp_mask > 0] = [255, 255, 255]
            elif self.stamp_removal_method == 'inpainting':
                # Продвинутая интерполяция фона
                result = cv2.inpaint(result, stamp_mask, 3, cv2.INPAINT_TELEA)
            
            # Подсчитываем количество удаленных пикселей для логирования
            removed_pixels = np.sum(stamp_mask > 0)
            total_pixels = image.shape[0] * image.shape[1]
            removal_percentage = (removed_pixels / total_pixels) * 100
            
            logger.debug(f"🔴 Удалено печатей: {removed_pixels} пикселей ({removal_percentage:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка удаления печатей: {e}")
            return image
    
    def _detect_layout_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Детекция layout зон."""
        try:
            if not self.layout_model:
                return []
            
            # Детекция layout элементов
            layout = self.layout_model.detect(image)
            
            # Конвертируем в удобный формат
            regions = []
            for element in layout:
                region = {
                    'type': element.type,
                    'coordinates': element.coordinates,
                    'confidence': getattr(element, 'score', 1.0),
                    'bbox': [element.coordinates[0], element.coordinates[1], 
                            element.coordinates[2], element.coordinates[3]]
                }
                regions.append(region)
            
            logger.debug(f"🎯 Найдено {len(regions)} layout зон")
            return regions
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка детекции layout: {e}")
            return []
    
    def _ocr_by_regions(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> str:
        """OCR обработка по зонам."""
        try:
            all_text = []
            
            for region in regions:
                # Извлекаем зону изображения
                bbox = region['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Обрезаем изображение
                region_image = image[y1:y2, x1:x2]
                
                if region_image.size > 0:
                    # OCR зоны
                    text = self._perform_ocr(region_image)
                    if text.strip():
                        all_text.append(f"[{region['type']}] {text}")
            
            result_text = '\n'.join(all_text)
            logger.debug(f"🔍 OCR по зонам: {len(all_text)} зон обработано")
            return result_text
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка OCR по зонам: {e}")
            return ""
    
    def _ocr_full_page(self, image: np.ndarray) -> str:
        """OCR обработка всей страницы."""
        try:
            text = self._perform_ocr(image)
            logger.debug(f"🔍 OCR всей страницы: {len(text)} символов")
            return text
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка OCR всей страницы: {e}")
            return ""
    
    def _perform_ocr(self, image: np.ndarray) -> str:
        """Выполняет OCR на изображении."""
        try:
            if self.ocr_method == 'paddleocr' and isinstance(self.ocr_agent, PaddleOCRParser):
                # Сохраняем временный файл для PaddleOCR
                temp_path = f"temp/ocr_temp_{id(image)}.png"
                Path("temp").mkdir(exist_ok=True)
                cv2.imwrite(temp_path, image)
                
                try:
                    tables = self.ocr_agent.extract_tables(temp_path)
                    # Извлекаем текст из таблиц
                    text_parts = []
                    for table in tables:
                        for row in table:
                            text_parts.append(' '.join(str(cell) for cell in row))
                    return '\n'.join(text_parts)
                finally:
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
            
            elif self.ocr_method == 'doctr' and isinstance(self.ocr_agent, DocTRParser):
                # Сохраняем временный файл для DocTR
                temp_path = f"temp/ocr_temp_{id(image)}.png"
                Path("temp").mkdir(exist_ok=True)
                cv2.imwrite(temp_path, image)
                
                try:
                    tables = self.ocr_agent.extract_tables(temp_path)
                    # Извлекаем текст из таблиц
                    text_parts = []
                    for table in tables:
                        for row in table:
                            text_parts.append(' '.join(str(cell) for cell in row))
                    return '\n'.join(text_parts)
                finally:
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
            
            elif self.ocr_method == 'tesseract' and hasattr(self.ocr_agent, 'detect'):
                # Tesseract через LayoutParser
                return self.ocr_agent.detect(image)
            
            else:
                logger.warning(f"⚠️ Неизвестный OCR метод: {self.ocr_method}")
                return ""
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка выполнения OCR: {e}")
            return ""
    
    def _detect_tables(self, image: np.ndarray, page_result: Dict[str, Any]) -> List[List[List[str]]]:
        """Детекция таблиц."""
        try:
            if not self.table_detector:
                return []
            
            # Сохраняем временный файл
            temp_path = f"temp/table_detection_{id(image)}.png"
            Path("temp").mkdir(exist_ok=True)
            cv2.imwrite(temp_path, image)
            
            try:
                tables = self.table_detector.extract_tables(temp_path)
                logger.debug(f"📊 Найдено {len(tables)} таблиц")
                return tables
            finally:
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                    
        except Exception as e:
            logger.warning(f"⚠️ Ошибка детекции таблиц: {e}")
            return []
    
    def _export_results(self, results: Dict[str, Any], file_path: str) -> List[str]:
        """Экспорт результатов в различные форматы."""
        export_files = []
        base_name = Path(file_path).stem
        
        try:
            # JSON экспорт
            if 'json' in self.export_formats:
                json_path = f"output/{base_name}_pipeline_results.json"
                Path("output").mkdir(exist_ok=True)
                
                # Подготавливаем данные для JSON
                json_data = {
                    'file_path': results['file_path'],
                    'pages_count': len(results['pages']),
                    'tables_count': len(results['tables']),
                    'text_length': len(results['text']),
                    'layout_regions_count': len(results['layout_regions']),
                    'pages': [
                        {
                            'page_idx': page['page_idx'],
                            'text_length': len(page.get('text', '')),
                            'tables_count': len(page.get('tables', [])),
                            'layout_regions_count': len(page.get('layout_regions', []))
                        }
                        for page in results['pages']
                    ]
                }
                
                import json
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                export_files.append(json_path)
                logger.info(f"💾 JSON экспорт: {json_path}")
            
            # Excel экспорт таблиц
            if 'excel' in self.export_formats and results['tables']:
                excel_path = f"output/{base_name}_tables.xlsx"
                
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    for i, table in enumerate(results['tables']):
                        if table:
                            df = pd.DataFrame(table)
                            sheet_name = f"Table_{i+1}"
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                export_files.append(excel_path)
                logger.info(f"💾 Excel экспорт: {excel_path}")
            
            # CSV экспорт таблиц
            if 'csv' in self.export_formats and results['tables']:
                csv_path = f"output/{base_name}_tables.csv"
                
                all_tables_data = []
                for i, table in enumerate(results['tables']):
                    if table:
                        for row in table:
                            all_tables_data.append(row)
                
                if all_tables_data:
                    df = pd.DataFrame(all_tables_data)
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    export_files.append(csv_path)
                    logger.info(f"💾 CSV экспорт: {csv_path}")
            
            # TXT экспорт текста
            if 'txt' in self.export_formats and results['text']:
                txt_path = f"output/{base_name}_text.txt"
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(results['text'])
                
                export_files.append(txt_path)
                logger.info(f"💾 TXT экспорт: {txt_path}")
            
            logger.info(f"✅ Экспорт завершен: {len(export_files)} файлов")
            return export_files
            
        except Exception as e:
            logger.error(f"❌ Ошибка экспорта: {e}")
            return []
