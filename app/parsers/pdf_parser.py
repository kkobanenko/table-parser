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
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    from pypdf import PdfReader, PdfWriter

from config.pdf_parser_settings import PDFParserSettings
from config.settings import settings as app_settings
from utils.logger import setup_logger
from utils.ocr_tuner import DEFAULT_OCR_CONFIGS, OCRTuner
from parsers.cell_table_parser import CellTableParser
from parsers.text_structure_parser import TextStructureParser
from parsers.spacing_parser import SpacingParser

# Условный импорт EasyOCR парсера
try:
    from parsers.easyocr_parser import EasyOCRParser
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from parsers.spacing_analysis_parser import SpacingAnalysisParser

# Условный импорт MarkItDown парсера
try:
    from parsers.markitdown_parser import MarkItDownParser
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

# Условный импорт PaddleOCR парсера
try:
    from parsers.paddleocr_parser import PaddleOCRParser
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# Условный импорт DocTR парсера
try:
    from parsers.doctr_parser import DocTRParser
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

# Условный импорт LayoutParser парсера
try:
    from parsers.layoutparser_parser import LayoutParserParser
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False

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

        # OCR-тюнер и парсеры
        self.tuner = OCRTuner(DEFAULT_OCR_CONFIGS)
        self.cell_parser = CellTableParser(
            ocr_psm=self.settings.get("ocr", {}).get("psm", 6),
            ocr_lang=app_settings.OCR_LANGUAGES,
            clahe=True,
            denoise=True
        )
        self.text_structure_parser = TextStructureParser(
            ocr_psm=self.settings.get("ocr", {}).get("psm", 6),
            ocr_lang=app_settings.OCR_LANGUAGES
        )
        self.spacing_parser = SpacingParser(
            ocr_psm=self.settings.get("ocr", {}).get("psm", 6),
            ocr_lang=app_settings.OCR_LANGUAGES
        )
        
        # Новые парсеры для улучшенного анализа таблиц
        if EASYOCR_AVAILABLE:
            self.easyocr_parser = EasyOCRParser(
                languages=['en', 'ru'],
                gpu=False  # Используем CPU для совместимости
            )
        else:
            self.easyocr_parser = None
            logger.warning("⚠️ EasyOCR не установлен, парсер будет недоступен")
        self.spacing_analysis_parser = SpacingAnalysisParser(
            ocr_psm=self.settings.get("ocr", {}).get("psm", 6),
            ocr_lang=app_settings.OCR_LANGUAGES
        )
        
        # MarkItDown парсер
        if MARKITDOWN_AVAILABLE:
            self.markitdown_parser = MarkItDownParser(enable_plugins=False)
        else:
            self.markitdown_parser = None
            logger.warning("⚠️ MarkItDown не установлен, парсер будет недоступен")
        
        # PaddleOCR парсер
        if PADDLEOCR_AVAILABLE:
            self.paddleocr_parser = PaddleOCRParser(
                use_server_model=True,  # Используем server модель для высокой точности
                lang='ru'  # Поддерживаем русский язык для лучшего распознавания
            )
        else:
            self.paddleocr_parser = None
            logger.warning("⚠️ PaddleOCR не установлен, парсер будет недоступен")
        
        # DocTR парсер
        if DOCTR_AVAILABLE:
            self.doctr_parser = DocTRParser(
                det_arch='db_resnet50',
                reco_arch='crnn_vgg16_bn',
                pretrained=True,
                assume_straight_pages=True,
                preserve_aspect_ratio=False
            )
        else:
            self.doctr_parser = None
            logger.warning("⚠️ DocTR не установлен, парсер будет недоступен")
        
        # LayoutParser парсер
        if LAYOUTPARSER_AVAILABLE:
            self.layoutparser_parser = LayoutParserParser(
                model_name='lp://EfficientDete/PubLayNet',
                confidence_threshold=0.8,
                ocr_agent='tesseract'
            )
        else:
            self.layoutparser_parser = None
            logger.warning("⚠️ LayoutParser не установлен, парсер будет недоступен")

        # Директория для скриншотов ячеек и OCR
        self.screenshots_dir = Path(app_settings.SCREENSHOTS_DIR)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Директория для промежуточных файлов (повороты, очищенные изображения)
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

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
        use_text_structure = opts.get("use_text_structure", True)
        use_spacing = opts.get("use_spacing", True)
        use_easyocr = opts.get("use_easyocr", True)
        use_spacing_analysis = opts.get("use_spacing_analysis", True)
        use_markitdown = opts.get("use_markitdown", True)
        use_paddleocr = opts.get("use_paddleocr", True)
        pages_opt = str(opts.get("pages", "all"))
        check_rotations = opts.get("check_rotations", True)  # Новый параметр для контроля поворотов
        
        # Парсим параметр pages для разных методов
        pages_info = self._parse_pages_option(pages_opt)

        file_name = file_path.name
        logger.info("📂 Начало обработки PDF: %s", file_name)
        final_tables: List[Dict[str, Any]] = []

        # Определяем углы для обработки
        angles_to_process = [0] if not check_rotations else self.rotation_angles
        
        # 1) Векторные методы
        for angle in angles_to_process:
            rotated_pdf = self._rotate_pdf(file_path, angle) if angle else file_path
            logger.info("🔄 Vector @ %d°", angle)

            # Camelot
            if use_camelot:
                try:
                    tables = camelot.read_pdf(
                        str(rotated_pdf),
                        pages=pages_info["camelot_tabula"],
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
                        pages=pages_info["camelot_tabula"],
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
            logger.info("🔍 OCR all configs on angles: %s", angles_to_process)
            images = convert_from_path(
                str(file_path), 
                dpi=self.settings["dpi"],
                first_page=pages_info["first_page"],
                last_page=pages_info["last_page"]
            )
            for page_idx, img in enumerate(images, start=1):
                for angle in angles_to_process:
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
                    
                    # Сохраняем повернутое изображение в temp для анализа
                    temp_rotated = self.temp_dir / f"{Path(file_name).stem}_rotated_p{page_idx}_r{angle}.png"
                    rotated.save(temp_rotated)
                    logger.info("💾 Rotated image saved to temp: %s", temp_rotated)
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
            logger.info("🔎 CellTableParser on angles: %s", angles_to_process)
            images = convert_from_path(
                str(file_path), 
                dpi=self.settings["dpi"],
                first_page=pages_info["first_page"],
                last_page=pages_info["last_page"]
            )
            stem = Path(file_name).stem
            for page_idx, img in enumerate(images, start=1):
                for angle in angles_to_process:
                    rotated = img.rotate(angle, expand=True)
                    img_path = self.screenshots_dir / f"{stem}_cells_p{page_idx}_r{angle}.png"
                    rotated.save(img_path)
                    logger.info("💾 Cell image saved: %s", img_path)
                    
                    # Сохраняем повернутое изображение в temp для анализа
                    temp_cell = self.temp_dir / f"{stem}_cell_analysis_p{page_idx}_r{angle}.png"
                    rotated.save(temp_cell)
                    logger.info("💾 Cell analysis image saved to temp: %s", temp_cell)

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

        # 4) MarkItDown парсер (работает с оригинальным PDF файлом)
        if use_markitdown and self.markitdown_parser is not None:
            logger.info("🔄 MarkItDown парсинг")
            try:
                markitdown_tables = self.markitdown_parser.extract_tables(str(file_path))
                for mt in markitdown_tables:
                    final_tables.append({
                        "data": mt["data"],
                        "sheet_name": mt["sheet_name"],
                        "source": mt["source"],
                        "rotation": 0,  # MarkItDown работает с оригинальным файлом
                        "cleaning_method": mt.get("cleaning_method", "markdown_parsing")
                    })
                    logger.info("✅ %s → %d×%d", mt["sheet_name"], *mt["data"].shape)
            except Exception as e:
                logger.warning("MarkItDown failed: %s", e)
        elif use_markitdown and self.markitdown_parser is None:
            logger.warning("MarkItDown Parser запрошен, но не доступен (модуль не установлен)")

        # 5) PaddleOCR парсер (работает с изображениями страниц)
        if use_paddleocr and self.paddleocr_parser is not None:
            logger.info("🔄 PaddleOCR PP-OCRv5 парсинг")
            try:
                # Конвертируем PDF страницы в изображения для PaddleOCR
                images = convert_from_path(
                    str(file_path), 
                    dpi=self.settings["dpi"],
                    first_page=pages_info["first_page"],
                    last_page=pages_info["last_page"]
                )
                
                for page_idx, image in enumerate(images, start=pages_info["first_page"] or 1):
                    # Сохраняем изображение во временный файл
                    temp_image_path = self.temp_dir / f"paddleocr_page_{page_idx}.png"
                    image.save(temp_image_path)
                    
                    # Обрабатываем изображение через PaddleOCR
                    paddleocr_tables = self.paddleocr_parser.extract_tables(str(temp_image_path))
                    
                    for pt in paddleocr_tables:
                        final_tables.append({
                            "data": pt["data"],
                            "sheet_name": f"{pt['sheet_name']}_page_{page_idx}",
                            "source": pt["source"],
                            "rotation": 0,  # PaddleOCR обрабатывает изображения напрямую
                            "cleaning_method": pt.get("cleaning_method", "paddleocr_pp-ocrv5")
                        })
                        logger.info("✅ %s → %d×%d", pt["sheet_name"], *pt["data"].shape)
                    
                    # Удаляем временный файл
                    temp_image_path.unlink(missing_ok=True)
                    
            except Exception as e:
                logger.warning("PaddleOCR PP-OCRv5 failed: %s", e)
        elif use_paddleocr and self.paddleocr_parser is None:
            logger.warning("PaddleOCR Parser запрошен, но не доступен (модуль не установлен)")

        # 6) DocTR парсер (работает с изображениями страниц)
        use_doctr = opts.get("use_doctr", False)
        if use_doctr and self.doctr_parser is not None:
            logger.info("🔄 DocTR (Mindee) парсинг")
            try:
                # DocTR может работать напрямую с PDF файлами
                doctr_tables = self.doctr_parser.extract_tables(str(file_path))
                
                for dt in doctr_tables:
                    final_tables.append({
                        "data": pd.DataFrame(dt),
                        "sheet_name": f"DocTR_Table_{len(final_tables) + 1}",
                        "source": "doctr",
                        "rotation": 0,
                        "cleaning_method": "doctr_mindee"
                    })
                    logger.info("✅ DocTR_Table → %d×%d", len(dt), len(dt[0]) if dt else 0)
                    
            except Exception as e:
                logger.warning("DocTR (Mindee) failed: %s", e)
        elif use_doctr and self.doctr_parser is None:
            logger.warning("DocTR Parser запрошен, но не доступен (модуль не установлен)")

        # 7) LayoutParser парсер (работает с изображениями страниц)
        use_layoutparser = opts.get("use_layoutparser", False)
        if use_layoutparser and self.layoutparser_parser is not None:
            logger.info("🔄 LayoutParser парсинг")
            try:
                # Конвертируем PDF страницы в изображения для LayoutParser
                images = convert_from_path(
                    str(file_path), 
                    dpi=self.settings["dpi"],
                    first_page=pages_info["first_page"],
                    last_page=pages_info["last_page"]
                )
                
                for page_idx, image in enumerate(images, start=pages_info["first_page"] or 1):
                    # Сохраняем изображение во временный файл
                    temp_image_path = self.temp_dir / f"layoutparser_page_{page_idx}.png"
                    image.save(temp_image_path)
                    
                    # Обрабатываем изображение через LayoutParser
                    layoutparser_tables = self.layoutparser_parser.extract_tables(str(temp_image_path))
                    
                    for lt in layoutparser_tables:
                        final_tables.append({
                            "data": pd.DataFrame(lt),
                            "sheet_name": f"LayoutParser_Table_page_{page_idx}_{len(final_tables) + 1}",
                            "source": "layoutparser",
                            "rotation": 0,
                            "cleaning_method": "layoutparser"
                        })
                        logger.info("✅ LayoutParser_Table → %d×%d", len(lt), len(lt[0]) if lt else 0)
                    
                    # Удаляем временный файл
                    temp_image_path.unlink(missing_ok=True)
                    
            except Exception as e:
                logger.warning("LayoutParser failed: %s", e)
        elif use_layoutparser and self.layoutparser_parser is None:
            logger.warning("LayoutParser Parser запрошен, но не доступен (модуль не установлен)")

        # 8) Дополнительные методы парсинга
        if (use_text_structure or use_spacing or use_easyocr or use_spacing_analysis):
            logger.info("🔍 Пробуем дополнительные методы парсинга")
            images = convert_from_path(
                str(file_path), 
                dpi=self.settings["dpi"],
                first_page=pages_info["first_page"],
                last_page=pages_info["last_page"]
            )
            stem = Path(file_name).stem
            
            for page_idx, img in enumerate(images, start=1):
                for angle in angles_to_process:
                    rotated = img.rotate(angle, expand=True)
                    img_path = self.screenshots_dir / f"{stem}_advanced_p{page_idx}_r{angle}.png"
                    rotated.save(img_path)
                    
                    # Сохраняем повернутое изображение в temp для анализа
                    temp_advanced = self.temp_dir / f"{stem}_advanced_analysis_p{page_idx}_r{angle}.png"
                    rotated.save(temp_advanced)
                    logger.info("💾 Advanced analysis image saved to temp: %s", temp_advanced)
                    
                    # Text Structure Parser
                    if use_text_structure:
                        try:
                            text_tables = self.text_structure_parser.extract_tables(str(img_path))
                            for tt in text_tables:
                                sheet = f"{tt['sheet_name']}_p{page_idx}_r{angle}"
                                final_tables.append({
                                    "data": tt["data"],
                                    "sheet_name": sheet,
                                    "source": tt["source"],
                                    "rotation": angle,
                                    "cleaning_method": tt.get("cleaning_method", "unknown")
                                })
                                logger.info("✅ %s → %d×%d", sheet, *tt["data"].shape)
                        except Exception as e:
                            logger.warning("TextStructureParser failed: %s", e)
                    
                    # Spacing Parser
                    if use_spacing:
                        try:
                            spacing_tables = self.spacing_parser.extract_tables(str(img_path))
                            for st in spacing_tables:
                                sheet = f"{st['sheet_name']}_p{page_idx}_r{angle}"
                                final_tables.append({
                                    "data": st["data"],
                                    "sheet_name": sheet,
                                    "source": st["source"],
                                    "rotation": angle,
                                    "cleaning_method": st.get("cleaning_method", "unknown")
                                })
                                logger.info("✅ %s → %d×%d", sheet, *st["data"].shape)
                        except Exception as e:
                            logger.warning("SpacingParser failed: %s", e)
                    
                    # EasyOCR Parser
                    if use_easyocr and self.easyocr_parser is not None:
                        try:
                            easyocr_tables = self.easyocr_parser.extract_tables(str(img_path))
                            for et in easyocr_tables:
                                sheet = f"{et['sheet_name']}_p{page_idx}_r{angle}"
                                final_tables.append({
                                    "data": et["data"],
                                    "sheet_name": sheet,
                                    "source": et["source"],
                                    "rotation": angle,
                                    "cleaning_method": et.get("cleaning_method", "unknown")
                                })
                                logger.info("✅ %s → %d×%d", sheet, *et["data"].shape)
                        except Exception as e:
                            logger.warning("EasyOCRParser failed: %s", e)
                    elif use_easyocr and self.easyocr_parser is None:
                        logger.warning("EasyOCR Parser запрошен, но не доступен (модуль не установлен)")
                    
                    # Spacing Analysis Parser
                    if use_spacing_analysis:
                        try:
                            spacing_analysis_tables = self.spacing_analysis_parser.extract_tables(str(img_path))
                            for sat in spacing_analysis_tables:
                                sheet = f"{sat['sheet_name']}_p{page_idx}_r{angle}"
                                final_tables.append({
                                    "data": sat["data"],
                                    "sheet_name": sheet,
                                    "source": sat["source"],
                                    "rotation": angle,
                                    "cleaning_method": sat.get("cleaning_method", "unknown")
                                })
                                logger.info("✅ %s → %d×%d", sheet, *sat["data"].shape)
                        except Exception as e:
                            logger.warning("SpacingAnalysisParser failed: %s", e)

        logger.info("🏁 Полная обработка завершена, всего таблиц: %d", len(final_tables))
        return final_tables

    def _parse_pages_option(self, pages_opt: str) -> Dict[str, Any]:
        """
        Парсит параметр pages для разных методов парсинга.
        
        Args:
            pages_opt: Строка с номерами страниц (например: "all", "1", "1-3", "1,3,5")
            
        Returns:
            Словарь с параметрами для разных методов
        """
        pages_info = {
            "camelot_tabula": pages_opt,  # Для Camelot и Tabula
            "first_page": None,           # Для pdf2image
            "last_page": None,            # Для pdf2image
            "page_numbers": []            # Список номеров страниц
        }
        
        if pages_opt.lower() == "all":
            # Для всех страниц
            pages_info["camelot_tabula"] = "all"
            pages_info["first_page"] = None
            pages_info["last_page"] = None
            pages_info["page_numbers"] = []
        else:
            # Парсим конкретные страницы
            try:
                if "-" in pages_opt and "," not in pages_opt:
                    # Интервал страниц (например: "1-3")
                    start, end = pages_opt.split("-", 1)
                    start_page = int(start.strip())
                    end_page = int(end.strip())
                    
                    pages_info["camelot_tabula"] = pages_opt
                    pages_info["first_page"] = start_page
                    pages_info["last_page"] = end_page
                    pages_info["page_numbers"] = list(range(start_page, end_page + 1))
                    
                elif "," in pages_opt:
                    # Конкретные страницы через запятую (например: "1,3,5")
                    page_list = [int(p.strip()) for p in pages_opt.split(",")]
                    
                    pages_info["camelot_tabula"] = pages_opt
                    pages_info["first_page"] = min(page_list)
                    pages_info["last_page"] = max(page_list)
                    pages_info["page_numbers"] = page_list
                    
                else:
                    # Одна страница (например: "1")
                    page_num = int(pages_opt.strip())
                    
                    pages_info["camelot_tabula"] = pages_opt
                    pages_info["first_page"] = page_num
                    pages_info["last_page"] = page_num
                    pages_info["page_numbers"] = [page_num]
                    
            except (ValueError, IndexError) as e:
                logger.warning("⚠️ Некорректный формат страниц '%s': %s. Используем страницу 1", pages_opt, e)
                pages_info["camelot_tabula"] = "1"
                pages_info["first_page"] = 1
                pages_info["last_page"] = 1
                pages_info["page_numbers"] = [1]
        
        logger.info("📄 Параметры страниц: %s", pages_info)
        return pages_info

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
