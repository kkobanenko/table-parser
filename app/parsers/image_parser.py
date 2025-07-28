import cv2
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple
import os
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger('image_parser')

class ImageParser:
    """Parser for image files with OCR"""
    
    def __init__(self, ocr_settings: dict):
        self.settings = Settings()
        self.ocr_settings = ocr_settings
        self.screenshot_dir = f"{self.settings.TEMP_DIR}/screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
    
    def extract_tables(self, file_path: str) -> List[Dict]:
        """Extract tables from image using OCR"""
        tables = []
        
        try:
            # Load and preprocess image
            image = self._load_and_preprocess(file_path)
            
            # Save preprocessed image for debugging
            debug_path = os.path.join(self.screenshot_dir, "preprocessed.png")
            cv2.imwrite(debug_path, image)
            
            # Detect table regions
            table_regions = self._detect_table_regions(image)
            logger.info(f"Detected {len(table_regions)} potential table regions")
            
            # Extract tables from regions
            for idx, region in enumerate(table_regions):
                try:
                    table_data = self._extract_table_from_region(image, region, idx)
                    
                    if table_data and not table_data['data'].empty:
                        tables.append(table_data)
                        
                except Exception as e:
                    logger.error(f"Error extracting table {idx}: {e}")
            
            # If no tables detected, try full image OCR
            if not tables:
                logger.info("No table regions detected, trying full image OCR")
                full_ocr = self._perform_full_ocr(image, file_path)
                if full_ocr and not full_ocr['data'].empty:
                    tables.append(full_ocr)
                    
        except Exception as e:
            logger.error(f"Error processing image: {e}")
        
        return tables
    
    def _load_and_preprocess(self, file_path: str) -> np.ndarray:
        """Load and preprocess image for OCR"""
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Auto-rotate if needed
        if self.ocr_settings.get('rotate_auto', True):
            gray = self._auto_rotate(gray)
        
        # Apply preprocessing based on settings
        if self.ocr_settings.get('enhance_contrast', True):
            gray = self._enhance_contrast(gray)
        
        if self.ocr_settings.get('denoise', True):
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Resize based on DPI setting
        target_dpi = self.ocr_settings.get('dpi', 300)
        gray = self._resize_for_dpi(gray, target_dpi)
        
        # Binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _auto_rotate(self, image: np.ndarray) -> np.ndarray:
        """Auto-rotate image based on text orientation"""
        try:
            # Detect orientation using Tesseract
            osd = pytesseract.image_to_osd(image)
            angle = int(osd.split('\n')[2].split(':')[1])
            
            if angle != 0:
                logger.info(f"Rotating image by {angle} degrees")
                # Get rotation matrix
                height, width = image.shape
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Calculate new dimensions
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_width = int((height * sin) + (width * cos))
                new_height = int((height * cos) + (width * sin))
                
                # Adjust rotation matrix
                M[0, 2] += (new_width / 2) - center[0]
                M[1, 2] += (new_height / 2) - center[1]
                
                # Rotate image
                rotated = cv2.warpAffine(image, M, (new_width, new_height), 
                                       borderValue=255)
                return rotated
        except:
            logger.warning("Could not detect orientation, skipping rotation")
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _resize_for_dpi(self, image: np.ndarray, target_dpi: int) -> np.ndarray:
        """Resize image based on target DPI"""
        # Assume original is 72 DPI if unknown
        scale = target_dpi / 72
        if scale > 1.5:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return image
    
    def _detect_table_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential table regions in image"""
        regions = []
        
        # Detect lines
        horizontal_lines = self._detect_lines(image, horizontal=True)
        vertical_lines = self._detect_lines(image, horizontal=False)
        
        # Combine lines
        combined = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that could be tables
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size and aspect ratio
            if (area > image_area * 0.01 and  # At least 1% of image
                area < image_area * 0.9 and   # Not more than 90% of image
                w > 100 and h > 50 and        # Minimum size
                w/h > 0.3 and w/h < 20):      # Reasonable aspect ratio
                
                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                regions.append((x, y, w, h))
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        # Remove overlapping regions
        regions = self._remove_overlapping_regions(regions)
        
        return regions
    
    def _detect_lines(self, image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """Detect horizontal or vertical lines in image"""
        # Create kernel for morphological operations
        if horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Apply morphological operations
        detected = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        return detected
    
    def _remove_overlapping_regions(self, regions: List[Tuple]) -> List[Tuple]:
        """Remove overlapping regions, keeping larger ones"""
        if not regions:
            return regions
        
        kept_regions = []
        
        for i, region1 in enumerate(regions):
            x1, y1, w1, h1 = region1
            keep = True
            
            for j, region2 in enumerate(regions[:i]):
                x2, y2, w2, h2 = region2
                
                # Check if region1 is inside region2
                if (x1 >= x2 and y1 >= y2 and 
                    x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                    keep = False
                    break
            
            if keep:
                kept_regions.append(region1)
        
        return kept_regions
    
    def _extract_table_from_region(self, image: np.ndarray, region: Tuple, idx: int) -> Dict:
        """Extract table data from image region"""
        x, y, w, h = region
        table_img = image[y:y+h, x:x+w]
        
        # Save screenshot of region
        screenshot_path = os.path.join(self.screenshot_dir, f"table_region_{idx}.png")
        cv2.imwrite(screenshot_path, table_img)
        
        # Perform OCR with custom config
        psm = self.ocr_settings.get('psm', 6)
        custom_config = f'--oem 3 --psm {psm} -l {self.settings.OCR_LANGUAGES}'
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(
            table_img, 
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Parse OCR results into table structure
        df, ocr_errors = self._parse_ocr_to_table(ocr_data, idx, screenshot_path)
        
        if df is not None and not df.empty:
            sheet_name = f"OCR_Table_{idx + 1}"
            
            return {
                'data': df,
                'sheet_name': sheet_name,
                'ocr_errors': ocr_errors,
                'screenshot': screenshot_path,
                'region': region
            }
        
        return None
    
    def _parse_ocr_to_table(self, ocr_data: dict, table_idx: int, screenshot_path: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """Parse OCR data into table structure"""
        # Group text by lines
        lines = {}
        ocr_errors = []
        
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            if ocr_data['conf'][i] > 0:  # Valid detection
                text = ocr_data['text'][i].strip()
                if text:
                    line_num = ocr_data['line_num'][i]
                    conf = ocr_data['conf'][i]
                    
                    if line_num not in lines:
                        lines[line_num] = []
                    
                    # Check confidence threshold
                    if conf < self.settings.OCR_CONFIDENCE_THRESHOLD:
                        original_text = text
                        text = f"[OCR_ERROR: {text}]"
                        ocr_errors.append({
                            'filename': screenshot_path,
                            'table_idx': table_idx,
                            'line': line_num,
                            'cell': f"Line{line_num}_Pos{len(lines[line_num])}",
                            'text': original_text,
                            'confidence': conf,
                            'bbox': (ocr_data['left'][i], ocr_data['top'][i], 
                                   ocr_data['width'][i], ocr_data['height'][i])
                        })
                    
                    lines[line_num].append({
                        'text': text,
                        'left': ocr_data['left'][i],
                        'top': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    })
        
        # Convert lines to table structure
        if lines:
            # Sort lines by line number
            sorted_lines = sorted(lines.items(), key=lambda x: x[0])
            
            # Group words into cells based on horizontal position
            rows = []
            
            for line_num, words in sorted_lines:
                if words:
                    # Sort words by horizontal position
                    words.sort(key=lambda x: x['left'])
                    
                    # Group words into cells
                    cells = self._group_words_into_cells(words)
                    
                    if cells:
                        rows.append(cells)
            
            # Create DataFrame
            if rows:
                # Determine the maximum number of columns
                max_cols = max(len(row) for row in rows)
                
                # Pad rows to have same number of columns
                for row in rows:
                    while len(row) < max_cols:
                        row.append('')
                
                # Create DataFrame
                if len(rows) > 1:
                    # Use first row as headers
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                else:
                    df = pd.DataFrame(rows)
                
                # Clean up the DataFrame
                df = self._clean_ocr_dataframe(df)
                
                return df, ocr_errors
        
        return pd.DataFrame(), ocr_errors
    
    def _group_words_into_cells(self, words: List[Dict]) -> List[str]:
        """Group words into cells based on horizontal gaps"""
        if not words:
            return []
        
        cells = []
        current_cell = [words[0]['text']]
        last_right = words[0]['left'] + words[0]['width']
        
        # Calculate average character width
        avg_char_width = sum(w['width'] / max(len(w['text']), 1) for w in words) / len(words)
        gap_threshold = avg_char_width * 3  # Gap of 3 characters indicates new cell
        
        for word in words[1:]:
            gap = word['left'] - last_right
            
            if gap > gap_threshold:
                # Start new cell
                cells.append(' '.join(current_cell))
                current_cell = [word['text']]
            else:
                # Continue current cell
                current_cell.append(word['text'])
            
            last_right = word['left'] + word['width']
        
        # Add last cell
        if current_cell:
            cells.append(' '.join(current_cell))
        
        return cells
    
    def _clean_ocr_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OCR-extracted DataFrame"""
        # Remove empty columns
        df = df.loc[:, (df != '').any(axis=0)]
        
        # Remove empty rows
        df = df[(df != '').any(axis=1)]
        
        # Replace OCR artifacts
        replacements = {
            '|': '',  # Vertical lines often detected as |
            '_': '',  # Underscores from lines
            '—': '-', # Em dash to hyphen
        }
        
        for old, new in replacements.items():
            df = df.replace(old, new, regex=False)
        
        # Strip whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        return df
    
    def _perform_full_ocr(self, image: np.ndarray, original_path: str) -> Dict:
        """Perform OCR on full image as fallback"""
        try:
            psm = self.ocr_settings.get('psm', 6)
            custom_config = f'--oem 3 --psm {psm} -l {self.settings.OCR_LANGUAGES}'
            
            # Get OCR data
            ocr_data = pytesseract.image_to_data(
                image, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Try to extract structured data
            df, ocr_errors = self._parse_ocr_to_table(ocr_data, 0, original_path)
            
            if not df.empty:
                return {
                    'data': df,
                    'sheet_name': 'OCR_FullImage',
                    'ocr_errors': ocr_errors,
                    'method': 'full_image_ocr'
                }
            
            # Fallback to simple text extraction
            text = pytesseract.image_to_string(image, config=custom_config)
            
            if text.strip():
                # Try to parse as simple table
                lines = text.strip().split('\n')
                rows = []
                
                for line in lines:
                    if line.strip():
                        # Split by multiple spaces or tabs
                        import re
                        cells = re.split(r'\s{2,}|\t', line.strip())
                        if len(cells) > 1:  # Only keep lines with multiple cells
                            rows.append(cells)
                
                if rows:
                    df = pd.DataFrame(rows)
                    return {
                        'data': df,
                        'sheet_name': 'OCR_Text',
                        'ocr_errors': [],
                        'method': 'text_parsing'
                    }
                
        except Exception as e:
            logger.error(f"Full image OCR failed: {e}")
        
        return None
