import os
import tempfile
from datetime import datetime
from typing import Dict, List
import pandas as pd
from io import BytesIO
import json
import time

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from parsers.csv_parser import CSVParser
from parsers.html_xml_parser import HTMLXMLParser
from parsers.image_parser import ImageParser
from processors.table_processor import TableProcessor
from utils.logger import setup_logger

logger = setup_logger('file_processor')

class FileProcessor:
    """Main file processing orchestrator"""
    
    def __init__(self, ocr_settings: dict = None):
        self.ocr_settings = ocr_settings or {}
        self.pdf_parser = PDFParser()
        self.docx_parser = DOCXParser()
        self.csv_parser = CSVParser()
        self.html_xml_parser = HTMLXMLParser()
        self.image_parser = ImageParser(self.ocr_settings)
        self.table_processor = TableProcessor()
    
    def process_file(self, uploaded_file) -> Dict:
        """Process single uploaded file"""
        start_time = time.time()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Extract file extension
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            logger.info(f"Processing file: {uploaded_file.name} (type: {file_ext})")
            
            # Parse based on file type
            tables = self._parse_file(tmp_path, file_ext)
            
            logger.info(f"Found {len(tables)} tables in {uploaded_file.name}")
            
            # Process tables
            processed_tables = []
            total_ocr_errors = 0
            
            for table_data in tables:
                processed = self.table_processor.process_table(
                    table_data['data'],
                    table_data.get('sheet_name', 'Sheet1')
                )
                
                # Add OCR error info if present
                ocr_errors = table_data.get('ocr_errors', [])
                processed['ocr_errors'] = ocr_errors
                total_ocr_errors += len(ocr_errors)
                
                # Add source info
                processed['source'] = table_data.get('source', file_ext)
                processed['page'] = table_data.get('page', None)
                
                processed_tables.append(processed)
            
            # Create Excel file
            output_bytes = self._create_excel(processed_tables, uploaded_file.name)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(uploaded_file.name)[0]
            output_filename = f"{base_name}_{timestamp}.xlsx"
            
            # Calculate processing time
            processing_time = round(time.time() - start_time, 2)
            
            return {
                'data': output_bytes,
                'output_filename': output_filename,
                'original_filename': uploaded_file.name,
                'tables_found': len(processed_tables),
                'status': 'success',
                'ocr_errors': total_ocr_errors,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}", exc_info=True)
            raise
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _parse_file(self, file_path: str, file_ext: str) -> List[Dict]:
        """Parse file based on extension"""
        logger.info(f"Parsing file with extension: {file_ext}")
        
        if file_ext == 'pdf':
            return self.pdf_parser.extract_tables(file_path)
        elif file_ext in ['docx', 'doc']:
            return self.docx_parser.extract_tables(file_path)
        elif file_ext == 'csv':
            return self.csv_parser.extract_tables(file_path)
        elif file_ext == 'txt':
            # Treat as CSV with unknown delimiter
            return self.csv_parser.extract_tables(file_path)
        elif file_ext in ['html', 'htm']:
            return self.html_xml_parser.extract_tables(file_path, 'html')
        elif file_ext == 'xml':
            return self.html_xml_parser.extract_tables(file_path, 'xml')
        elif file_ext in ['jpg', 'jpeg', 'png']:
            return self.image_parser.extract_tables(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _create_excel(self, tables: List[Dict], original_filename: str) -> bytes:
        """Create Excel file with processed tables"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Track used sheet names to avoid duplicates
            used_names = set()
            
            for idx, table_info in enumerate(tables):
                df = table_info['dataframe']
                sheet_name = table_info['sheet_name']
                
                # Ensure unique sheet names
                sheet_name = self._ensure_unique_sheet_name(sheet_name, used_names)
                used_names.add(sheet_name)
                
                # Write to Excel
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Format worksheet
                worksheet = writer.sheets[sheet_name]
                self._format_worksheet(worksheet, df, table_info)
                
                # Add metadata as comments
                self._add_metadata(worksheet, table_info)
        
        output.seek(0)
        return output.getvalue()
    
    def _ensure_unique_sheet_name(self, sheet_name: str, used_names: set) -> str:
        """Ensure sheet name is unique and valid"""
        # Excel sheet name limitations
        invalid_chars = ['\\', '/', '?', '*', '[', ']', ':']
        for char in invalid_chars:
            sheet_name = sheet_name.replace(char, '_')
        
        # Truncate to 31 characters (Excel limit)
        sheet_name = sheet_name[:31]
        
        # Make unique if needed
        original = sheet_name
        counter = 1
        while sheet_name in used_names:
            # Leave room for counter
            max_length = 31 - len(str(counter)) - 1
            sheet_name = f"{original[:max_length]}_{counter}"
            counter += 1
        
        return sheet_name
    
    def _format_worksheet(self, worksheet, df, table_info):
        """Apply formatting to worksheet"""
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        
        # Header formatting
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")
        
        # Border style
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Apply header formatting
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border
        
        # Apply borders to all cells
        for row in worksheet.iter_rows(min_row=2, max_row=len(df)+1):
            for cell in row:
                cell.border = thin_border
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Freeze header row
        worksheet.freeze_panes = 'A2'
        
        # Highlight OCR errors if any
        if table_info.get('ocr_errors'):
            error_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
            error_font = Font(color="9C0006")
            
            for error in table_info['ocr_errors']:
                # This would need more sophisticated mapping of OCR errors to cells
                # For now, just mark cells containing [OCR_ERROR:
                for row in worksheet.iter_rows(min_row=2):
                    for cell in row:
                        if cell.value and '[OCR_ERROR:' in str(cell.value):
                            cell.fill = error_fill
                            cell.font = error_font
    
    def _add_metadata(self, worksheet, table_info):
        """Add metadata as comments to worksheet"""
        from openpyxl.comments import Comment
        
        # Add comment to A1 with table info
        metadata = []
        
        if table_info.get('source'):
            metadata.append(f"Source: {table_info['source']}")
        
        if table_info.get('page'):
            metadata.append(f"Page: {table_info['page']}")
        
        if table_info.get('ocr_errors'):
            metadata.append(f"OCR Errors: {len(table_info['ocr_errors'])}")
        
        if table_info.get('processed'):
            metadata.append("Status: Successfully processed")
        else:
            metadata.append(f"Status: Processing failed - {table_info.get('error', 'Unknown error')}")
        
        if metadata:
            comment_text = '\n'.join(metadata)
            comment = Comment(comment_text, "Table Parser")
            worksheet['A1'].comment = comment
