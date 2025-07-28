import pdfplumber
import tabula
import pandas as pd
from typing import List, Dict
from utils.logger import setup_logger

logger = setup_logger('pdf_parser')

class PDFParser:
    """Parser for PDF files"""
    
    def extract_tables(self, file_path: str) -> List[Dict]:
        """Extract tables from PDF file"""
        tables = []
        
        try:
            # Try pdfplumber first
            tables.extend(self._extract_with_pdfplumber(file_path))
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        try:
            # Try tabula as fallback
            tables.extend(self._extract_with_tabula(file_path))
        except Exception as e:
            logger.warning(f"tabula failed: {e}")
        
        # Remove duplicates
        return self._deduplicate_tables(tables)
    
    def _extract_with_pdfplumber(self, file_path: str) -> List[Dict]:
        """Extract tables using pdfplumber"""
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for table_idx, table in enumerate(page_tables):
                    if table and len(table) > 1:  # At least header + 1 row
                        df = pd.DataFrame(table[1:], columns=table[0])
                        
                        # Generate sheet name
                        sheet_name = self._generate_sheet_name(df, page_num, table_idx)
                        
                        tables.append({
                            'data': df,
                            'sheet_name': sheet_name,
                            'source': 'pdfplumber',
                            'page': page_num + 1
                        })
        
        return tables
    
    def _extract_with_tabula(self, file_path: str) -> List[Dict]:
        """Extract tables using tabula"""
        tables = []
        
        try:
            dfs = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            
            for idx, df in enumerate(dfs):
                if not df.empty:
                    sheet_name = self._generate_sheet_name(df, 0, idx)
                    
                    tables.append({
                        'data': df,
                        'sheet_name': sheet_name,
                        'source': 'tabula'
                    })
        except Exception as e:
            logger.error(f"Tabula extraction error: {e}")
        
        return tables
    
    def _generate_sheet_name(self, df: pd.DataFrame, page_num: int, table_idx: int) -> str:
        """Generate sheet name based on content"""
        # Try to use first non-empty cell as base
        for col in df.columns:
            first_val = str(df[col].iloc[0]) if not df.empty else ''
            if first_val and first_val != 'nan':
                # Clean and truncate
                sheet_name = first_val[:20].replace('/', '_').replace('\\', '_')
                return f"{sheet_name}_p{page_num + 1}"
        
        return f"Table_{page_num + 1}_{table_idx + 1}"
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables"""
        unique_tables = []
        seen_hashes = set()
        
        for table in tables:
            # Create hash of table content
            table_hash = pd.util.hash_pandas_object(table['data']).sum()
            
            if table_hash not in seen_hashes:
                seen_hashes.add(table_hash)
                unique_tables.append(table)
        
        return unique_tables
