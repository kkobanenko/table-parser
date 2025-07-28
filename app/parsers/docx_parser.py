from docx import Document
import pandas as pd
from typing import List, Dict
from utils.logger import setup_logger

logger = setup_logger('docx_parser')

class DOCXParser:
    """Parser for DOCX files"""
    
    def extract_tables(self, file_path: str) -> List[Dict]:
        """Extract tables from DOCX file"""
        tables = []
        doc = Document(file_path)
        
        for table_idx, table in enumerate(doc.tables):
            try:
                data = []
                
                # Extract data from table
                for row_idx, row in enumerate(table.rows):
                    row_data = []
                    for cell in row.cells:
                        # Handle merged cells
                        cell_text = cell.text.strip()
                        row_data.append(cell_text)
                    
                    # Skip completely empty rows
                    if any(cell for cell in row_data):
                        data.append(row_data)
                
                if len(data) > 1:  # At least header + 1 row
                    # Create DataFrame
                    df = pd.DataFrame(data[1:], columns=data[0])
                    
                    # Clean DataFrame
                    df = self._clean_dataframe(df)
                    
                    if not df.empty:
                        # Generate sheet name
                        sheet_name = self._generate_sheet_name(df, table_idx)
                        
                        tables.append({
                            'data': df,
                            'sheet_name': sheet_name,
                            'table_index': table_idx
                        })
                    
            except Exception as e:
                logger.error(f"Error parsing table {table_idx}: {e}")
        
        return tables
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame"""
        # Remove completely empty columns
        df = df.loc[:, (df != '').any(axis=0)]
        
        # Remove completely empty rows
        df = df[(df != '').any(axis=1)]
        
        # Replace empty strings with NaN
        df = df.replace('', pd.NA)
        
        # Handle merged cells (forward fill for empty cells in same column)
        for col in df.columns:
            # Only fill if there's a pattern suggesting merged cells
            mask = df[col].isna()
            if mask.any() and not mask.all():
                # Check if NaN values are between non-NaN values
                non_na_indices = df.index[df[col].notna()].tolist()
                if len(non_na_indices) > 1:
                    # Forward fill between non-NaN values
                    df[col] = df[col].fillna(method='ffill')
        
        return df
    
    def _generate_sheet_name(self, df: pd.DataFrame, table_idx: int) -> str:
        """Generate sheet name based on content"""
        # Try to use first meaningful header
        for col in df.columns:
            if col and str(col) != 'nan' and len(str(col)) > 2:
                sheet_name = str(col)[:25].strip()
                # Remove invalid Excel sheet name characters
                for char in ['/', '\\', '?', '*', '[', ']', ':']:
                    sheet_name = sheet_name.replace(char, '_')
                return sheet_name
        
        # Try first cell content
        if not df.empty:
            for col in df.columns:
                first_cell = str(df[col].iloc[0])
                if first_cell and first_cell != 'nan' and len(first_cell) > 2:
                    sheet_name = first_cell[:25].strip()
                    for char in ['/', '\\', '?', '*', '[', ']', ':']:
                        sheet_name = sheet_name.replace(char, '_')
                    return sheet_name
        
        return f"Table_{table_idx + 1}"
