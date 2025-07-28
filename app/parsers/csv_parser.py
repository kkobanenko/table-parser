import pandas as pd
import chardet
from typing import List, Dict
from utils.logger import setup_logger
import csv

logger = setup_logger('csv_parser')

class CSVParser:
    """Parser for CSV files"""
    
    def extract_tables(self, file_path: str) -> List[Dict]:
        """Extract table from CSV file"""
        tables = []
        
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            
            # Detect delimiter
            delimiter = self._detect_delimiter(file_path, encoding)
            
            # Read CSV
            df = pd.read_csv(
                file_path, 
                encoding=encoding,
                delimiter=delimiter,
                on_bad_lines='skip',
                keep_default_na=True,
                na_values=['', 'NA', 'N/A', 'null', 'NULL']
            )
            
            # Check if DataFrame is valid
            if not df.empty and len(df.columns) > 1:
                # Clean data
                df = self._clean_dataframe(df)
                
                # Generate sheet name
                sheet_name = self._generate_sheet_name(df)
                
                tables.append({
                    'data': df,
                    'sheet_name': sheet_name,
                    'delimiter': delimiter,
                    'encoding': encoding
                })
                logger.info(f"Successfully parsed CSV with {len(df)} rows and {len(df.columns)} columns")
            else:
                logger.warning("CSV file appears to be empty or has only one column")
                    
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            
        return tables
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
        
        encoding = result['encoding']
        confidence = result['confidence']
        
        # Fallback encodings
        if not encoding or confidence < 0.7:
            # Try common encodings
            for enc in ['utf-8', 'cp1251', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        f.read(1000)
                    encoding = enc
                    break
                except:
                    continue
        
        encoding = encoding or 'utf-8'
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        return encoding
    
    def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect CSV delimiter"""
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            # Read sample of file
            sample = f.read(5000)
            
            # Try to detect using csv.Sniffer
            try:
                dialect = csv.Sniffer().sniff(sample)
                return dialect.delimiter
            except:
                pass
            
            # Manual detection
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                counts = [line.count(delimiter) for line in sample.split('\n')[:10]]
                # Check if delimiter appears consistently
                if counts and all(c == counts[0] for c in counts) and counts[0] > 0:
                    delimiter_counts[delimiter] = counts[0]
            
            if delimiter_counts:
                # Return delimiter with highest count
                return max(delimiter_counts, key=delimiter_counts.get)
            
            # Default to comma
            return ','
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame"""
        # Remove completely empty rows/columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Remove unnamed columns that are likely index columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols and len(unnamed_cols) == 1:
            # Check if it's just sequential numbers (likely an index)
            try:
                if pd.to_numeric(df[unnamed_cols[0]], errors='coerce').notna().all():
                    df = df.drop(columns=unnamed_cols)
            except:
                pass
        
        return df
    
    def _generate_sheet_name(self, df: pd.DataFrame) -> str:
        """Generate sheet name"""
        # Use first meaningful column name
        for col in df.columns:
            col_str = str(col)
            if col_str and not col_str.startswith('Unnamed') and len(col_str) > 2:
                sheet_name = col_str[:25].strip()
                for char in ['/', '\\', '?', '*', '[', ']', ':']:
                    sheet_name = sheet_name.replace(char, '_')
                return sheet_name
        
        return "CSV_Data"
