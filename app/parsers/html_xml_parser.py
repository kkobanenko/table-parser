from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import lxml.etree as ET
from utils.logger import setup_logger

logger = setup_logger('html_xml_parser')

class HTMLXMLParser:
    """Parser for HTML and XML files"""
    
    def extract_tables(self, file_path: str, file_type: str) -> List[Dict]:
        """Extract tables from HTML or XML file"""
        file_type = file_type.lower()
        
        if file_type in ['html', 'htm']:
            return self._extract_from_html(file_path)
        elif file_type == 'xml':
            return self._extract_from_xml(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_from_html(self, file_path: str) -> List[Dict]:
        """Extract tables from HTML"""
        tables = []
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1251']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except:
                continue
        else:
            logger.error("Could not decode HTML file")
            return tables
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all tables
        html_tables = soup.find_all('table')
        
        for idx, table in enumerate(html_tables):
            try:
                # Parse table to DataFrame
                df = self._parse_html_table(table)
                
                if not df.empty and len(df) > 1:
                    sheet_name = self._generate_sheet_name(df, idx)
                    
                    tables.append({
                        'data': df,
                        'sheet_name': sheet_name,
                        'table_index': idx
                    })
                    
            except Exception as e:
                logger.error(f"Error parsing HTML table {idx}: {e}")
        
        # If no tables found, try to find table-like structures
        if not tables:
            tables.extend(self._find_div_tables(soup))
        
        return tables
    
    def _parse_html_table(self, table) -> pd.DataFrame:
        """Parse HTML table to DataFrame"""
        rows = []
        headers = []
        
        # Extract headers from thead
        thead = table.find('thead')
        if thead:
            header_rows = thead.find_all('tr')
            if header_rows:
                # Handle multi-row headers
                for row in header_rows:
                    row_headers = []
                    for cell in row.find_all(['th', 'td']):
                        colspan = int(cell.get('colspan', 1))
                        cell_text = cell.get_text(strip=True)
                        row_headers.extend([cell_text] * colspan)
                    
                    if not headers:
                        headers = row_headers
                    else:
                        # Merge multi-row headers
                        headers = [f"{h1} {h2}" if h2 else h1 
                                 for h1, h2 in zip(headers, row_headers)]
        
        # If no thead, check first row
        if not headers:
            first_row = table.find('tr')
            if first_row:
                # Check if first row contains th elements
                if first_row.find('th'):
                    headers = [cell.get_text(strip=True) 
                             for cell in first_row.find_all(['th', 'td'])]
        
        # Extract data rows
        tbody = table.find('tbody') or table
        data_rows = tbody.find_all('tr')
        
        # Skip header row if it's in tbody
        start_idx = 1 if headers and not thead else 0
        
        for row in data_rows[start_idx:]:
            row_data = []
            for cell in row.find_all(['td', 'th']):
                colspan = int(cell.get('colspan', 1))
                cell_text = cell.get_text(strip=True)
                row_data.extend([cell_text] * colspan)
            
            if row_data:  # Skip empty rows
                rows.append(row_data)
        
        # Create DataFrame
        if rows:
            if headers:
                # Ensure headers match data width
                max_cols = max(len(row) for row in rows)
                if len(headers) < max_cols:
                    headers.extend([f'Column_{i}' for i in range(len(headers), max_cols)])
                elif len(headers) > max_cols:
                    headers = headers[:max_cols]
                
                df = pd.DataFrame(rows, columns=headers)
            else:
                df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame()
        
        return df
    
    def _find_div_tables(self, soup) -> List[Dict]:
        """Find table-like structures in divs"""
        tables = []
        
        # Look for common table-like class names
        table_classes = ['table', 'grid', 'data-table', 'price-list']
        
        for class_name in table_classes:
            divs = soup.find_all('div', class_=lambda x: x and class_name in x)
            
            for idx, div in enumerate(divs):
                # Try to extract structured data
                rows = div.find_all('div', class_=lambda x: x and 'row' in x)
                if len(rows) > 1:
                    data = []
                    for row in rows:
                        cells = row.find_all(['div', 'span'])
                        if cells:
                            row_data = [cell.get_text(strip=True) for cell in cells]
                            data.append(row_data)
                    
                    if data:
                        df = pd.DataFrame(data)
                        if not df.empty:
                            tables.append({
                                'data': df,
                                'sheet_name': f'Div_Table_{idx + 1}',
                                'source': 'div_structure'
                            })
        
        return tables
    
    def _extract_from_xml(self, file_path: str) -> List[Dict]:
        """Extract tables from XML"""
        tables = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Remove namespaces for easier parsing
            for elem in root.iter():
                elem.tag = elem.tag.split('}')[-1]
            
            # Look for table-like structures
            tables.extend(self._extract_xml_tables(root))
            
            # Look for repeating structures
            tables.extend(self._extract_xml_repeating_structures(root))
            
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
        
        return tables
    
    def _extract_xml_tables(self, root) -> List[Dict]:
        """Extract explicit table structures from XML"""
        tables = []
        
        # Look for elements named 'table', 'Table', etc.
        for table_elem in root.findall('.//table') + root.findall('.//Table'):
            rows = []
            
            # Look for row elements
            for row_elem in table_elem.findall('.//row') + table_elem.findall('.//Row'):
                row_data = {}
                
                # Extract all child elements as columns
                for child in row_elem:
                    row_data[child.tag] = child.text or ''
                
                if row_data:
                    rows.append(row_data)
            
            if rows:
                df = pd.DataFrame(rows)
                tables.append({
                    'data': df,
                    'sheet_name': f'XML_Table_{len(tables) + 1}',
                    'source': 'xml_table'
                })
        
        return tables
    
    def _extract_xml_repeating_structures(self, root) -> List[Dict]:
        """Extract repeating structures that might be tables"""
        tables = []
        
        # Find elements that repeat (potential rows)
        element_counts = {}
        for elem in root.iter():
            if len(elem) > 0:  # Has children
                key = (elem.tag, tuple(child.tag for child in elem))
                element_counts[key] = element_counts.get(key, 0) + 1
        
        # Process elements that repeat more than once
        processed_tags = set()
        
        for (tag, child_structure), count in element_counts.items():
            if count > 1 and tag not in processed_tags:
                processed_tags.add(tag)
                
                # Extract all instances
                rows = []
                for elem in root.findall(f'.//{tag}'):
                    if tuple(child.tag for child in elem) == child_structure:
                        row_data = {}
                        for child in elem:
                            row_data[child.tag] = child.text or ''
                        
                        # Include attributes
                        for attr_name, attr_value in elem.attrib.items():
                            row_data[f'@{attr_name}'] = attr_value
                        
                        if row_data:
                            rows.append(row_data)
                
                if len(rows) > 1:  # At least 2 rows
                    df = pd.DataFrame(rows)
                    if not df.empty:
                        tables.append({
                            'data': df,
                            'sheet_name': f'XML_{tag}',
                            'source': 'xml_repeating'
                        })
        
        return tables
    
    def _generate_sheet_name(self, df: pd.DataFrame, idx: int) -> str:
        """Generate sheet name"""
        if not df.empty and len(df.columns) > 0:
            # Use first column name if meaningful
            first_col = str(df.columns[0])
            if first_col and len(first_col) > 2 and first_col != '0':
                sheet_name = first_col[:20].strip()
                for char in ['/', '\\', '?', '*', '[', ']', ':']:
                    sheet_name = sheet_name.replace(char, '_')
                return f"{sheet_name}_t{idx + 1}"
        
        return f"Table_{idx + 1}"
