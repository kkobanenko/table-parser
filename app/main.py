import streamlit as st
import os
from datetime import datetime
from processors.file_processor import FileProcessor
from utils.logger import setup_logger
from config.settings import Settings
import json

# Page configuration
st.set_page_config(
    page_title="Table Parser System",
    page_icon="📊",
    layout="wide"
)

# Initialize logger
logger = setup_logger('main')

# Load settings
settings = Settings()

def main_app():
    """Main application interface"""
    # Header
    st.title("📊 Table Data Parser")
    st.markdown("Extract tables from various file formats and save as XLSX")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Information")
        st.info(f"""
        **Supported formats:**
        - PDF, DOCX, CSV, TXT
        - HTML, XML
        - JPG, JPEG, PNG
        
        **Limits:**
        - Max file size: {settings.MAX_FILE_SIZE_MB} MB
        - Max files per batch: {settings.MAX_FILES_PER_BATCH}
        """)
        
        st.markdown("---")
        
        st.header("📊 Statistics")
        if 'processed_count' not in st.session_state:
            st.session_state.processed_count = 0
        if 'tables_extracted' not in st.session_state:
            st.session_state.tables_extracted = 0
            
        st.metric("Files Processed", st.session_state.processed_count)
        st.metric("Tables Extracted", st.session_state.tables_extracted)
        
        if st.button("🔄 Reset Statistics"):
            st.session_state.processed_count = 0
            st.session_state.tables_extracted = 0
            st.rerun()
    
    # Main content
    st.header("📤 Upload Files")
    
    uploaded_files = st.file_uploader(
        "Choose files to process",
        type=['pdf', 'docx', 'csv', 'txt', 'html', 'xml', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="You can select multiple files at once"
    )
    
    if uploaded_files:
        # Validate file count
        if len(uploaded_files) > settings.MAX_FILES_PER_BATCH:
            st.error(f"❌ Please upload no more than {settings.MAX_FILES_PER_BATCH} files at once.")
            return
        
        # Validate file sizes
        invalid_files = []
        for file in uploaded_files:
            if file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                invalid_files.append(f"{file.name} ({file.size / 1024 / 1024:.1f} MB)")
        
        if invalid_files:
            st.error(f"❌ Following files exceed size limit of {settings.MAX_FILE_SIZE_MB} MB:")
            for f in invalid_files:
                st.error(f"  • {f}")
            return
        
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
        
        # Display uploaded files
        with st.expander("📁 Uploaded Files", expanded=True):
            for file in uploaded_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    st.text(f"{file.size / 1024:.1f} KB")
                with col3:
                    st.text(file.type or "Unknown")
        
        # OCR Settings (for images)
        image_files = [f for f in uploaded_files if f.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            with st.expander("⚙️ OCR Settings", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    ocr_dpi = st.slider("OCR DPI", 100, 400, 300, 
                                       help="Higher DPI may improve recognition but takes longer")
                    ocr_psm = st.selectbox("Page Segmentation Mode", 
                        options=[3, 4, 6, 11, 12],
                        format_func=lambda x: {
                            3: "3 - Fully automatic (recommended)",
                            4: "4 - Single column",
                            6: "6 - Uniform block",
                            11: "11 - Sparse text",
                            12: "12 - Sparse text with OSD"
                        }[x],
                        index=0
                    )
                with col2:
                    enhance_contrast = st.checkbox("Enhance contrast", value=True,
                                                 help="Improve image contrast for better OCR")
                    denoise = st.checkbox("Denoise image", value=True,
                                        help="Remove noise from scanned images")
                    
                st.info("💡 OCR settings only apply to image files (JPG, JPEG, PNG)")
        else:
            ocr_dpi = 300
            ocr_psm = 3
            enhance_contrast = True
            denoise = True
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                process_files(uploaded_files, {
                    'dpi': ocr_dpi,
                    'psm': ocr_psm,
                    'enhance_contrast': enhance_contrast,
                    'denoise': denoise
                })

def process_files(files, ocr_settings):
    """Process uploaded files"""
    processor = FileProcessor(ocr_settings)
    
    # Create containers for progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_file_info = st.empty()
    
    results = []
    errors = []
    all_downloads = []
    
    # Process each file
    for idx, file in enumerate(files):
        progress = idx / len(files)
        progress_bar.progress(progress)
        status_text.text(f"Processing files... ({idx}/{len(files)})")
        current_file_info.info(f"🔄 Processing: {file.name}")
        
        try:
            result = processor.process_file(file)
            results.append(result)
            
            # Store download info
            all_downloads.append({
                'filename': result['output_filename'],
                'data': result['data']
            })
            
            # Update statistics
            st.session_state.processed_count += 1
            st.session_state.tables_extracted += result.get('tables_found', 0)
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            errors.append({
                'filename': file.name,
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    current_file_info.empty()
    
    # Show results
    with results_container:
        st.markdown("---")
        show_processing_summary(results, errors, all_downloads)

def show_processing_summary(results, errors, all_downloads):
    """Display processing summary and download links"""
    
    # Summary metrics
    st.header("📊 Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Successful", len(results), 
                 delta=f"{len(results)}" if len(results) > 0 else None,
                 delta_color="normal")
    with col2:
        st.metric("❌ Failed", len(errors),
                 delta=f"-{len(errors)}" if len(errors) > 0 else "0",
                 delta_color="inverse")
    with col3:
        total_tables = sum(r.get('tables_found', 0) for r in results)
        st.metric("📊 Total Tables", total_tables)
    with col4:
        total_ocr_errors = sum(r.get('ocr_errors', 0) for r in results)
        st.metric("⚠️ OCR Errors", total_ocr_errors,
                 delta=f"-{total_ocr_errors}" if total_ocr_errors > 0 else "0",
                 delta_color="inverse")
    
    # Download section
    if all_downloads:
        st.header("📥 Download Results")
        
        # Individual downloads
        for download in all_downloads:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {download['filename']}")
            with col2:
                st.download_button(
                    label="Download",
                    data=download['data'],
                    file_name=download['filename'],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{download['filename']}"
                )
    
    # Error details
    if errors:
        st.header("⚠️ Error Details")
        with st.expander("View errors", expanded=True):
            for error in errors:
                st.error(f"""
                **File:** {error['filename']}  
                **Error:** {error['error']}  
                **Time:** {error['timestamp']}
                """)
    
    # Generate detailed report
    if results or errors:
        st.header("📄 Detailed Report")
        
        report = generate_detailed_report(results, errors)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                "📥 Download Processing Report (JSON)",
                data=report,
                file_name=f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

def generate_detailed_report(results, errors):
    """Generate comprehensive processing report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_files': len(results) + len(errors),
            'successful': len(results),
            'failed': len(errors),
            'total_tables_extracted': sum(r.get('tables_found', 0) for r in results),
            'total_ocr_errors': sum(r.get('ocr_errors', 0) for r in results)
        },
        'successful_files': [
            {
                'filename': r.get('output_filename', ''),
                'original_filename': r.get('original_filename', ''),
                'tables_found': r.get('tables_found', 0),
                'ocr_errors': r.get('ocr_errors', 0),
                'processing_time': r.get('processing_time', 0)
            }
            for r in results
        ],
        'failed_files': errors,
        'environment': {
            'max_file_size_mb': Settings().MAX_FILE_SIZE_MB,
            'max_files_per_batch': Settings().MAX_FILES_PER_BATCH,
            'ocr_languages': Settings().OCR_LANGUAGES
        }
    }
    
    return json.dumps(report, indent=2, ensure_ascii=False)

def main():
    """Main application entry point"""
    main_app()

if __name__ == "__main__":
    main()
