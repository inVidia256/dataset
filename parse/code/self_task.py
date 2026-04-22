import time
import os
from abc import ABC, abstractmethod
from pdf2image import convert_from_path
import fitz  # PyMuPDF for PDF image extraction
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm

# 定义任务指令
TASK_INSTRUCTIONS = {
    'text': 'Please output the text content from the image.',
    'formula': 'Please write out the expression of the formula in the image using LaTeX format.',
    'table': 'Please output the table in the image in LaTeX format.'
}


class DataWriter(ABC):
    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write the data to the file.

        Args:
            path (str): the target file where to write
            data (bytes): the data want to write
        """
        pass

    def write_string(self, path: str, data: str) -> None:
        """Write the data to file, the data will be encoded to bytes.

        Args:
            path (str): the target file where to write
            data (str): the data want to write
        """

        def safe_encode(data: str, method: str):
            try:
                bit_data = data.encode(encoding=method, errors='replace')
                return bit_data, True
            except:  # noqa
                return None, False

        for method in ['utf-8', 'ascii']:
            bit_data, flag = safe_encode(data, method)
            if flag:
                self.write(path, bit_data)
                break


class FileBasedDataWriter(DataWriter):
    def __init__(self, parent_dir: str = '') -> None:
        """Initialized with parent_dir.

        Args:
            parent_dir (str, optional): the parent directory that may be used within methods. Defaults to ''.
        """
        self._parent_dir = parent_dir

    def write(self, path: str, data: bytes) -> None:
        """Write file with data.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            data (bytes): the data want to write
        """
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        if not os.path.exists(os.path.dirname(fn_path)) and os.path.dirname(fn_path) != "":
            os.makedirs(os.path.dirname(fn_path), exist_ok=True)

        with open(fn_path, 'wb') as f:
            f.write(data)


def single_task_recognition(input_file, output_dir, OCR_model, task):
    """
    Single task recognition for specific content type

    Args:
        input_file: Input file path
        output_dir: Output directory
        TaichuOCR_model: Pre-initialized model instance
        task: Task type ('text', 'formula', 'table')
    """
    print(f"Starting single task recognition: {task}")
    print(f"Processing file: {input_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Get filename
    name_without_suff = '.'.join(os.path.basename(input_file).split(".")[:-1])

    # Prepare output directory
    local_md_dir = os.path.join(output_dir, name_without_suff)
    os.makedirs(local_md_dir, exist_ok=True)

    print(f"Output dir: {local_md_dir}")
    md_writer = FileBasedDataWriter(local_md_dir)

    # Get task instruction
    instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS['text'])

    # Check file type and prepare images
    file_extension = input_file.split(".")[-1].lower()
    images = []

    if file_extension == 'pdf':
        print("⚠️  WARNING: PDF input detected for single task recognition.")
        print("⚠️  WARNING: Converting all PDF pages to images for processing.")
        print("⚠️  WARNING: This may take longer and use more resources than image input.")
        print("⚠️  WARNING: Consider using individual images for better performance.")

        try:
            # Convert PDF pages to PIL images directly
            print("Converting PDF pages to images...")
            images = convert_from_path(input_file, dpi=150)
            print(f"Converted {len(images)} pages to images")

        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")

    elif file_extension in ['jpg', 'jpeg', 'png']:
        # Load single image
        from PIL import Image
        images = [Image.open(input_file)]
    else:
        raise ValueError(f"Single task recognition supports PDF and image files, got: {file_extension}")

    # Start recognition
    print(f"Performing {task} recognition on {len(images)} image(s)...")
    start_time = time.time()

    try:
        # Prepare instructions for all images
        instructions = [instruction] * len(images)

        # Use chat model for single task recognition with PIL images directly
        # responses = OCR_model.chat_model.batch_inference(images, instructions)

        responses = OCR_model.chat_model.batch_inference(images, instructions)

        recognition_time = time.time() - start_time
        print(f"Recognition time: {recognition_time:.2f}s")

        # Combine results
        combined_result = responses[0]
        for i, response in enumerate(responses):
            if i > 0:
                combined_result = combined_result + "\n\n" + response

        # Save result
        result_filename = f"{name_without_suff}_{task}_result.md"
        md_writer.write(result_filename, combined_result.encode('utf-8'))

        print(f"Single task recognition completed!")
        print(f"Task: {task}")
        print(f"Processed {len(images)} image(s)")
        print(f"Result saved to: {os.path.join(local_md_dir, result_filename)}")

        # Clean up resources
        try:
            # Give some time for async tasks to complete
            time.sleep(0.5)

            # Close images if they were opened
            for img in images:
                if hasattr(img, 'close'):
                    img.close()

        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")

        return local_md_dir

    except Exception as e:
        raise RuntimeError(f"Single task recognition failed: {str(e)}")


def parse_pdf(input_file, output_dir, OCR_model):
    """
    Parse PDF file and save results

    Args:
        input_file: Input PDF file path
        output_dir: Output directory
        OCR_model: Pre-initialized model instance
    """
    print(f"Starting to parse file: {input_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Get filename
    name_without_suff = '.'.join(os.path.basename(input_file).split(".")[:-1])

    # Prepare output directory
    local_image_dir = os.path.join(output_dir, name_without_suff, "images")
    local_md_dir = os.path.join(output_dir, name_without_suff)
    image_dir = os.path.basename(local_image_dir)
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)

    print(f"Output dir: {local_md_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)

    # Read file content
    reader = FileBasedDataReader()
    file_bytes = reader.read(input_file)

    # Create dataset instance
    file_extension = input_file.split(".")[-1].lower()
    if file_extension == "pdf":
        ds = PymuDocDataset(file_bytes)
    else:
        ds = ImageDataset(file_bytes)

    # Start inference
    print("Performing document parsing...")
    start_time = time.time()

    infer_result = ds.apply(doc_analyze_llm, OCR_model=OCR_model)

    # Pipeline processing
    pipe_result = infer_result.pipe_ocr_mode(image_writer, OCR_model)

    parsing_time = time.time() - start_time
    print(f"Parsing time: {parsing_time:.2f}s")

    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

    # Layout and spans PDF output disabled (not needed)
    # pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))
    # pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

    # Extract and save different content types separately
    print("Extracting content by type...")
    # Use original model output (infer_result) for extraction to avoid pdf_parse_union issues
    extract_content_by_type(infer_result, md_writer, local_md_dir, name_without_suff, file_bytes, ds)

    print("Results saved to ", local_md_dir)
    return local_md_dir


def extract_content_by_type(infer_result, md_writer, output_dir, name_without_suff, pdf_bytes=None, dataset=None):
    """
    Extract text, figures, formulas, and tables directly from model output.
    
    Output structure:
    - {name}_text.json: Text content only
    - {name}_formulas.json: Formula content only
    - figures/ folder: Cropped figure images with serial numbers
    - tables/ folder: Cropped table images with serial numbers
    - {name}_figures.json: Figure metadata with serial numbers and paths
    - {name}_tables.json: Table metadata with serial numbers and paths
    """
    from magic_pdf.config.ocr_content_type import BlockType
    from magic_pdf.config.ocr_content_type import CategoryId
    from magic_pdf.model.magic_model import MagicModel
    import copy
    import json
    import os
    
    # Get original model inference results (layout_dets format)
    # infer_result is InferenceResultLLM with _infer_res = [{'page_info': {}, 'layout_dets': []}, ...]
    model_list = infer_result.get_infer_res() if hasattr(infer_result, 'get_infer_res') else infer_result
    
    # Use MagicModel to convert coordinates from image space to PDF space
    # This ensures bbox coordinates match what draw_model() uses
    if dataset is not None:
        magic_model = MagicModel(copy.deepcopy(model_list), dataset)
        # After MagicModel init, the model_list inside it has been modified with scaled bboxes
        # Use the internal model_list which has scaled coordinates
        scaled_model_list = magic_model._MagicModel__model_list
    else:
        magic_model = None
        scaled_model_list = model_list

    # Initialize containers for different content types
    text_content = []
    figure_content = []
    table_content = []
    formula_content = []

    # Create subdirectories for figures and tables
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Open PDF for image extraction if bytes available
    pdf_doc = None
    if pdf_bytes:
        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            print(f"Warning: Could not open PDF for image extraction: {e}")

    # Counters for serial numbers
    figure_counter = 1
    table_counter = 1

    # Process each page's layout_dets directly from model output
    for page_idx, page_data in enumerate(scaled_model_list):
        if isinstance(page_data, dict) and 'layout_dets' in page_data:
            layout_dets = page_data['layout_dets']
        else:
            continue

        for det in layout_dets:
            category_id = det.get('category_id', 1)
            bbox = det.get('bbox', [0, 0, 0, 0])
            
            # Convert category_id to int if it's string
            if isinstance(category_id, str):
                try:
                    category_id = int(category_id)
                except ValueError:
                    continue
            
            # Get text/latex content if available
            content = det.get('text', '')
            latex = det.get('latex', '')
            
            # Extract text content (Text=1, Title=0, OcrText=15)
            if category_id in [CategoryId.Text, CategoryId.Title, CategoryId.OcrText]:
                if content.strip():
                    text_content.append({
                        'page': page_idx + 1,
                        'type': 'text' if category_id == CategoryId.Text else ('title' if category_id == CategoryId.Title else 'ocr_text'),
                        'bbox': bbox,
                        'content': content
                    })
            
            # Extract titles specifically
            elif category_id == CategoryId.Title:
                if content.strip():
                    text_content.append({
                        'page': page_idx + 1,
                        'type': 'title',
                        'bbox': bbox,
                        'content': content
                    })
            
            # Extract inline equations (13)
            elif category_id == CategoryId.InlineEquation:
                if latex or content:
                    formula_content.append({
                        'page': page_idx + 1,
                        'type': 'inline_formula',
                        'bbox': bbox,
                        'content': content or latex,
                        'latex': latex
                    })
            
            # Extract interline equations (8 or 14)
            elif category_id in [CategoryId.InterlineEquation_Layout, CategoryId.InterlineEquation_YOLO]:
                if latex or content:
                    formula_content.append({
                        'page': page_idx + 1,
                        'type': 'formula',
                        'bbox': bbox,
                        'content': content or latex,
                        'latex': latex
                    })
        
        # Extract figures and tables with captions using MagicModel
        if magic_model is not None:
            # Get figures with captions for this page
            page_figures = magic_model.get_imgs(page_idx)
            for fig in page_figures:
                figure_info = extract_figure_with_caption(
                    fig, page_idx, figure_counter, figures_dir, pdf_doc, scaled_model_list
                )
                if figure_info:
                    figure_content.append(figure_info)
                    figure_counter += 1
            
            # Get tables with captions for this page
            page_tables = magic_model.get_tables(page_idx)
            for table in page_tables:
                table_info = extract_table_with_caption(
                    table, page_idx, table_counter, tables_dir, pdf_doc, scaled_model_list
                )
                if table_info:
                    table_content.append(table_info)
                    table_counter += 1

    # Close PDF document
    if pdf_doc:
        pdf_doc.close()

    # Save text and formulas to JSON (text content only)
    save_json_content(md_writer, name_without_suff, 'text', text_content)
    save_json_content(md_writer, name_without_suff, 'formulas', formula_content)

    # Save figures metadata to JSON (with serial numbers and paths)
    save_json_content(md_writer, name_without_suff, 'figures', figure_content)
    print(f"  ✓ Figures: {len(figure_content)} items, images saved to figures/")

    # Save tables metadata to JSON (with serial numbers and paths)
    save_json_content(md_writer, name_without_suff, 'tables', table_content)
    print(f"  ✓ Tables: {len(table_content)} items, images saved to tables/")

    # Save summary
    save_text_summary(md_writer, name_without_suff, text_content, figure_content, table_content, formula_content)


def save_json_content(md_writer, name_without_suff, content_type, content_list):
    """Save content to JSON file."""
    import json
    file_path = f"{name_without_suff}_{content_type}.json"
    content_data = {
        'total_count': len(content_list),
        'content_type': content_type,
        'data': content_list
    }
    md_writer.write_string(
        file_path,
        json.dumps(content_data, ensure_ascii=False, indent=4)
    )
    if content_type in ['text', 'formulas']:
        print(f"  ✓ {content_type.capitalize()}: {len(content_list)} items saved to {file_path}")


def extract_text_from_block(block):
    """Extract text content from a block."""
    texts = []
    for line in block.get('lines', []):
        for span in line.get('spans', []):
            content = span.get('content', '')
            if content:
                texts.append(content)
    return ' '.join(texts)


def extract_figure_info(block, page_idx):
    """Extract figure information from an image block."""
    bbox = block.get('bbox', [0, 0, 0, 0])
    figure_info = {
        'page': page_idx + 1,
        'type': 'figure',
        'bbox': bbox,
        'caption': '',
        'footnote': '',
        'image_path': ''
    }

    # Extract caption, footnote, and image path from nested blocks
    for sub_block in block.get('blocks', []):
        sub_type = sub_block.get('type', '')
        if sub_type == 'image_caption':
            figure_info['caption'] = extract_text_from_block(sub_block)
        elif sub_type == 'image_footnote':
            figure_info['footnote'] = extract_text_from_block(sub_block)
        elif sub_type == 'image_body':
            # Try to find image path in spans
            for line in sub_block.get('lines', []):
                for span in line.get('spans', []):
                    if span.get('type') == 'image':
                        figure_info['image_path'] = span.get('image_path', '')

    return figure_info


def extract_table_info(block, page_idx):
    """Extract table information from a table block."""
    bbox = block.get('bbox', [0, 0, 0, 0])
    table_info = {
        'page': page_idx + 1,
        'type': 'table',
        'bbox': bbox,
        'caption': '',
        'footnote': '',
        'latex': '',
        'html': ''
    }

    # Extract caption, footnote, and table content from nested blocks
    for sub_block in block.get('blocks', []):
        sub_type = sub_block.get('type', '')
        if sub_type == 'table_caption':
            table_info['caption'] = extract_text_from_block(sub_block)
        elif sub_type == 'table_footnote':
            table_info['footnote'] = extract_text_from_block(sub_block)
        elif sub_type == 'table_body':
            # Try to find latex or html in spans
            for line in sub_block.get('lines', []):
                for span in line.get('spans', []):
                    if span.get('type') == 'table':
                        table_info['latex'] = span.get('latex', '')
                        table_info['html'] = span.get('html', '')

    return table_info


def extract_figure_info_with_image(block, page_idx, figure_number, figures_dir, pdf_doc):
    """Extract figure information and crop image from PDF."""
    bbox = block.get('bbox', [0, 0, 0, 0])
    figure_info = {
        'serial_number': figure_number,
        'page': page_idx + 1,
        'type': 'figure',
        'bbox': bbox,
        'caption': '',
        'footnote': '',
        'image_path': '',
        'cropped_image': ''
    }

    # Extract caption and footnote from nested blocks
    for sub_block in block.get('blocks', []):
        sub_type = sub_block.get('type', '')
        if sub_type == 'image_caption':
            figure_info['caption'] = extract_text_from_block(sub_block)
        elif sub_type == 'image_footnote':
            figure_info['footnote'] = extract_text_from_block(sub_block)

    # Crop image from PDF if available
    if pdf_doc and bbox != [0, 0, 0, 0]:
        try:
            page = pdf_doc[page_idx]
            # Convert bbox to fitz Rect (x0, y0, x1, y1)
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            # Add padding
            padding = 5
            rect = rect + (-padding, -padding, padding, padding)
            rect = rect & page.rect  # Ensure within page bounds
            
            # Extract image at 2x scale for better quality
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
            image_filename = f"figure_{figure_number:03d}_page{figure_info['page']:03d}.png"
            image_path = os.path.join(figures_dir, image_filename)
            pix.save(image_path)
            
            figure_info['cropped_image'] = f"figures/{image_filename}"
            figure_info['image_path'] = f"figures/{image_filename}"
        except Exception as e:
            print(f"Warning: Failed to crop figure {figure_number} on page {page_idx + 1}: {e}")

    return figure_info


def extract_table_info_with_image(block, page_idx, table_number, tables_dir, pdf_doc):
    """Extract table information and crop image from PDF."""
    bbox = block.get('bbox', [0, 0, 0, 0])
    table_info = {
        'serial_number': table_number,
        'page': page_idx + 1,
        'type': 'table',
        'bbox': bbox,
        'caption': '',
        'footnote': '',
        'latex': '',
        'html': '',
        'cropped_image': ''
    }

    # Extract caption and footnote from nested blocks
    for sub_block in block.get('blocks', []):
        sub_type = sub_block.get('type', '')
        if sub_type == 'table_caption':
            table_info['caption'] = extract_text_from_block(sub_block)
        elif sub_type == 'table_footnote':
            table_info['footnote'] = extract_text_from_block(sub_block)
        elif sub_type == 'table_body':
            # Try to find latex or html in spans
            for line in sub_block.get('lines', []):
                for span in line.get('spans', []):
                    if span.get('type') == 'table':
                        table_info['latex'] = span.get('latex', '')
                        table_info['html'] = span.get('html', '')

    # Crop image from PDF if available
    if pdf_doc and bbox != [0, 0, 0, 0]:
        try:
            page = pdf_doc[page_idx]
            # Convert bbox to fitz Rect (x0, y0, x1, y1)
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            # Add padding
            padding = 5
            rect = rect + (-padding, -padding, padding, padding)
            rect = rect & page.rect  # Ensure within page bounds
            
            # Extract image at 2x scale for better quality
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
            image_filename = f"table_{table_number:03d}_page{table_info['page']:03d}.png"
            image_path = os.path.join(tables_dir, image_filename)
            pix.save(image_path)
            
            table_info['cropped_image'] = f"tables/{image_filename}"
        except Exception as e:
            print(f"Warning: Failed to crop table {table_number} on page {page_idx + 1}: {e}")

    return table_info


def extract_latex_from_block(block):
    """Extract LaTeX content from a formula block."""
    for line in block.get('lines', []):
        for span in line.get('spans', []):
            if span.get('type') == 'interline_equation':
                return span.get('latex', '')
    return ''


def extract_from_layout_det(det, page_idx, item_number, item_type, output_dir, pdf_doc):
    """
    Extract information from a layout detection item and crop image from PDF.
    
    Args:
        det: Layout detection item from model output
        page_idx: Page index (0-based)
        item_number: Serial number for this item
        item_type: 'figure' or 'table'
        output_dir: Directory to save cropped image
        pdf_doc: PyMuPDF document object
    
    Returns:
        dict with item information including cropped_image path
    """
    bbox = det.get('bbox', [0, 0, 0, 0])
    
    # Skip if no valid bbox
    if bbox == [0, 0, 0, 0] or len(bbox) != 4:
        return None
    
    info = {
        'serial_number': item_number,
        'page': page_idx + 1,
        'type': item_type,
        'bbox': bbox,
        'cropped_image': ''
    }
    
    # Add type-specific fields
    if item_type == 'table':
        info['caption'] = det.get('caption', '')
        info['footnote'] = det.get('footnote', '')
        info['latex'] = det.get('latex', '')
        info['html'] = det.get('html', '')
    else:  # figure
        info['caption'] = det.get('caption', '')
        info['footnote'] = det.get('footnote', '')
    
    # Crop image from PDF if available
    if pdf_doc:
        try:
            page = pdf_doc[page_idx]
            # Convert bbox to fitz Rect (x0, y0, x1, y1)
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            # Add padding
            padding = 5
            rect = rect + (-padding, -padding, padding, padding)
            rect = rect & page.rect  # Ensure within page bounds
            
            # Extract image at 2x scale for better quality
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
            
            # Determine filename based on type
            if item_type == 'table':
                image_filename = f"table_{item_number:03d}_page{info['page']:03d}.png"
                subfolder = "tables"
            else:
                image_filename = f"figure_{item_number:03d}_page{info['page']:03d}.png"
                subfolder = "figures"
            
            image_path = os.path.join(output_dir, image_filename)
            pix.save(image_path)
            
            info['cropped_image'] = f"{subfolder}/{image_filename}"
        except Exception as e:
            print(f"Warning: Failed to crop {item_type} {item_number} on page {page_idx + 1}: {e}")
    
    return info


def extract_figure_with_caption(fig_data, page_idx, figure_number, figures_dir, pdf_doc, model_list):
    """
    Extract figure with its caption and footnote from MagicModel output.
    
    Args:
        fig_data: Figure data from MagicModel.get_imgs()
        page_idx: Page index (0-based)
        figure_number: Serial number for this figure
        figures_dir: Directory to save cropped image
        pdf_doc: PyMuPDF document object
        model_list: Model output list containing layout_dets with text content
    
    Returns:
        dict with figure information including caption and footnote text
    """
    import os
    
    # Get image body bbox
    image_body = fig_data.get('image_body', {})
    bbox = image_body.get('bbox', [0, 0, 0, 0])
    
    # Skip if no valid bbox
    if bbox == [0, 0, 0, 0] or len(bbox) != 4:
        return None
    
    # Get caption and footnote bboxes
    caption_list = fig_data.get('image_caption_list', [])
    footnote_list = fig_data.get('image_footnote_list', [])
    
    # Extract caption text from model output
    captions = []
    for caption_item in caption_list:
        caption_bbox = caption_item.get('bbox')
        if caption_bbox:
            caption_text = find_text_in_bbox(caption_bbox, page_idx, model_list)
            if caption_text:
                captions.append(caption_text)
    
    # Extract footnote text from model output
    footnotes = []
    for footnote_item in footnote_list:
        footnote_bbox = footnote_item.get('bbox')
        if footnote_bbox:
            footnote_text = find_text_in_bbox(footnote_bbox, page_idx, model_list)
            if footnote_text:
                footnotes.append(footnote_text)
    
    info = {
        'serial_number': figure_number,
        'page': page_idx + 1,
        'type': 'figure',
        'bbox': bbox,
        'caption': ' '.join(captions),
        'footnote': ' '.join(footnotes),
        'cropped_image': ''
    }
    
    # Crop image from PDF if available
    if pdf_doc:
        try:
            page = pdf_doc[page_idx]
            # Convert bbox to fitz Rect (x0, y0, x1, y1)
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            # Add padding
            padding = 5
            rect = rect + (-padding, -padding, padding, padding)
            rect = rect & page.rect  # Ensure within page bounds
            
            # Extract image at 2x scale for better quality
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
            
            image_filename = f"figure_{figure_number:03d}_page{info['page']:03d}.png"
            image_path = os.path.join(figures_dir, image_filename)
            pix.save(image_path)
            
            info['cropped_image'] = f"figures/{image_filename}"
        except Exception as e:
            print(f"Warning: Failed to crop figure {figure_number} on page {page_idx + 1}: {e}")
    
    return info


def extract_table_with_caption(table_data, page_idx, table_number, tables_dir, pdf_doc, model_list):
    """
    Extract table with its caption and footnote from MagicModel output.
    
    Args:
        table_data: Table data from MagicModel.get_tables()
        page_idx: Page index (0-based)
        table_number: Serial number for this table
        tables_dir: Directory to save cropped image
        pdf_doc: PyMuPDF document object
        model_list: Model output list containing layout_dets with text content
    
    Returns:
        dict with table information including caption, footnote, latex, and html
    """
    import os
    
    # Get table body bbox
    table_body = table_data.get('table_body', {})
    bbox = table_body.get('bbox', [0, 0, 0, 0])
    
    # Skip if no valid bbox
    if bbox == [0, 0, 0, 0] or len(bbox) != 4:
        return None
    
    # Get caption and footnote bboxes
    caption_list = table_data.get('table_caption_list', [])
    footnote_list = table_data.get('table_footnote_list', [])
    
    # Extract caption text from model output
    captions = []
    for caption_item in caption_list:
        caption_bbox = caption_item.get('bbox')
        if caption_bbox:
            caption_text = find_text_in_bbox(caption_bbox, page_idx, model_list)
            if caption_text:
                captions.append(caption_text)
    
    # Extract footnote text from model output
    footnotes = []
    for footnote_item in footnote_list:
        footnote_bbox = footnote_item.get('bbox')
        if footnote_bbox:
            footnote_text = find_text_in_bbox(footnote_bbox, page_idx, model_list)
            if footnote_text:
                footnotes.append(footnote_text)
    
    # Get latex and html from table body in model output
    latex = ''
    html = ''
    page_data = model_list[page_idx] if page_idx < len(model_list) else {}
    layout_dets = page_data.get('layout_dets', []) if isinstance(page_data, dict) else []
    for det in layout_dets:
        det_bbox = det.get('bbox', [0, 0, 0, 0])
        # Match bbox with tolerance for all four coordinates
        tolerance = 2.0
        if (abs(det_bbox[0] - bbox[0]) < tolerance and
            abs(det_bbox[1] - bbox[1]) < tolerance and
            abs(det_bbox[2] - bbox[2]) < tolerance and
            abs(det_bbox[3] - bbox[3]) < tolerance):
            latex = det.get('latex', '')
            html = det.get('html', '')
            break
    
    info = {
        'serial_number': table_number,
        'page': page_idx + 1,
        'type': 'table',
        'bbox': bbox,
        'caption': ' '.join(captions),
        'footnote': ' '.join(footnotes),
        'latex': latex,
        'html': html,
        'cropped_image': ''
    }
    
    # Crop image from PDF if available
    if pdf_doc:
        try:
            page = pdf_doc[page_idx]
            # Convert bbox to fitz Rect (x0, y0, x1, y1)
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            # Add padding
            padding = 5
            rect = rect + (-padding, -padding, padding, padding)
            rect = rect & page.rect  # Ensure within page bounds
            
            # Extract image at 2x scale for better quality
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
            
            image_filename = f"table_{table_number:03d}_page{info['page']:03d}.png"
            image_path = os.path.join(tables_dir, image_filename)
            pix.save(image_path)
            
            info['cropped_image'] = f"tables/{image_filename}"
        except Exception as e:
            print(f"Warning: Failed to crop table {table_number} on page {page_idx + 1}: {e}")
    
    return info


def find_text_in_bbox(target_bbox, page_idx, model_list):
    """
    Find text content within a given bbox from model output.
    
    Args:
        target_bbox: The bbox to search within [x0, y0, x1, y1]
        page_idx: Page index
        model_list: Model output list containing layout_dets
    
    Returns:
        str: Concatenated text content found within the bbox
    """
    from magic_pdf.config.ocr_content_type import CategoryId
    
    if page_idx >= len(model_list):
        return ''
    
    page_data = model_list[page_idx]
    if not isinstance(page_data, dict) or 'layout_dets' not in page_data:
        return ''
    
    texts = []
    layout_dets = page_data['layout_dets']
    
    for det in layout_dets:
        det_bbox = det.get('bbox', [0, 0, 0, 0])
        category_id = det.get('category_id', -1)
        
        # Check if this bbox matches target bbox
        # Use a tolerance for comparison
        tolerance = 2.0
        if (abs(det_bbox[0] - target_bbox[0]) < tolerance and
            abs(det_bbox[1] - target_bbox[1]) < tolerance and
            abs(det_bbox[2] - target_bbox[2]) < tolerance and
            abs(det_bbox[3] - target_bbox[3]) < tolerance):
            # Found matching bbox, extract text based on category
            if category_id in [CategoryId.Title, CategoryId.Text, CategoryId.OcrText]:
                text = det.get('text', '')
                if text:
                    texts.append(text)
    
    return ' '.join(texts)


def save_text_summary(md_writer, name_without_suff, text_content, figure_content, table_content, formula_content):
    """Save a human-readable summary text file."""
    summary_lines = [
        "=" * 60,
        "PDF Content Extraction Summary",
        "=" * 60,
        "",
        f"Text Blocks: {len(text_content)}",
        f"Figures: {len(figure_content)}",
        f"Tables: {len(table_content)}",
        f"Formulas: {len(formula_content)}",
        "",
        "-" * 60,
        "TEXT CONTENT",
        "-" * 60,
        ""
    ]

    for item in text_content:
        summary_lines.append(f"[Page {item['page']}] ({item['type']})")
        summary_lines.append(item['content'])
        summary_lines.append("")

    summary_lines.extend([
        "",
        "-" * 60,
        "FIGURES",
        "-" * 60,
        ""
    ])

    for item in figure_content:
        summary_lines.append(f"[Page {item['page']}] Figure")
        if item.get('caption'):
            summary_lines.append(f"Caption: {item['caption']}")
        if item.get('cropped_image'):
            summary_lines.append(f"Image: {item['cropped_image']}")
        summary_lines.append("")

    summary_lines.extend([
        "",
        "-" * 60,
        "TABLES",
        "-" * 60,
        ""
    ])

    for item in table_content:
        summary_lines.append(f"[Page {item['page']}] Table")
        if item['caption']:
            summary_lines.append(f"Caption: {item['caption']}")
        if item['latex']:
            summary_lines.append(f"LaTeX: {item['latex']}")
        summary_lines.append("")

    summary_lines.extend([
        "",
        "-" * 60,
        "FORMULAS",
        "-" * 60,
        ""
    ])

    for item in formula_content:
        summary_lines.append(f"[Page {item['page']}] Formula")
        summary_lines.append(item['content'])
        if item['latex']:
            summary_lines.append(f"LaTeX: {item['latex']}")
        summary_lines.append("")

    summary_text = '\n'.join(summary_lines)
    md_writer.write_string(f"{name_without_suff}_summary.txt", summary_text)
    print(f"  ✓ Summary saved to {name_without_suff}_summary.txt")
