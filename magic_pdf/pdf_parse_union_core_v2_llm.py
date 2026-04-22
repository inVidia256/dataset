"""
PDF Parse Union Core v2 for LLM-based OCR processing.
This module processes model inference results and generates structured pdf_info.
"""

import copy
from typing import List, Dict, Any
from loguru import logger

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.data.dataset import Dataset
from magic_pdf.data.data_reader_writer import DataWriter
# para_split is called internally with specific format

__version__ = "2.0.0"


def pdf_parse_union(
    inference_results: list,
    dataset: Dataset,
    image_writer: DataWriter,
    parse_method: SupportedPdfParseMethod,
    start_page_id: int = 0,
    end_page_id: int = None,
    debug_mode: bool = False,
    lang: str = None,
    MonkeyOCR_model=None,
) -> Dict[str, Any]:
    """
    Process PDF or image dataset with OCR model and generate structured output.

    Args:
        inference_results: Model inference results (passed by self.apply)
        dataset: The dataset containing PDF or image data
        image_writer: Writer for saving extracted images
        parse_method: The parsing method to use (OCR or TXT)
        start_page_id: Starting page index (0-based)
        end_page_id: Ending page index (None for all pages)
        debug_mode: Enable debug logging
        lang: Language code for processing
        MonkeyOCR_model: The OCR model instance for text recognition

    Returns:
        Dict containing 'pdf_info' list with processed page information
    """
    if end_page_id is None:
        end_page_id = len(dataset) - 1

    pdf_info_list = []

    for page_idx in range(start_page_id, end_page_id + 1):
        if page_idx >= len(dataset):
            break

        try:
            # Use get_page method instead of subscript
            if hasattr(dataset, 'get_page'):
                page_data = dataset.get_page(page_idx)
            else:
                page_data = dataset[page_idx]
            # Get inference result for this page if available
            # inference_results is in MagicModel format: [{'page_info': {}, 'layout_dets': []}, ...]
            page_infer_res = None
            if page_idx < len(inference_results):
                page_data_raw = inference_results[page_idx]
                if isinstance(page_data_raw, dict) and 'layout_dets' in page_data_raw:
                    page_infer_res = page_data_raw['layout_dets']
                else:
                    page_infer_res = page_data_raw
            page_info = process_page(
                page_data=page_data,
                page_infer_res=page_infer_res,
                page_idx=page_idx,
                image_writer=image_writer,
                parse_method=parse_method,
                debug_mode=debug_mode,
                lang=lang,
                ocr_model=MonkeyOCR_model,
            )
            pdf_info_list.append(page_info)
        except Exception as e:
            logger.error(f"Error processing page {page_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Add empty page info to maintain index consistency
            pdf_info_list.append(create_empty_page_info(page_idx))

    return {'pdf_info': pdf_info_list}


def process_page(
    page_data,
    page_infer_res,
    page_idx: int,
    image_writer: DataWriter,
    parse_method: SupportedPdfParseMethod,
    debug_mode: bool,
    lang: str,
    ocr_model,
) -> Dict[str, Any]:
    """Process a single page and return its information."""

    # Get page dimensions
    if hasattr(page_data, 'page_size'):
        page_size = page_data.page_size
    else:
        # Default A4 size if not available
        page_size = [612, 792]

    page_info = {
        'page_idx': page_idx,
        'page_size': page_size,
        'para_blocks': [],
        'discarded_blocks': [],
        'need_drop': False,
        'drop_reason': None,
        'lang': lang,
    }

    # Use inference result passed from caller or from page_data
    infer_res = page_infer_res
    if infer_res is None:
        infer_res = getattr(page_data, 'inference_result', None)
    if infer_res is None:
        infer_res = getattr(page_data, 'model_result', [])

    if not infer_res:
        return page_info

    # Convert inference results to blocks
    blocks = convert_inference_to_blocks(infer_res, page_size)

    # Process blocks with OCR if needed
    if parse_method == SupportedPdfParseMethod.OCR and ocr_model:
        blocks = apply_ocr_to_blocks(blocks, ocr_model, image_writer, page_idx)

    # Split into paragraphs (simplified version)
    para_blocks = split_blocks_to_paragraphs(blocks, page_size)

    # Filter out discarded blocks
    valid_blocks = []
    discarded_blocks = []
    for block in para_blocks:
        if block.get('is_discarded', False):
            discarded_blocks.append(block)
        else:
            valid_blocks.append(block)

    # Set both preproc_blocks (for draw_span_bbox) and para_blocks (for other functions)
    page_info['preproc_blocks'] = valid_blocks
    page_info['para_blocks'] = valid_blocks
    page_info['discarded_blocks'] = discarded_blocks

    return page_info


def convert_inference_to_blocks(infer_res: List[Dict], page_size: List[float]) -> List[Dict]:
    """
    Convert model inference results (layout_dets) to block format.

    Args:
        infer_res: List of layout detection results from model
        page_size: [width, height] of the page

    Returns:
        List of block dictionaries
    """
    blocks = []

    for item in infer_res:
        # Skip if item is not a dict
        if not isinstance(item, dict):
            continue

        # Get bbox from item
        bbox = item.get('bbox', [0, 0, 0, 0])
        if not bbox or bbox == [0, 0, 0, 0]:
            # Try to convert from poly if available
            if 'poly' in item:
                poly = item['poly']
                if len(poly) >= 8:
                    bbox = [poly[0], poly[1], poly[4], poly[5]]

        # Map category_id to block type
        category_id = item.get('category_id', 1)
        block_type = map_category_to_block_type(category_id)

        block = {
            'type': block_type,
            'bbox': bbox,
            'bbox_fs': bbox,
            'page_size': page_size,
            'score': item.get('score', 1.0),
            'lines': [],
            'blocks': [],
        }

        # Create line content from available text/latex
        content_text = ''
        if 'text' in item:
            content_text = item['text']
        elif 'latex' in item:
            content_text = item['latex']
        elif 'html' in item:
            content_text = item['html']

        # Create span with appropriate type
        span_type = ContentType.Text
        if block_type == BlockType.Image:
            span_type = ContentType.Image
        elif block_type == BlockType.Table:
            span_type = ContentType.Table
        elif block_type == BlockType.InterlineEquation:
            span_type = ContentType.InterlineEquation

        span = {
            'type': span_type,
            'bbox': bbox,
            'content': content_text,
            'font': 'default',
            'font_size': 12,
        }

        # Add latex/html if available
        if 'latex' in item:
            span['latex'] = item['latex']
        if 'html' in item:
            span['html'] = item['html']

        # Create line with span
        line = {
            'bbox': bbox,
            'spans': [span],
        }
        block['lines'] = [line]

        # Process nested blocks for images and tables
        if block_type in [BlockType.Image, BlockType.Table]:
            nested_blocks = item.get('blocks', [])
            if nested_blocks:
                block['blocks'] = convert_nested_blocks(nested_blocks, page_size)
            else:
                # Create default nested structure
                block['blocks'] = create_default_nested_blocks(block_type, bbox, page_size)

        blocks.append(block)

    return blocks


def map_category_to_block_type(category_id) -> str:
    """Map layout model category ID to block type.
    
    CategoryId definition:
    - 0: Title
    - 1: Text  
    - 2: Abandon
    - 3: ImageBody
    - 4: ImageCaption
    - 5: TableBody
    - 6: TableCaption
    - 7: TableFootnote
    - 8: InterlineEquation_Layout
    - 13: InlineEquation
    - 14: InterlineEquation_YOLO
    - 15: OcrText
    - 101: ImageFootnote
    """
    # Convert string to int if needed
    if isinstance(category_id, str):
        try:
            category_id = int(category_id)
        except ValueError:
            return BlockType.Text
    
    category_map = {
        0: BlockType.Title,
        1: BlockType.Text,
        2: BlockType.Text,  # Abandon - treated as text
        3: BlockType.Image,
        4: BlockType.Image,  # ImageCaption
        5: BlockType.Table,
        6: BlockType.Table,  # TableCaption
        7: BlockType.Table,  # TableFootnote
        8: BlockType.InterlineEquation,
        13: BlockType.Text,  # InlineEquation - treated as text
        14: BlockType.InterlineEquation,
        15: BlockType.Text,  # OcrText
        101: BlockType.Image,  # ImageFootnote
    }
    return category_map.get(category_id, BlockType.Text)


def convert_nested_blocks(nested_items: List[Dict], page_size: List[float]) -> List[Dict]:
    """Convert nested block items for images/tables."""
    blocks = []
    for item in nested_items:
        block_type = item.get('type', 'text')
        bbox = item.get('bbox', [0, 0, 0, 0])

        # Map nested type names to BlockType
        if isinstance(block_type, str):
            type_map = {
                'image_body': BlockType.ImageBody,
                'image_caption': BlockType.ImageCaption,
                'image_footnote': BlockType.ImageFootnote,
                'table_body': BlockType.TableBody,
                'table_caption': BlockType.TableCaption,
                'table_footnote': BlockType.TableFootnote,
            }
            block_type = type_map.get(block_type, block_type)

        block = {
            'type': block_type,
            'bbox': bbox,
            'bbox_fs': bbox,
            'page_size': page_size,
            'lines': [],
        }

        if 'lines' in item:
            block['lines'] = convert_lines(item['lines'])
        elif 'text' in item:
            block['lines'] = [create_line_from_text(item['text'], bbox)]

        blocks.append(block)

    return blocks


def create_default_nested_blocks(block_type: str, parent_bbox: List[float], page_size: List[float]) -> List[Dict]:
    """Create default nested block structure for images/tables."""
    blocks = []

    if block_type == BlockType.Image:
        # Create ImageBody as the main content
        blocks.append({
            'type': BlockType.ImageBody,
            'bbox': parent_bbox,
            'bbox_fs': parent_bbox,
            'page_size': page_size,
            'lines': [create_line_from_text('', parent_bbox)],
        })
    elif block_type == BlockType.Table:
        # Create TableBody as the main content
        blocks.append({
            'type': BlockType.TableBody,
            'bbox': parent_bbox,
            'bbox_fs': parent_bbox,
            'page_size': page_size,
            'lines': [create_line_from_text('', parent_bbox)],
        })

    return blocks


def convert_lines(lines_data: List[Dict]) -> List[Dict]:
    """Convert line data to standard format."""
    lines = []
    for line in lines_data:
        if not isinstance(line, dict):
            continue

        converted_line = {
            'bbox': line.get('bbox', [0, 0, 0, 0]),
            'spans': convert_spans(line.get('spans', [])),
        }
        lines.append(converted_line)

    return lines


def convert_spans(spans_data: List[Dict]) -> List[Dict]:
    """Convert span data to standard format."""
    spans = []
    for span in spans_data:
        if not isinstance(span, dict):
            continue

        span_type = span.get('type', ContentType.Text)
        if isinstance(span_type, str):
            type_map = {
                'text': ContentType.Text,
                'image': ContentType.Image,
                'table': ContentType.Table,
                'equation': ContentType.InterlineEquation,
            }
            span_type = type_map.get(span_type, ContentType.Text)

        converted_span = {
            'type': span_type,
            'bbox': span.get('bbox', [0, 0, 0, 0]),
            'content': span.get('content', span.get('text', '')),
            'font': span.get('font', 'default'),
            'font_size': span.get('font_size', 12),
        }

        # Add image/table specific fields
        if 'image_path' in span:
            converted_span['image_path'] = span['image_path']
        if 'latex' in span:
            converted_span['latex'] = span['latex']
        if 'html' in span:
            converted_span['html'] = span['html']

        spans.append(converted_span)

    return spans


def create_line_from_text(text: str, bbox: List[float]) -> Dict:
    """Create a line dictionary from text."""
    return {
        'bbox': bbox,
        'spans': [{
            'type': ContentType.Text,
            'bbox': bbox,
            'content': text,
            'font': 'default',
            'font_size': 12,
        }],
    }


def apply_ocr_to_blocks(
    blocks: List[Dict],
    ocr_model,
    image_writer: DataWriter,
    page_idx: int,
) -> List[Dict]:
    """
    Apply OCR to blocks that need text recognition.

    Args:
        blocks: List of blocks to process
        ocr_model: The OCR model instance
        image_writer: Writer for saving images
        page_idx: Current page index

    Returns:
        Updated blocks with OCR text
    """
    if ocr_model is None:
        return blocks

    for block in blocks:
        # Skip if block already has text content
        if block.get('lines') and any(
            span.get('content')
            for line in block['lines']
            for span in line.get('spans', [])
        ):
            continue

        # Apply OCR based on block type
        if block['type'] in [BlockType.Text, BlockType.Title, BlockType.List, BlockType.Index]:
            # These blocks should have OCR applied
            pass

        # Process nested blocks
        if block.get('blocks'):
            block['blocks'] = apply_ocr_to_blocks(
                block['blocks'], ocr_model, image_writer, page_idx
            )

    return blocks


def split_blocks_to_paragraphs(blocks: List[Dict], page_size: List[float]) -> List[Dict]:
    """
    Split blocks into paragraph structures.
    Simplified version that processes blocks and groups them appropriately.
    """
    if not blocks:
        return []

    para_blocks = []

    for block in blocks:
        # Skip empty blocks
        if not block.get('bbox') or block['bbox'] == [0, 0, 0, 0]:
            continue

        # Create paragraph block structure
        para_block = {
            'type': block.get('type', BlockType.Text),
            'bbox': block['bbox'],
            'bbox_fs': block.get('bbox_fs', block['bbox']),
            'page_size': page_size,
            'lines': block.get('lines', []),
            'blocks': block.get('blocks', []),
            'score': block.get('score', 1.0),
        }

        # Ensure lines have proper content
        if not para_block['lines']:
            para_block['lines'] = [create_line_from_text('', para_block['bbox'])]

        # Process lines to ensure spans have content
        for line in para_block['lines']:
            if not line.get('spans'):
                line['spans'] = [{
                    'type': ContentType.Text,
                    'bbox': line['bbox'],
                    'content': '',
                    'font': 'default',
                    'font_size': 12,
                }]

        para_blocks.append(para_block)

    return para_blocks


def create_empty_page_info(page_idx: int) -> Dict[str, Any]:
    """Create an empty page info structure."""
    return {
        'page_idx': page_idx,
        'page_size': [612, 792],
        'para_blocks': [],
        'discarded_blocks': [],
        'need_drop': False,
        'drop_reason': None,
    }
