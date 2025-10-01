import base64
import requests
import os
from mistralai import Mistral
import json
from parse import compile as parse_compile
from werkzeug.utils import secure_filename
from pathlib import Path

session_output_dir = Path("/var/www/RAG/Data_parse/test/")
def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

# Path to your pdf
pdf_path = "/var/www/RAG/Data/CWD FR/SELLES/CODIFICATIONS ET FICHES PRODUITS/SE01_SELLE OPTIM PLATE_fr.pdf"
pdf_path = Path(pdf_path)

pdf_base = pdf_path.stem
pdf_base_sanitized = secure_filename(pdf_base) # Use sanitized version for directory/file names
print(f"Processing {pdf_path.name}...")

pdf_output_dir = session_output_dir / pdf_base_sanitized
pdf_output_dir.mkdir(exist_ok=True)
images_dir = pdf_output_dir / "images"
images_dir.mkdir(exist_ok=True)

# Getting the base64 string
base64_pdf = encode_pdf(pdf_path)

api_key = "EWyA4Oi0d4JSirCxt5IlDQYrbHun8YID"
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    # include_image_base64=True,
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}" 
    }
)

# Optional: Save Raw OCR Response
ocr_json_path = pdf_output_dir / "ocr_response.json"
try:
    with open(ocr_json_path, "w", encoding="utf-8") as json_file:
        if hasattr(ocr_response, 'model_dump'):
            json.dump(ocr_response.model_dump(), json_file, indent=4, ensure_ascii=False)
        else:
                json.dump(ocr_response.dict(), json_file, indent=4, ensure_ascii=False)
    print(f"  Raw OCR response saved to {ocr_json_path}")
except Exception as json_err:
    print(f"  Warning: Could not save raw OCR JSON: {json_err}")

def replace_images_in_markdown_with_wikilinks(markdown_str: str, image_mapping: dict) -> str:
    updated_markdown = markdown_str
    for original_id, new_name in image_mapping.items():
        updated_markdown = updated_markdown.replace(
            f"![{original_id}]({original_id})",
            f"![[{new_name}]]"
        )
    return updated_markdown



# Process OCR Response -> Markdown & Images
global_image_counter = 1
updated_markdown_pages = []
extracted_image_filenames = [] # Store filenames for preview

print(f"  Extracting images and generating Markdown...")
for page_index, page in enumerate(ocr_response.pages):
    current_page_markdown = page.markdown
    page_image_mapping = {}

    for image_obj in page.images:
        base64_str = image_obj.image_base64
        if not base64_str: continue # Skip if no image data

        if base64_str.startswith("data:"):
                try: base64_str = base64_str.split(",", 1)[1]
                except IndexError: continue

        try: image_bytes = base64.b64decode(base64_str)
        except Exception as decode_err:
            print(f"  Warning: Base64 decode error for image {image_obj.id} on page {page_index+1}: {decode_err}")
            continue

        original_ext = Path(image_obj.id).suffix
        ext = original_ext if original_ext else ".png"
        new_image_name = f"{pdf_base_sanitized}_p{page_index+1}_img{global_image_counter}{ext}"
        global_image_counter += 1

        image_output_path = images_dir / new_image_name
        try:
            with open(image_output_path, "wb") as img_file:
                img_file.write(image_bytes)
            extracted_image_filenames.append(new_image_name) # Add to list for preview
            page_image_mapping[image_obj.id] = new_image_name
        except IOError as io_err:
                print(f"  Warning: Could not write image file {image_output_path}: {io_err}")
                continue

    updated_page_markdown = replace_images_in_markdown_with_wikilinks(current_page_markdown, page_image_mapping)
    updated_markdown_pages.append(updated_page_markdown)

final_markdown_content = "\n\n---\n\n".join(updated_markdown_pages) # Page separator
output_markdown_path = pdf_output_dir / f"{pdf_base_sanitized}_output.md"

try:
    with open(output_markdown_path, "w", encoding="utf-8") as md_file:
        md_file.write(final_markdown_content)
    print(f"  Markdown generated successfully at {output_markdown_path}")
except IOError as io_err:
    raise Exception(f"Failed to write final markdown file: {io_err}") from io_err

# print(ocr_response)
# with open("/var/www/RAG/Data/test/output.txt", "w") as f:
#     f.write(ocr_response["text"])

