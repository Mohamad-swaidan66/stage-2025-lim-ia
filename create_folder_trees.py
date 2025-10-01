from pathlib import Path
# from video_transcription import structured_transcription_pipeline
# from smol_docling import convert_pdf, convert_office_document
from smol_docling import structured_pdf_pipeline  
from tqdm import tqdm


# create folder trees
input_folder = Path('/var/www/RAG/Data/')
output_folder = Path('/var/www/RAG/Data_parse/')


output_folder.mkdir(parents=True, exist_ok=True)
print(input_folder, input_folder.exists())

# ------------------------
# Get all subdirectories recursively
def create_folder_trees(input_folder, output_folder):
    subfolders = list(input_folder.rglob('*/'))
    print("subfolders", subfolders)
    for folder in subfolders:
        print(folder)
        # Create a new folder in the output folder
        new_folder = output_folder / folder.relative_to(input_folder)
        if new_folder.exists():
            continue
        new_folder.mkdir(parents=True, exist_ok=True)
        print(folder, '->', new_folder)

# ------------------------
# create folder trees
create_folder_trees(input_folder, output_folder)

# Get all subdirectories recursively
files = list(input_folder.rglob('*'))

# remove temporary files and folders
files = [f for f in files if (not f.name.startswith('.') and not f.is_dir())]
files = [f for f in files if (not f.name.startswith('~$'))]

# ------------------------
# get video files from files
# video_extension = ['.mp4', '.avi', '.mkv']
# video_files = [f for f in files if f.suffix in video_extension]

# # remove video files from files list
# files = [f for f in files if f not in video_files]

# print(video_files)

# # transcript video files
# for f_ in video_files[:]:
#     print(f_)
#     output_file = output_folder / f_.relative_to(input_folder)
#     output_file = output_file.with_suffix(".md")
#     if output_file.exists():
#         continue

#     print(output_file)

#     structured_transcription_pipeline(
#         video_path=f_,
#         output_text_path=str(output_file),
#         model_size="large",
#         language="fr",
#         chunk_duration=60,
#         model_name="llama3.3:latest"
#     )
#     break

# ------------------------
# get office files from files
# office_extension = ['.docx', '.pptx']
# office_files = [f for f in files if f.suffix in office_extension]

# # remove office files from files list
# files = [f for f in files if f not in office_files]

# print(office_files)

# # transcript office files
# for f_ in tqdm(office_files[:]):
#     print(f_)
#     output_file = output_folder / f_.relative_to(input_folder)
#     output_file = output_file.with_suffix(".md")

#     if output_file.exists():
#         continue

#     print(output_file)

#     convert_office_document(f_, output_file)

# ------------------------
# get pdf files from files
pdf_extension = ['.pdf']
pdf_files = [f for f in files if f.suffix in pdf_extension]

# remove pdf files from files list
files = [f for f in files if f not in pdf_files]

print(pdf_files)

# transcript pdf files
for f_ in tqdm(pdf_files[:]):
    print(f_)
    output_file = output_folder / f_.relative_to(input_folder)
    output_file = output_file.with_suffix(".md")
    if output_file.exists():
        continue

    print(output_file)

    try:
        # ✅ Utilise le pipeline structuré (OCR + enrichissement Markdown)
        structured_pdf_pipeline(f_, output_file, model_name="magistral:24b")
        print(f"✅ Fichier traité avec succès : {f_}")

    except Exception as e:
        print(f"❌ Erreur sur {f_} : {e}")

print("Remaining files", files)
