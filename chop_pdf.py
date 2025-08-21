import os
import subprocess

def chop_pdf_files_in_directory(directory):
    save_dir = "chop_pdf"
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            save_dir_tree = directory.replace("visualize/figures", save_dir)
            os.makedirs(save_dir_tree, exist_ok=True)
            filepath = os.path.join(directory, filename)
            save_filepath = os.path.join(save_dir_tree, filename)
            subprocess.run(["pdf-crop-margins", "-p", "0", "-o", save_filepath, filepath])

if __name__ == "__main__":
    root = "visualize/figures"
    directorys =[
        root,
        *[os.path.join(root, x) for x in os.listdir(root)]
    ]
    
    directorys = list(filter(lambda x: os.path.isdir(x), directorys))
    for directory in directorys:
        chop_pdf_files_in_directory(directory)
