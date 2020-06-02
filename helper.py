import os

def get_file_path(folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    destination = os.path.join(folder, file_name)
    return destination