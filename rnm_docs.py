import os

# Directories
folder_path_chatbot = "Chatbot/Documentation/"
folder_path_facerec = "FaceRec/Dokumentasi/"

# Define the file type extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
document_extensions = ('.pdf', '.docx', '.doc', '.txt', '.pptx', '.xlsx')

# Initialize counters for each file type
counters = {'image': 1, 'video': 1, 'document': 1}

# Function to get new file name
def get_new_file_name(file_type, extension):
    new_name = f"{file_type}_{counters[file_type]:03}{extension}"
    counters[file_type] += 1
    return new_name


print("Opening Chatbot Documentation\n")
# Iterate over files in the directory
for filename in os.listdir(folder_path_chatbot):
    file_path = os.path.join(folder_path_chatbot, filename)
    if os.path.isfile(file_path):
        _, ext = os.path.splitext(filename.lower())
        
        # Determine the file type and rename accordingly
        if ext in image_extensions:
            new_name = get_new_file_name('image', ext)
        elif ext in video_extensions:
            new_name = get_new_file_name('video', ext)
        elif ext in document_extensions:
            new_name = get_new_file_name('document', ext)
        else:
            continue
        
        new_file_path = os.path.join(folder_path_chatbot, new_name)
        os.rename(file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_name}")

print("Renaming Chatbot docs completed.\n")

print("Opening FaceRec Documentation\n")
# Iterate over files in the directory
for filename in os.listdir(folder_path_facerec):
    file_path = os.path.join(folder_path_facerec, filename)
    if os.path.isfile(file_path):
        _, ext = os.path.splitext(filename.lower())
        
        # Determine the file type and rename accordingly
        if ext in image_extensions:
            new_name = get_new_file_name('image', ext)
        elif ext in video_extensions:
            new_name = get_new_file_name('video', ext)
        elif ext in document_extensions:
            new_name = get_new_file_name('document', ext)
        else:
            continue
        
        new_file_path = os.path.join(folder_path_facerec, new_name)
        os.rename(file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_name}")

print("Renaming FaceRec docs completed.\n")