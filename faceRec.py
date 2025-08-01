import face_recognition
import os

def load_picture(file_path):
    img=face_recognition.load_image_file(os.path.abspath(file_path))
    return img

def load_directory(directory_path):
    imgs=[]
    for file in os.listdir(os.path.abspath(directory_path)):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".PNG") or file.endswith(".JPG"):
            imgs.append(face_recognition.load_image_file(os.path.abspath(file)))

    return imgs

def load (path):
    if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".PNG") or path.endswith(".JPG"):
        load_picture(path)
    else:
        load_directory(path)

    return

