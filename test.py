from mtcnn import MTCNN
from mtcnn.utils.images import load_image, load_images_batch
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt
import os
import json

directory = os.fsencode(os.path.abspath(os.sys.argv[1]))
print(directory)
img_paths = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".JPG"):
        print(os.path.abspath(os.sys.argv[1]+"/"+filename))
        img_paths.append(os.path.abspath(os.sys.argv[1]+"/"+filename))

print(img_paths)

detector = MTCNN()

for img_path in img_paths :
    img = load_image(img_path)
    result = detector.detect_faces(img)
    plt.figure()
    plt.imshow(plot(img, result))
    plt.title("Results for image")
    plt.show()

    
