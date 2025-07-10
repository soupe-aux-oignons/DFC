from mtcnn import MTCNN
from mtcnn.utils.images import load_image, load_images_batch
from mtcnn.utils.plotting import plot
import os
import cv2

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
i=0
for img_path in img_paths :
    img_analyse = load_image(img_path)
    img_save = cv2.imread(img_path)
    results = detector.detect_faces(img_analyse)
    for result in results :
        for item in result.items():
            if item[0] == "box":
                x,y,w,h = item[1]
        face = img_save[y:y+h,x:x+w]
        cv2.imwrite(f"{os.path.abspath(os.sys.argv[2])}/face.{i}.jpg",face)
        i+=1
    
