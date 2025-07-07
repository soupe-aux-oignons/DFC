from mtcnn import MTCNN
from mtcnn.utils.images import load_image, load_images_batch
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt
import os

directory = os.fsencode(os.path.abspath(os.sys.argv[1]))
img_paths = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_paths.append(os.path.abspath(os.sys.argv[1]+"/"+filename))


detector = MTCNN(device="CPU:0")

imgs = load_images_batch(img_paths)

results = detector.detect_faces(imgs, batch_stack_justification="center")

for i, img in enumerate(imgs):
    plt.figure()
    plt.imshow(plot(img, results[i]))
    plt.title(f"Results for image{i+1}")
    plt.show()

