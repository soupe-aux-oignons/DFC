import os
from PIL import Image
import torch
import clip
import faiss
import numpy as np
import cv2



#charge le modèle CLIP de openAI
device = "cuda" if torch.cuda.is_available() else "cpu" #par défaut sera CPU
model, preprocess = clip.load("Vit-B/32", device=device) #renvoie le modele, stocke dans model /
#renvoie le transforme necessaire au TorchVision

# 2. Fonction pour extraire les features d'une image
def get_image_features(image_path):
    """prend en argument un chemin vers une image, renvoie les features de l'image"""
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features[0].cpu().numpy()

# 3. Indexer toutes les images dans un dossier
image_folder = os.path.abspath(os.sys.argv[1])
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

features_list = []
for path in image_paths:
    try:
        features_list.append(get_image_features(path))
    except:
        print(f"Erreur avec : {path}")
features_array = np.stack(features_list)

# 4. Créer un index FAISS
index = faiss.IndexFlatL2(features_array.shape[1])
index.add(features_array)

# 5. Comparer avec une image de référence
query_image_path = os.sys.argv[2]
query_features = get_image_features(query_image_path).reshape(1, -1)

# 6. Recherche des k plus proches
k = 5
D, I = index.search(query_features, k)

# 7. Afficher les résultats
print("Images les plus similaires :")
for i in I[0]:
    print(image_paths[i])
    cv2.imshow("Image match", cv2.imread(image_paths))

cv2.destroyAllWindows()