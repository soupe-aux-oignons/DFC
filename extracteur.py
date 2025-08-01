from mtcnn import MTCNN
from mtcnn.utils.images import load_image, load_images_batch
from mtcnn.utils.plotting import plot
import os
import cv2
import time



#-----Charger des fichiers--------
def isCorrectFile(filepath):
    """prend en argument un chemin dans l'arborescence
    renvoie un booléen verifiant si un fichier est un jpg ou png"""
    path=os.fsdecode(filepath)
    return (path.endswith(".jpg") or path.endswith(".png") or path.endswith(".JPG") or path.endswith(".PNG")) and (not path.startswith('.'))

def load_file(path):
    """prend en argument un chemin dans l'arborescence
    renvoie une liste contenant le chemin absolu du fichier"""
    return [os.path.abspath(path)]

def load_directory(path):
    """prend en argument un chemin dans l'arborescence
    renvoie une liste contenant le chemin absolu de chaque fichier valide dans le dossier étudié"""
    path_array = []
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if isCorrectFile(filename):
            path_array.append(os.path.abspath(os.fsdecode(path)+"/"+filename))
    return path_array

def load(path):
    """prend en argument un chemin dans l'arborescence
    appelle une des deux fonctions de chargement, renvoie une liste de chemin absolu"""
    if os.path.isfile(path):
        return load_file(path)
    elif os.path.isdir(path):
        return load_directory(path)
    else:
        print("Error : path does not lead to a valid file or directory\n")
        return



#--------Extraction--------
def sizeAreaRatioCheck (imagecv2,croppedcv2,importance):
    """prend en argument une image et une zone d'interet de cette image, ainsi qu'un degré d'importance de la zone d'interet
    renvoie un booléen comparant l'aire d'une image et d'une zone d'interet contenu dans l'image, selon un pourcentage"""
    h1=imagecv2.shape[0]
    w1=imagecv2.shape[1]

    imgsize = h1*w1

    h2=croppedcv2.shape[0]
    w2=croppedcv2.shape[1]

    cropsize = h2*w2

    if cropsize >= imgsize*(importance/100):
        return True
    else:
        return False
    
def sizeAreaRatio(imagecv2,croppedcv2):
    h1=imagecv2.shape[0]
    w1=imagecv2.shape[1]

    imgsize = h1*w1

    h2=croppedcv2.shape[0]
    w2=croppedcv2.shape[1]

    cropsize = h2*w2

    return (cropsize*100)/imgsize

def saveface(img_path,x,y,h,i, output_path):
    img_save = cv2.imread(img_path)
    face = img_save[y:y+h,x:x+h]
    cv2.imwrite(f"{output_path}/face.{i}.jpg",face)

    
def extractface(input_path,detector,importance,margin,output_path):
    """prend en argument deux chemins absolu de dossier/fichier, un degré d'importance et une taille de marge
    charge, analyse puis extrait les visages d'images passées en argument sous forme de chemin absolu vers un fichier
    ou un dossier
    enregistre ensuite les visages extrait selon leur taille dans un dossier destination si il est spécifié"""
    img_paths = load(input_path)

    i=0
    pictures={}
    faces={}
    for img_path in img_paths :
        img_analyse = load_image(img_path)
        


        results = detector.detect_faces(img_analyse)

        for result in results :
            for item in result.items():
                if item[0] == "box":
                    x,y,w,h = item[1]

            centerX = x+(w//2)
            centerY = y+(h//2)
            newX = centerX - (h//2)-margin
            newY =centerY - (h//2)-margin
            face = img_save[newY:newY+h+margin, newX:newX+h+margin]
            
            if face.size != 0 and sizeAreaRatioCheck(img_save,face,importance) :
                
                    

                    facedic = {"centerCoords" : (centerX,centerY), "originalImage":str(img_path)}
                    
                    faces[f"{output_path}/face.{i}.jpg"] = facedic
                    seconds = time.time()
                    localtime = time.ctime(seconds)
                    print(i, localtime, img_path)
                    i+=1
                    

    return faces
            

if __name__=="__main__":
    analysis_directory = os.fsencode(os.path.abspath(os.sys.argv[1]))
    output_directory = os.path.abspath(os.sys.argv[2])
    detector = MTCNN()

    seconds = time.time()
    extractface(analysis_directory,detector,2,30,output_directory)
    print(time.ctime(seconds))
