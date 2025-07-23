import os
import cv2
from deepface import DeepFace
import extracteur as ex

def isAFace(path):
    pass

def comparateur (path1, path2):
    """prends en argument deux chemins absolus vers un fichier/dossier représentant/contenant des visages 
    compare les visages présents dans les deux fichiers et renvoie une liste de booleen et de distance entre les visages"""
    result = DeepFace.verify(path1, path2, enforce_detection=False)
    return result


def rechercheImageParVisage(visage_path,data_paths):
    """prend en argument un chemin absolu vers un visage et un dossier servant de base de données d'image
    renvoie une liste contenant les chemins absolus vers les images de la base de donnée contenant au moins un visage similaire
    à celui donné en entrée"""

    corresponding_list=[]

    for image_path in data_paths:
        check = comparateur(visage_path, image_path)
        print(check["verified"])
        if check["verified"]:
            corresponding_list.append(image_path)
    return corresponding_list
    

def classificationVisage(visage_path,data_path,unidentified_path, threshold):
    """prend en argument un chemin absolu vers un visage, un dossier contenant des sous dossiers agissant comme une
    base de recenssement d'identité et un dossier servant de recueil aux visages non indentifiés
    vérifie si le visage appartient à une des identités déja enregitrées, si c'est le cas, sauvegarde une copie du visage
    dans le sous dossier de l'identité correspondante
    sinon le visage est copié dans le dossier des visages non identifiés"""
    visage = cv2.imread(visage_path)
    directory_path=os.path.abspath(data_path)

    for sous_dir_path in os.listdir(directory_path):
        sous_dir_path=directory_path+'/'+sous_dir_path
        if os.path.isdir(sous_dir_path) : 
            sous_dir_path_load = ex.load(sous_dir_path)
        else :
            continue
        trues=0
        size_of_directory=0 

        for image_path in sous_dir_path_load:
            check = comparateur(visage_path, image_path)
            if check["verified"]:
                trues+=1
            size_of_directory+=1
        if trues/size_of_directory>=threshold:
            cv2.imwrite(sous_dir_path+'/'+f"visageconnu{len(sous_dir_path_load)}.jpg",visage)
            return
    
    cv2.imwrite(unidentified_path+'/'+f"visageinconnu{len(unidentified_path)}.jpg",visage)


def affichage_liste(list_path):
    for path in list_path:
        image=cv2.imread(path)
        cv2.imshow("image correspondante",image)
        cv2.waitKey()
    cv2.destroyAllWindows()
            

if __name__=="__main__":
    path1=os.sys.argv[1]
    path2=os.sys.argv[2]
    #unidentifiedpath=os.sys.argv[3]
    #classificationVisage(ex.load(path1)[0],path2,ex.load(unidentifiedpath),0.4)
    liste = rechercheImageParVisage(ex.load(path1)[0],ex.load(path2))
    print(liste, len(liste))
    #affichage_liste(liste)

    #comparateur(ex.load(path1)[0], ex.load(path2)[0])

    #print(isAFace(path1))




