#Solyme - May BOUTON

#Import des modeles
from mtcnn import MTCNN
from mtcnn.utils.images import load_image, load_images_batch
from mtcnn.utils.plotting import plot
import face_recognition
from deepface import DeepFace

#Import des librairies d'outils
import mysql.connector
import time
import cv2
import os




#Connection à la base de données SQL
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="MySQL@69",
    database="demo"#il est possible de créer une nouvelle base de donnée puis de nouvelles tables dans celle-ci
)

detector = MTCNN()
imgTable="table1"
faceTable = "table2"

#-----Charger des fichiers-------- Complete
"""Charge correctement des fichiers et dossiers d'images à partir d'un chemin dans l'arborescence"""

def isCorrectFile(filepath):
    """prend en argument un chemin vers un fichier
    
    renvoie si un fichier est un jpg/png ou non"""
    path=os.fsdecode(filepath)
    return (path.endswith(".jpg") or path.endswith(".png") or path.endswith(".JPG") or path.endswith(".PNG")) and (not path.startswith('.'))

def load_file(path):
    """prend en argument un chemin vers un fichier

    renvoie une liste contenant le chemin absolu du fichier"""
    return [os.fsdecode(os.path.abspath(path))]

def load_directory(path):
    """prend en argument un chemin vers un dossier

    renvoie une liste contenant le chemin absolu de chaque png/jpg dans le dossier"""
    path_array = []
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if isCorrectFile(filename):
            path_array.append(os.path.abspath(os.fsdecode(path)+"/"+filename))
    return path_array

def load(path):
    """prend en argument un chemin vers un fichier ou dossier

    renvoie une liste de chemin absolu"""
    if os.path.isfile(path):
        return load_file(path)
    elif os.path.isdir(path):
        return load_directory(path)
    else:
        print("Error : path does not lead to a valid file or directory\n")
        return


#------Base de données MySQL-------- Complete 
"""Fonctions pour créer et alterér des tables dans la base de données SQL"""

#création du curseur lié a la base de données
mycursor = mydb.cursor()

def insertRow(table,strParam,values):
    """prend en argument un nom de table, une chaine contenant le nom des colonnes,les types des colonnes et et les valeurs

    insere une ligne dans la table"""
    sql=f"INSERT INTO {table} ({strParam}) VALUES {values}"
    mycursor.execute(sql)
    mydb.commit()


#------Identificateur------ Complete
"""Identifie un visage en le comparant aux visages déjà présent dans la base de données MySQL"""

def comparateurDlib(img_path1, img_path2, mode):
    """prends en argument deux chemins absolus vers deux images de visages et un mode d'analyse

    compare les visages présents dans les deux fichiers et renvoie une liste de booleen ou de distance entre les visages
    modes acceptés : 'distance', 'booleen' """
    img1_encoding = face_recognition.face_encodings(face_recognition.load_image_file(img_path1))
    img2_encoding = face_recognition.face_encodings(face_recognition.load_image_file(img_path2))

    if img1_encoding==[] or img2_encoding==[]:
        return []
    
    check_bool =[]
    check_dist =[]

    #si mode d'opération booleen, renvoie une liste de booleen numpy
    if mode == "booleen":
        for encoding in img2_encoding :
            check_bool.append(face_recognition.compare_faces(img1_encoding,encoding))
        return check_bool
    
    #si mode d'opération distance, renvoie une liste de float numpy
    elif mode=="distance":
        for encoding in img2_encoding :
            check_dist.append(face_recognition.face_distance(img1_encoding, encoding))
        return check_dist
    
    else :
        print("Error : no mode specified or incorrect mode specified when calling for dlib face comparison")
        return

def comparateurDeepface (img_path1, img_path2):
    """prends en argument deux chemins absolus vers un fichier/dossier représentant/contenant des visages 

    compare les visages présents dans les deux fichiers et renvoie une liste de booleen et de distance entre les visages"""

    result = DeepFace.verify(img_path1, img_path2,model_name="VGG-Face",enforce_detection=False)
    return result

def IDinsertion(table,face_path,results,idface,j):
    """prend en argument un nom de table, un chemin vers un fichier image d'un visage et une liste de données propre aux visages à
    insérer dans la table
    
    parcours les visages de la table en les comparant à celui donné en argument, si l'un d'entre eux est similaire,assigne
    son identité au nouveau visage puis insère le visage et ses données associées dans la table, sinon marque le visage avec
    l'identité 'unknown' et l'insère lui et ses valeurs associées dans la table"""

    colums = "visagePath, boundingbox, nose, mouthright, righteye, lefteye, mouthleft, confidence, originalImage, identity"
    sql = f"SELECT visagePath,identity FROM {table}"
    mycursor.execute(sql)

    #liste de tuple contenant les resultats de la query SQL
    myresult = mycursor.fetchall()

    #si la liste n'est pas vide:
    if myresult!=[]:
        #parcours les chemin et les identités des entrées dans la table
        for record in myresult:
            #compare le visage 'face_path' avec le visage du chemin de l'entrée 'record'
            comparaisons = comparateurDlib(face_path, record[0],'distance')
            if comparaisons!=[]:
                for check in comparaisons:
                    #si la distance entre les visages <=0.5 :
                    if check[0]<=0.5:
                        #ajoute aux données de 'face_path' l'identité du visage de la table
                        results.append(record[1])
                        insertRow(table,colums,tuple(results))#insère le visage et ses données dans la table
                        return 
    #sinon:
    results.append(idface+str(j))#ajoute aux valeurs de 'face_path' l'identité unknown
    insertRow(table,colums,tuple(results))#insère le visage et ses données dans la table
    return (idface)
 

#--------Extraction et enregistrement-------- Complete
"""Extrait les visages d'une image, les identifie grâce aux visages déjà stockés dans la base de données puis les enregistre"""

def sizeAreaRatioCheck (imagecv2,croppedcv2,importance):
    """prend en argument une image et une zone d'interet de cette image, ainsi qu'un % d'importance

    renvoie un booléen comparant l'aire d'une image et d'une zone d'interet contenu dans l'image, selon le %"""
    h1=imagecv2.shape[0]
    w1=imagecv2.shape[1]
    imgsize = h1*w1

    h2=croppedcv2.shape[0]
    w2=croppedcv2.shape[1]
    cropsize = h2*w2

    if cropsize >= imgsize*(importance/100):
        return True
    
def sizeAreaRatio(imagecv2,croppedcv2):
    """prend en argument une image et une zone d'intérêt de l'image
    
    renvoie le pourcentage de l'image que prend la zone d'interêt"""
    h1=imagecv2.shape[0]
    w1=imagecv2.shape[1]
    imgsize = h1*w1

    h2=croppedcv2.shape[0]
    w2=croppedcv2.shape[1]
    cropsize = h2*w2

    return (cropsize*100)/imgsize

def cropFace(img,x,y,h,w, margin):
    """prend en argument une image, des coords xy, la largeur et hauteur d'une zone et une valeur de marge
    
    renvoie une nouvelle image extraite de celle passée en argument selon les valeurs spécifiées"""
    newX = x-margin
    newY = y-margin
    face = img[newY:y+h+margin, newX:x+w+margin]
    return face,(newX,newY,w+margin,h+margin)

def extractFace(detector,input_path,output_path,importance,margin=0,idface="unknown"):
    """prend en argument un detecteur de visage, deux chemin vers des dossiers/fichiers d'entrée et de sortie, un % d'importance (optionnel)
    et une marge (optionnel)

    enregistre ensuite les visages extrait selon leur taille dans un dossier destination si il est spécifié et enregistre dans une
    base de données les visages un à un avec une identité correspondante"""

    img_paths = load(input_path)
    faces_paths=load(output_path)

    #initialisation de la variable i pour numéroter les visages extraits et j les identités
    i=len(faces_paths)
    j=0

    #ouvre chaque chemin absolu dans la liste de chemin d'entrée
    for img_path in img_paths :
        img_analyse = load_image(img_path)
        img_save = cv2.imread(img_path)
        
        #insère chaque image originale dans sa table
        insertRow(imgTable,"imagePath,hauteur,largeur",(img_path,img_save.shape[0],img_save.shape[1]))

        #récupère les données en liste de dictionnaire
        results = detector.detect_faces(img_analyse)

        #étudie chaque dictionnaire obtenu dans la liste:
        for result_dic in results :
            result_list=[]
            x,y,w,h = result_dic["box"]#coord de la bounding box

            #récupère les nouvelles coords de la boundingbox + visage extrait
            face,(x,y,w,h) = cropFace(img_save,x,y,h,w,margin)
            
            #si le visage extrait représente un % suffisant de l'image original:
            if  sizeAreaRatioCheck(img_save,face,importance) :
                #enregistre un jpg du visage
                cv2.imwrite(f"{output_path}/face.{i}.jpg",face)

                #création de la liste de valeurs à insérer dans la table
                result_list.append(f"{output_path}/face.{i}.jpg")
                result_list.append(str((x,y,w,h)))
                for feature in result_dic["keypoints"].items():
                    result_list.append(str(feature[1]))
                floatconfidence = (result_dic["confidence"].item())
                result_list.append(floatconfidence)
                result_list.append(img_path)

                #insertion dans la table du visage extrait
                identifiedface = IDinsertion(faceTable,f"{output_path}/face.{i}.jpg",result_list,idface,j)
                if identifiedface == idface :
                    j+=1
                i+=1
                    

#------Programme Principal-------
if __name__=="__main__":
    analysis_path = os.fsencode(os.path.abspath(os.sys.argv[1]))
    output_path = os.path.abspath(os.sys.argv[2])
    idface = os.sys.argv[3]
    timefile = os.sys.argv[4]
    file = open(timefile,"a")
  
    file.write("heure de début  : " + str(time.ctime(time.time())))
    extractFace(detector,analysis_path,output_path,2,30,idface)
    file.write("heure de fin  : " + str(time.ctime(time.time())))
    file.close()