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
    database="demo"
)

detector = MTCNN()#flag
imgTable="images"
faceTable = "visages2"


#-----Charger des fichiers-------- Complete
"""Charge correctement des fichiers et dossiers d'images à partir d'un chemin dans l'arborescence"""

def isCorrectFile(filepath):
    """prend en argument un chemin vers un fichier
    
    renvoie si un fichier est un jpg/png ou non"""
    path=os.fsdecode(filepath)
    return (path.endswith(".jpg") or path.endswith(".png") or path.endswith(".JPG") or path.endswith(".PNG")) and (not path.startswith('.'))

def load_file(path):#flag
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

def tableExists(newtable):
    """prend en argument un nom de table
    
    renvoie True si cette table existe deja dans la base de données étudiée"""
    mycursor.execute("SHOW TABLES")
    myresult = mycursor.fetchall()
    for table in myresult:
        print(table)
        if table[0]==newtable:
            return True
    return False

def createTable(newtable, primaryKeyName, datatype):
    """prend en argument un nom de table, un nom de clé primaire et un type de donnée
    
    vérifie si la table existe, si elle n'existe pas, la créée"""
    mycursor.execute(f"CREATE TABLE {newtable} ({primaryKeyName} {datatype}, PRIMARY KEY ({primaryKeyName}))")

def addColumn(table, columnName, datatype):
    """prend en argument un nom de table, un nom de colonne et un type de donnée

    ajoute une colonne a la table"""
    mycursor.execute(f"ALTER TABLE {table} ADD {columnName} {datatype}")

def addForeignKey(table, keyName, referenceTable):
    """prend en argument un nom de table, un nom de clé et la table de référence

    ajoute une clé étrangère a la table"""
    mycursor.execute(f"ALTER TABLE {table} ADD FOREIGN KEY ({keyName}) REFERENCES {referenceTable}")

def insertRow(table,strParam,values):
    """prend en argument un nom de table, une chaine contenant le nom des colonnes,les types des colonnes et et les valeurs

    insere une ligne dans la table"""
    sql=f"INSERT INTO {table} ({strParam}) VALUES {values}"#flag
    print(sql)
    mycursor.execute(sql)
    mydb.commit()


#------Identificateur------ IDinsertion to finish
"""Identifie un visage en le comparant aux visages déjà présent dans la base de données MySQL"""

def comparateurDlib(img_path1, img_path2, mode):
    """prends en argument deux chemins absolus vers deux images de visages et un mode d'analyse

    compare les visages présents dans les deux fichiers et renvoie une liste de booleen ou de distance entre les visages"""
    img1_encoding = face_recognition.face_encodings(face_recognition.load_image_file(img_path1))
    img2_encoding = face_recognition.face_encodings(face_recognition.load_image_file(img_path2))
    check_bool =[]
    check_dist =[]
    
    if mode == "booleen":
        for encoding in img2_encoding :
            check_bool.append(face_recognition.compare_faces(img1_encoding,encoding))
        return check_bool
    elif mode=="distance":
        for encoding in img2_encoding :
            check_dist.append(face_recognition.face_distance(img1_encoding, encoding))
        return check_dist
    else :
        print("Error : no mode specified or incorrect mode specified when calling for dlib face comparison")
        return





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
    #centerX = x+(w//2)
    #centerY = y+(h//2)
    newX = x-margin
    newY = y-margin
    face = img[newY:y+h+margin, newX:x+w+margin]
    return face,(newX,newY,w+margin,h+margin)

def testResultDic(detector,input_path):
    img_path = load(input_path)[0]
    img_analyse = load_image(img_path)
    result_raw = detector.detect_faces(img_analyse)

    for dictionnary in result_raw:
        liste_attributs=[]
        for tup in dictionnary.items():
            if tup[0]=="confidence":
                liste_attributs.append(tup[1].item())
                print(type(tup[1]), type(tup[1].item()))
                print(tup[1].item())
                continue
            liste_attributs.append(tup[1])
        print(liste_attributs,"\n")


                    

#------Programme Principal-------
if __name__=="__main__":
  analysis_path = os.fsencode(os.path.abspath(os.sys.argv[1]))
  output_path = os.path.abspath(os.sys.argv[2])

  #extractFace(detector,analysis_path,output_path,2,30)
  #testResultDic(detector, analysis_path)
 
