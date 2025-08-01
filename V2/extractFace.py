import SQLdatabaseRoam as data
import comparateurDeepface as CDF 
import os

def IDinsertion(table,face_path,results):#assume insertRow works
    """prend en argument un nom de table, un chemin vers un fichier image d'un visage et une liste de données propre aux visages à
    insérer dans la table
    
    parcours les visages de la table en les comparant à celui donné en argument, si l'un d'entre eux est similaire,assigne
    son identité au nouveau visage puis insère le visage et ses données associées dans la table, sinon marque le visage avec
    l'identité 'unknown' et l'insère lui et ses valeurs associées dans la table"""

    colums = "visagePath, boundingbox, nose, mouthright, righteye, lefteye, mouthleft, confidence, originalImage, identity"#flag
    sql = f"SELECT visagePath,identity FROM {table}"#flag
    data.mycursor.execute(sql)#flag
    print("mycursor is this :",data.mycursor)
    myresult = data.mycursor.fetchall()#liste de tuple contenant les resultats de la query SQL
    print("my result is this :", myresult)
    if myresult!=[]:#flag
        for record in myresult:#parcours les chemin et les identités des entrées dans la table, flag
            print(record)
            comparaison = CDF.comparateurDeepface(face_path, record[0])#compare le visage 'face_path' avec le visage du chemin de l'entrée 'record'
            if comparaison["distance"]<=0.5:
                results.append(record[1])#ajoute aux données de 'face_path' l'identité du visage de la table
                print(results)
                data.insertRow(table,colums,tuple(results))#insère le visage et ses données dans la table
                return
    results.append("unknown")#ajoute aux valeurs de 'face_path' l'identité unknown
    print(results)
    data.insertRow(table,colums,tuple(results))#insère le visage et ses données dans la table
    return ("i did something")


def extractFace(detector,input_path,output_path,importance,margin=0):
    """prend en argument un detecteur de visage, deux chemin vers des dossiers/fichiers d'entrée et de sortie, un % d'importance
    et une marge (optionnel)

    enregistre ensuite les visages extrait selon leur taille dans un dossier destination si il est spécifié et enregistre dans une
    base de données les visages un à un avec une identité correspondante"""

    img_paths = data.load(input_path)
    #initialisation de la variable i pour numéroter les visages extraits
    faces_paths=data.load(output_path)#flag
    i=len(faces_paths)#flag

    #ouvre chaque chemin absolu dans la liste de chemin d'entrée
    for img_path in img_paths :
        img_analyse = data.load_image(img_path)
        img_save = data.cv2.imread(img_path)
        
        #insère chaque image originale dans sa table
        #insertRow(imgTable,"imagePath,hauteur,largeur",(img_path,img_save.shape[0],img_save.shape[1]))

        #récupère les données en liste de dictionnaire
        results = detector.detect_faces(img_analyse)
        print(results)

        #étudie chaque dictionnaire obtenu dans la liste:
        for result_dic in results :
            result_list=[]
            x,y,w,h = result_dic["box"]#coord de la bounding box

            #récupère les nouvelles coords de la boundingbox + visage extrait
            face,(x,y,w,h) = data.cropFace(img_save,x,y,h,w,margin)
            
            #si le visage extrait représente un % suffisant de l'image original:
            if  data.sizeAreaRatioCheck(img_save,face,importance) :
                #enregistre un jpg du visage
                data.cv2.imwrite(f"{output_path}/face.{i}.jpg",face)

                #création de la liste de valeurs à insérer dans la table
                result_list.append(f"{output_path}/face.{i}.jpg")
                result_list.append(str((x,y,w,h)))
                for feature in result_dic["keypoints"].items():
                    print(feature)
                    result_list.append(str(feature[1]))
                floatconfidence = (result_dic["confidence"].item())
                result_list.append(floatconfidence)
                result_list.append(img_path)

                print(result_list,"\n")
                #insertion dans la table du visage extrait
                print(IDinsertion(data.faceTable,f"{output_path}/face.{i}.jpg",result_list))
                i+=1



analysis_path = os.path.abspath(os.sys.argv[1])
output_path = os.path.abspath(os.sys.argv[2])
extractFace(data.detector, analysis_path, output_path,2,30)