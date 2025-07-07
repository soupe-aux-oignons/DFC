import cv2 
import os 

if __name__=="__main__":
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    directory = os.fsencode(os.path.abspath(os.sys.argv[1]))

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.abspath(os.sys.argv[1]+"/"+filename)
            img = cv2.imread(img_path)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(grey_img,1.1,5)
            i=0
            for (x,y,w,h) in faces:
                face = img[y:y+h,x:x+w]
                cv2.imwrite(f"{os.path.abspath(os.sys.argv[2])}/face_{i}.jpg",face)
                i+=1



    #prends arguments de run un chemin relatif
    #le transforme en chemin absolu et l'assigne à la variable img_path
   # img_path=os.path.abspath(os.sys.argv[1])

    #assigne le resultat de la lecture de l'image
    #dont le chemin est img_path à la variable img
    #img = cv2.imread(img_path)
    #grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(grey_img, 1.1, 3)

    #i = 0
    #for (x,y,w,h) in faces:
     #   face = img[y:y+h, x:x+w]
      #  cv2.imshow("Cropped", face)
       # cv2.waitKey(0)
        #cv2.imwrite(f"./croppedtestFace_{i}.png", face)
        #i+=1

    #ouvre une fenêtre "Original" pour afficher img
    #cv2.imshow("Original",img)

    #en attente de la pression d'une touche pour continuer le programme
    #cv2.waitKey(0)

    #x,y,w,h = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
    #print(f"{x = }    {y = }   {w = }   {h = }")

    #cropped_img = img[y:y+h, x:x+w]

    #cv2.imshow("Cropped", cropped_img)
    #cv2.waitKey(0)

    #cv2.imwrite("./croppedtestFace.png", cropped_img)

    #ferme toutes les instances de fenetres ouvertes par cv2
    cv2.destroyAllWindows()