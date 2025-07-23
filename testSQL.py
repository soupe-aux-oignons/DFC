import mysql.connector 
import comparateurDlib as comp
import extracteur as ex
from mtcnn import MTCNN
import os

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="MySQL@69",
    database="demo"
)

mycursor=mydb.cursor()

#mycursor.execute("CREATE TABLE picturesTest (path VARCHAR(255) PRIMARY KEY, centerCoords VARCHAR(255), originalImage VARCHAR(255))")


analysis_directory = os.fsencode(os.path.abspath(os.sys.argv[1]))
output_directory = os.path.abspath(os.sys.argv[2])
detector = MTCNN()
faces = ex.extract(analysis_directory,detector,2,30,output_directory)

sql="INSERT INTO picturesTest (path,centerCoords,originalImage) VALUES (%s,%s,%s)"

for items in faces.items():
    val=[]
    path = items[0]
    val.append(items[0])
    for value in items[1].values():
        val.append(str(value))
    mycursor.execute(sql,tuple(val))
    mydb.commit()

mycursor.execute("SELECT * FROM picturesTest")
myresult = mycursor.fetchall()
for x in myresult:
    print(x)