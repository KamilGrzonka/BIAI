import os
import cv2

#Paths to three directories from the downloaded dataset
train_dir= 'dataset/train'
val_dir='dataset/val'
test_dir='dataset/test'

#List of selected animal species
animals_detect=["Lion", "Tortoise", "Crocodile", "Deer", "Elephant", "Rhinoceros", "Giraffe", "Horse", "Leopard", "Tiger", "Zebra"]
animals_encoding = {"Lion":0,"Tortoise":1,"Crocodile":2,"Deer":3,"Elephant":4,"Rhinoceros":5,"Giraffe":6,"Horse":7,"Leopard":8,"Tiger":9,"Zebra":10}

#Creating the dataset structure
os.mkdir("yolo")

os.mkdir("yolo/train")
os.mkdir("yolo/train/images")
os.mkdir("yolo/train/labels")

os.mkdir("yolo/val")
os.mkdir("yolo/val/images")
os.mkdir("yolo/val/labels")

os.mkdir("yolo/test")
os.mkdir("yolo/test/images")
os.mkdir("yolo/test/labels")

#Scaling the pictures from the original dataset and copying them to our dataset
size = (640,640)
for animal_specie in animals_detect:
    image_file_name = os.listdir(train_dir+"/"+animal_specie)
    for i in range(0,len(image_file_name)):
            if image_file_name[i] != "Label":
                img = cv2.imread(
                        train_dir+"/"+animal_specie+"/"+image_file_name[i]
                      , cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                cv2.imwrite("yolo/train/images/"+image_file_name[i], img) 

    image_file_name = os.listdir(val_dir+"/"+animal_specie)
    for i in range(0,len(image_file_name)):
            if image_file_name[i] != "Label":
                img = cv2.imread(
                        val_dir+"/"+animal_specie+"/"+image_file_name[i]
                      , cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                cv2.imwrite("yolo/val/images/"+image_file_name[i], img) 

    image_file_name = os.listdir(test_dir+"/"+animal_specie)
    for i in range(0,len(image_file_name)):
            if image_file_name[i] != "Label":
                img = cv2.imread(
                        test_dir+"/"+animal_specie+"/"+image_file_name[i]
                      , cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                cv2.imwrite("yolo/test/images/"+image_file_name[i], img) 

#Scaling the bounding boxes from the original dataset and copying them to our dataset
def process_files(input_files_path,output_files_path):
    for animal_specie in animals_detect:
        txt_file_name = os.listdir(input_files_path+"/"+animal_specie+"/Label")
        for i in range(0,len(txt_file_name)):
                with open(
                       input_files_path
                     + "/" + animal_specie
                     + "/Label/" + txt_file_name[i]
                     , "r") as source:
                       with open(  output_files_path
                                 + "/" + txt_file_name[i]
                                 , "w") as destination :
                            image_file_name_no_ext = txt_file_name[i][0:len(txt_file_name[i])-4]
                            img = cv2.imread(
                                   input_files_path+"/"+animal_specie+"/"+image_file_name_no_ext+".jpg"
                                 , cv2.IMREAD_COLOR)
                            height = img.shape[0]
                            width = img.shape[1]
                            for line in source:
                                labeling_data = line.split()
                                labeling_data[0] = animals_encoding[labeling_data[0]]
                                xmin = float(labeling_data[1])
                                ymin = float(labeling_data[2])
                                xmax = float(labeling_data[3])
                                ymax = float(labeling_data[4])
                                cx = (xmin + xmax)/2.0/width
                                cy = (ymin + ymax)/2.0/height
                                box_width = (xmax - xmin)/width
                                box_height = (ymax - ymin)/height
                                destination.write(str(labeling_data[0])+" ")
                                destination.write(str(cx)+" ")
                                destination.write(str(cy)+" ")
                                destination.write(str(box_width)+" ")
                                destination.write(str(box_height)+"\n")

process_files(train_dir,"yolo/train/labels")
process_files(val_dir,"yolo/val/labels")
process_files(test_dir,"yolo/test/labels")

#Creating YOLO configuration file
with open("data/animals.yaml", "w") as yaml_file:
    yaml_file.write("path: ../yolo\n")
    yaml_file.write("train: train/images\n")
    yaml_file.write("val: val/images\n")
    yaml_file.write("test: test/images\n")
    yaml_file.write("names:"+"\n")
    yaml_file.write(" 0: Lion"+"\n")
    yaml_file.write(" 1: Tortoise"+"\n")
    yaml_file.write(" 2: Crocodile"+"\n")
    yaml_file.write(" 3: Deer"+"\n")
    yaml_file.write(" 4: Elephant"+"\n")
    yaml_file.write(" 5: Rhinoceros"+"\n")
    yaml_file.write(" 6: Giraffe"+"\n")
    yaml_file.write(" 7: Horse"+"\n")
    yaml_file.write(" 8: Leopard"+"\n")
    yaml_file.write(" 9: Tiger"+"\n")
    yaml_file.write(" 10: Zebra"+"\n")
