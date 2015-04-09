import os
import sys
import subprocess

fi = "/Users/michaelsegala/Documents/GIT_CODE/Kaggle-National_Data_Science_Bowl/Data/train/"
fo = "/Users/michaelsegala/Documents/GIT_CODE/Kaggle-National_Data_Science_Bowl/Data_converted/train/"

cmd = "-resize 60x60 -gravity center -background white -extent 60x60" 

classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
    print cls
    try:
        os.mkdir(cls)
    except:
        pass
    imgs = os.listdir(fi + cls + "/")
    #print imgs
    for img in imgs:
        md = "convert "
        md += fi + cls + "/" + img
        md += " " + cmd
        md += " " + fo + cls + "/" + img

        #print md
        os.system(md)
