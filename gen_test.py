import os
import sys
import subprocess

fi = "/Users/michaelsegala/Documents/GIT_CODE/Kaggle-National_Data_Science_Bowl/Data/test/"
fo = "/Users/michaelsegala/Documents/GIT_CODE/Kaggle-National_Data_Science_Bowl/Data_converted/test/"

cmd = "-resize 60x60 -gravity center -background white -extent 60x60" 

imgs = os.listdir(fi)
#print imgs
for img in imgs:
    md = "convert "
    md += fi + img
    md += " " + cmd
    md += " " + fo + img

    #print md
    os.system(md)
