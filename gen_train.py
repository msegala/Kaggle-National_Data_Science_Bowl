import os
import sys
import subprocess

#if len(sys.argv) < 3:
#    print "Usage: python gen_train.py input_folder output_folder"
#    exit(1)

#fi = sys.argv[1]
#fo = sys.argv[2]

fi = "/Users/msegala/Documents/Personal/Kaggle/National_Data_Science_Bowl/Data/train/"
fo = "/Users/msegala/Documents/Personal/Kaggle/National_Data_Science_Bowl/Data_converted/train/"


#cmd = "convert -resize 48x48\! "
#cmd = "convert -resize 96x96\! -gravity center -background white -extent 96x96\!" 
cmd = "-resize 80x80 -gravity center -background white -extent 80x80" 

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
        #md += cmd
        #md += fi + cls + "/" + img
        #md += " " + fo + cls + "/" + img
        
        md += fi + cls + "/" + img
        md += " " + cmd
        md += " " + fo + cls + "/" + img

        #print md
        os.system(md)
