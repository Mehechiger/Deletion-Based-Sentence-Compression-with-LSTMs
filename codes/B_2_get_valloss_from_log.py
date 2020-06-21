import os
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("file_path")
args = parser.parse_args()

file_path = args.file_path

files = os.listdir(file_path)

pattern = re.compile(r'Val Loss.*\n')

for file_ in files:
    file_name = file_.split('.')[0]
    res = ''
    if file_name[:13] == 'output_epoch_':
        with open(file_path+"/"+file_name+".log", 'r') as f:
            ff = f.read()
            res += "".join(re.findall(pattern, ff))
        with open(file_path+"/"+file_name+"_val_loss.txt", "w") as f:
            f.write(res)
