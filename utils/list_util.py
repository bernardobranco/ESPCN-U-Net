import os

def get_filenames():
    with open("output.txt", "w") as a:
        for path, subdirs, files in os.walk('../../SegmentationClassAug'):
            for filename in files:
                print(filename)
                if filename == '.DS_Store':
                    continue
                a.write(filename + os.linesep)

def create_train_txt():
    val_files = [line.rstrip('\n') for line in open('val.txt')]
    #print(val_files)
    train_files = []
    with open("train.txt", "w") as train_txt:
        with open("output.txt", "r") as out_txt:
            for file in out_txt:
                file = file.rstrip('\n')
                file = file.split('.')[0]
                print(file)
                if file not in val_files:
                    train_txt.write(file + os.linesep)
                    #train_files.append(file)
                #print(file.split('.'))
                #print(files)


create_train_txt()