import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd()
savedir = './'
dataset_list = ['all', 'val', 'novel']

# if not os.path.exists(savedir):
#    os.makedirs(savedir)

for dataset in dataset_list:
    data_path = join(cwd, 'images')
    if dataset == "all":
        data_path = os.path.join(data_path, 'all')
    if dataset == "val":
        data_path = os.path.join(data_path, 'train')
    if dataset == "novel":
        data_path = os.path.join(data_path, "test")

    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]  # 所有类别文件夹
    folder_list.sort()
    label_dict = dict(zip(folder_list, range(0, len(folder_list))))  # 类别文件夹：label

    classfile_list_all = []  # 第i个元素表示label i对应的全部图片

    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list_all.append(
            [join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')])
        random.shuffle(classfile_list_all[i])

    sum = 0
    for i in range(len(classfile_list_all)):
        sum += len(classfile_list_all[i])
    print(dataset, ":", sum)

    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        file_list = file_list + classfile_list
        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" % dataset)
