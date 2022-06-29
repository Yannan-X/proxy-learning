from aermanager.parsers import parse_nmnist
from aermanager.dataset_generator import gen_dataset_from_folders

'''
This script implements the dataset generation for NMNIST dataset, with the assumption of dataset already been downloaded

from: https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABrCc6FewNZSNsoObWJqY74a?dl=0

and extract at the location where this script is located

Output: folder contained processed *bin files that contains both the original events{x,y,p,t} and frammed data, pls set
the destination_path to where is appropriate
'''

path_to_dataset = "../../workspace/dynap_cnn_benchmark/"

#50ms for train
# gen_dataset_from_folders(source_path=path_to_dataset + "Train",
#                          destination_path="./train_set_time50/",
#                          pattern="*.bin",
#                          time_window=50e3,
#                          parser=parse_nmnist
#                          )


# 250ms for test
gen_dataset_from_folders(source_path=path_to_dataset + "Test",
                         destination_path="./test_set_time50/",
                         pattern="*.bin",
                         time_window=50e3,
                         parser=parse_nmnist
                         )




