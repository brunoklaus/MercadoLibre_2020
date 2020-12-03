'''
Created on Mar 21 2020

@author: klaus
'''
import os.path as path
import numpy as np
TOP_DIR_NAME = "src"
def get_src_dir():
    """ Goes up from the current file directory until 'src' is found """
    i = 0
    p = __file__
    while not p.endswith(TOP_DIR_NAME):
        p = path.dirname(p)
        i += 1
        if len(p) < len(TOP_DIR_NAME) or i > 100:
            raise FileNotFoundError("Could not go up till a directory named {} was found".format(TOP_DIR_NAME))
    return(p)    
        
SRC_DIR = get_src_dir()
TOP_DIR = path.dirname(SRC_DIR)
#: Path to the folder destined for plot output.
DATA_DIR = path.join(TOP_DIR,"data")
CSV_DIR = path.join(DATA_DIR,"csv")
TFRECORD_DIR = path.join(DATA_DIR,"tfrecord")



SUBMISSIONS_DIR = path.join(TOP_DIR,"results","submissions")
PLOT_DIR = path.join(TOP_DIR,"results","plots")










if __name__ == '__main__':
    pass