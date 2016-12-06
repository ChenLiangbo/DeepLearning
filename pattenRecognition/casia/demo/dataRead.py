#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import struct
import numpy as np
import scipy.misc
import os
import pickle
from sys import stdout

def get_from_file(filepath, reshape=True):
    '''
    parse the gnt file
    return
        bitmap,tag
    '''
    with open(filepath, 'rb') as f:
        while True:
            header = f.read(10)
            if not header: break
            size = struct.unpack('l', header[:4])[0]
            tag = header[4:6]
            width = struct.unpack('h', header[6:8])[0]
            height = struct.unpack('h', header[8:10])[0]
            assert size == 10+width*height
            bitmap = np.fromfile(f, dtype='uint8', count=width*height).reshape((height,width))
            if reshape: bitmap = bitmap_reshape(bitmap)
            yield bitmap,tag