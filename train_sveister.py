from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots
import sveister; reload(sveister)
from sveister import Medium




path = "data/sveister/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
batch_size=48

vgg = Medium()
#vgg load weights
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)

for epoch in range(10):
    vgg.fit(batches, val_batches, epoch, nb_epoch=1)
vgg.model.save_weights(model_path+'sveister' + str(epoch + 1) +  '.h5')