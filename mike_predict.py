import cPickle as pickle
import re
import glob
import os
import sys
import time

import theano
import theano.tensor as T
import numpy as np
import pandas as p
import lasagne as nn

from utils import hms, architecture_string


# argument 1: img_dir

dump_path = dump_path = 'dumps/2015_07_17_123003.pkl'
model_data = pickle.load(open(dump_path, 'r'))

# Setting some vars for easier ref.
chunk_size = model_data['chunk_size'] * 2
batch_size = model_data['batch_size']

l_out = model_data['l_out']
l_ins = model_data['l_ins']

# Set up Theano stuff to compute output.
output = nn.layers.get_output(l_out, deterministic=True)
input_ndims = [len(nn.layers.get_output_shape(l_in))
               for l_in in l_ins]
xs_shared = [nn.utils.shared_empty(dim=ndim)
             for ndim in input_ndims]
idx = T.lscalar('idx')

givens = {}
for l_in, x_shared in zip(l_ins, xs_shared):
    givens[l_in.input_var] = x_shared[
        idx * batch_size:(idx + 1) * batch_size
    ]

compute_output = theano.function(
    [idx],
    output,
    givens=givens,
    on_unused_input='ignore'
)



# Get ids of imgs in directory.
def get_img_ids(img_dir):
    test_files = list(set(glob.glob(os.path.join(img_dir, "*.jpg"))))
    test_ids = []

    prog = re.compile(r'(\d+)_(\w+)\.jpg')
    for img_fn in test_files:
        test_id, test_side = prog.search(img_fn).groups()
        test_id = int(test_id)

        test_ids.append(test_id)

    return sorted(set(test_ids))

#give input to img_dir 
prefix_path = "/home/ubuntu/git/eye_images/"
input_dir = sys.argv[1] + "/"
img_dir = prefix_path + input_dir
print(img_dir)

img_ids = get_img_ids(img_dir)

if len(img_ids) == 0:
    raise ValueError('No img ids!\n')

print "\n\nDoing prediction on %s set.\n" % img_dir
print "\n\t%i test ids.\n" % len(img_ids)

# Create dataloader with the test ids.
from jpeg_generator import DataLoader #from jpeg_generator
data_loader = DataLoader()  # model_data['data_loader']
new_dataloader_params = model_data['data_loader_params']
new_dataloader_params.update({'images_test': img_ids})
data_loader.set_params(new_dataloader_params)

if 'paired_transfos' in model_data:
    paired_transfos = model_data['paired_transfos']
else:
    paired_transfos = False

print "\tChunk size: %i.\n" % chunk_size

#computes number of chunks
num_chunks = int(np.ceil((2 * len(img_ids)) / float(chunk_size)))

no_transfo_params = model_data['data_loader_params']['no_transfo_params']

# The default gen with "no transfos".
test_gen = lambda: data_loader.create_fixed_gen(
    data_loader.images_test,
    chunk_size=chunk_size,
    prefix_train=img_dir,
    prefix_test=img_dir,
    transfo_params=no_transfo_params,
    paired_transfos=paired_transfos,
)


def do_pred(test_gen):
    outputs = []

    for e, (xs_chunk, chunk_shape, chunk_length) in enumerate(test_gen()):
        num_batches_chunk = int(np.ceil(chunk_length / float(batch_size)))

        print "Chunk %i/%i" % (e + 1, num_chunks)

        print "  load data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)

        print "  compute output in batches"
        outputs_chunk = []
        for b in xrange(num_batches_chunk):
            out = compute_output(b)
            outputs_chunk.append(out)

        outputs_chunk = np.vstack(outputs_chunk)
        outputs_chunk = outputs_chunk[:chunk_length]
        outputs.append(outputs_chunk)

    return np.vstack(outputs)

# Normal no transfo predict.
outputs = do_pred(test_gen)

test_names = np.vstack([map(lambda x: str(x) + '_left', img_ids),
                        map(lambda x: str(x) + '_right', img_ids)]).T
test_names = test_names.reshape((-1, 1))

print "Saving...\n"
target_path = prefix_path + input_dir + "results"
print test_names.shape
print outputs.shape

np.save(target_path, np.concatenate([test_names, outputs], axis=1))

print "  Outputs saved to %s.\n" % target_path

print(np.load(target_path + ".npy"))

