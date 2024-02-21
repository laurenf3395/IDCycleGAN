import numpy as np
import scipy.misc
import time
import os
from glob import glob

path='/cluster/scratch/laurenf/IDCycleGAN/test_image_128'
def make_generator(path, batch_size):

    epoch_count = [1]
    def get_epoch():
        # images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        images = np.zeros((batch_size, 128, 128, 3), dtype='int32')


        if os.path.exists(path + '.npy'):
            data = np.load(path + '.npy')
        else:
            data = sorted(glob(os.path.join(path, "*.*")))
            np.save(path + '.npy', data)

        random_order = np.random.permutation(len(data))
        data = [data[i] for i in random_order[:]]

        epoch_count[0] += 1
        for idx in xrange(0, len(data)):

            batch_files = data[idx]
            # print batch_files

            image = scipy.misc.imread("{}".format(batch_files))
            # images[idx % batch_size] = image.transpose(2,0,1)
            images[idx % batch_size] = image

            if idx > 0 and idx % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir='/cluster/scratch/laurenf/IDCycleGAN/test_image_128'):
    return (
        make_generator(data_dir, batch_size),
        make_generator(data_dir, batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
