"""
Code to train the generation model

"""
from __future__ import print_function

from data.input_pipeline import InputPipeline
from data.input_pipeline_img_128 import InputPipelineImg

from model.idcyclegan_model1_lau import idcyclegan_model1
from model.idcyclegan_model2_lau_128_Chcost import idcyclegan_model2
from model.idcyclegan_model3_128_stride4_UNET import idcyclegan_model3


import os
import re
import tensorflow as tf

#import data.celeba


#
# Configuration for running on ETH GPU cluster
#
'''os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True'''

#
# input flags
#
flags = tf.app.flags
flags.DEFINE_string('mode', 'generate', 'one of [generate, predict, bw2rgb, inpaint]')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train [15]')
flags.DEFINE_integer('batch_size', 64, 'Batch size [16]')
flags.DEFINE_integer('crop_size', 64, 'Crop size to shrink videos [64]')
flags.DEFINE_integer('crop_size_img', 128, 'Crop size to shrink images [128]')
flags.DEFINE_integer('frame_count', 32, 'How long videos should be in frames [32]')
flags.DEFINE_integer('z_dim', 100, 'Dimensionality of hidden features [100]')

flags.DEFINE_integer('read_threads', 16, 'Read threads [16]')

flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate (alpha) for Adam [0.1]')
flags.DEFINE_float('beta1', 0.5, 'Beta parameter for Adam [0.5]')

flags.DEFINE_string('root_dir', '/cluster/scratch/laurenf/IDCycleGAN/videos',
                    'Directory containing all videos and the index file')
flags.DEFINE_string('root_dir_img', '/cluster/scratch/laurenf/IDCycleGAN/test_image_128',
                    'Directory containing all images and the index file')

flags.DEFINE_string('index_file', 'train_data.txt', 'Index file referencing all videos relative to root_dir')
flags.DEFINE_string('index_file_img', 'index_file_img_test_128.txt', 'Index file referencing all images relative to root_dir')
flags.DEFINE_string('type_image', 'jpeg', 'image type')

flags.DEFINE_string('facenet_model', '/cluster/scratch/laurenf/IDCycleGAN/facenet/20180408-102900/20180408-102900.pb', 'dir path of facenet model')

flags.DEFINE_string('experiment_name', 'Experiment_01', 'Log directory')
flags.DEFINE_integer('output_every', 25, 'output loss to stdout every xx steps')
flags.DEFINE_integer('sample_every', 200, 'generate random samples from generator every xx steps')
flags.DEFINE_integer('save_model_every', 200, 'save complete model and parameters every xx steps')
#flags.DEFINE_integer('ngf',64, 'number of generator filters in first conv layer')

flags.DEFINE_bool('recover_model', False, 'recover model')
flags.DEFINE_string('model_name', '', 'checkpoint file if not latest one')
params = flags.FLAGS

#
# make sure all necessary directories are created
#

experiment_dir = os.path.join('/cluster/scratch/laurenf/IDCycleGAN/experiments', params.experiment_name)
checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
sample_dir = os.path.join(experiment_dir, 'samples')
log_dir = os.path.join(experiment_dir, 'logs')

for path in [experiment_dir, checkpoint_dir, sample_dir, log_dir]:
    if not os.path.exists(path):
        os.mkdir(path)



#
# set up input pipeline for images
#
#train_gen, dev_gen = data.celeba.load(params.batch_size, data_dir='/srv/glusterfs/laurenf/IdCycleGAN_FaceTranslation-master/CelebA/64_crop')
#def inf_train_gen():
#     while True:
#         for (images,) in train_gen():
#             yield images
#gen = inf_train_gen()
#batch_img = gen.next()

data_set_img = InputPipelineImg(params.root_dir_img,
                         params.index_file_img,
                         params.read_threads,
                         params.batch_size,
                         num_epochs=params.num_epochs,
                         video_frames=params.frame_count,
                         reshape_size=params.crop_size_img)
batch_img = data_set_img.input_pipeline()


#
# set up input pipeline for videos
#
data_set = InputPipeline(params.root_dir,
                         params.index_file,
                         params.type_image,
                         params.read_threads,
                         params.batch_size,
                         num_epochs=params.num_epochs,
                         video_frames=params.frame_count,
                         reshape_size=params.crop_size)
batch = data_set.input_pipeline()

#
# set up model
#

if params.mode == 'idcyclegan_model1':
    model = idcyclegan_model1(batch_img, batch,
                              batch_size=params.batch_size,
                              frame_size=params.frame_count,
                              crop_size=params.crop_size,
                              crop_size_img=params.crop_size_img,
                              learning_rate=params.learning_rate,
                              beta1=params.beta1)
elif params.mode == 'idcyclegan_model2':
    model = idcyclegan_model2(batch_img, batch,
                              batch_size=params.batch_size,
                              frame_size=params.frame_count,
                              crop_size=params.crop_size,
                              crop_size_img=params.crop_size_img,
                              learning_rate=params.learning_rate,
                              beta1=params.beta1)
elif params.mode == 'idcyclegan_model3':
    model = idcyclegan_model3(batch_img, batch,
                              batch_size=params.batch_size,
                              frame_size=params.frame_count,
                              crop_size=params.crop_size,
                              crop_size_img=params.crop_size_img,
                              learning_rate=params.learning_rate,
                              beta1=params.beta1,
                              #ngf= params.ngf,
                              facenet_model = params.facenet_model)
else:
    raise Exception("unknown training mode")


#
# Set up coordinator, session and thread queues
#

# Saver for model.
saver = tf.train.Saver()
# Coordinator for threads in queues etc.
coord = tf.train.Coordinator()
# Create a session for running operations in the Graph.
#sess = tf.Session(config=config)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Create a summary writer
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
# Initialize the variables (like the epoch counter).
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
# Start input enqueue threads.
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#
# Recover Model
#
if params.recover_model:
    latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest_cp)
    if latest_cp is not None:
        print("restore....")
        saver.restore(sess, latest_cp)
        i = int(re.findall('\d+', latest_cp)[-1]) + 1
    else:
        raise Exception("no checkpoint found to recover")
else:
    i = 0

#saver1 = tf.train.import_meta_graph(params.facenet_model)
#print("loading facenet meta data is over")
#latest_cp = tf.train.latest_checkpoint(params.pretrained_model)
#print(latest_cp)
#saver1.restore(sess,latest_cp)



#if params.recover_model:
#     print("USING MODEL RECOVERY... ")
#     if params.model_name == '':
#         mostRecentCheckpoint = -1
#         for cp in os.listdir(checkpoint_dir):
#             if cp.startswith("cp-"):
#                 cp_num = int(re.findall('\d+', cp)[0])
#                 if cp_num > mostRecentCheckpoint:
#                     mostRecentCheckpoint = cp_num

#         if mostRecentCheckpoint == -1:
#             raise Exception("most recent checkpoint file not found!")

#         checkpoint_file = os.path.join(checkpoint_dir, "cp-{}".format(mostRecentCheckpoint))
#         i = mostRecentCheckpoint + 1
#     else:
#         raise Exception("NOT IMPLEMENTED YET") #TODO
#     print("... checkpoint found: {}".format(checkpoint_file))
#else:
#     i = 0
#     for path in [experiment_dir, checkpoint_dir, sample_dir, log_dir]:
#         if not os.path.exists(path):
#             os.mkdir(path)

#
# backup parameter configurations
#
with open(os.path.join(experiment_dir, 'hyperparams_{}.txt'.format(i)), 'w+') as f:
    f.write('general\n')
    f.write('crop_size: %d\n' % params.crop_size)
    f.write('frame_count: %d\n' % params.frame_count)
    f.write('batch_size: %d\n' % params.batch_size)
    f.write('z_dim: %d\n' % params.z_dim)
    f.write('\nlearning\n')
    f.write('learning_rate: %f\n' % params.learning_rate)
    f.write('beta1 (adam): %f\n' % params.beta1)  # TODO make beta parametrizable in BEGAN as well
    f.close()


#
# TRAINING
#

kt = 0.0
lr = params.learning_rate
k_i = 0
try:
    while not coord.should_stop():

        model.train(sess, i, k_i, summary_writer=summary_writer, log_summary=(i % params.output_every == 0),
                    sample_dir=sample_dir, generate_sample=(i % params.sample_every == 0))
        if i % params.save_model_every == 0:
            print('Backup model ..')
            saver.save(sess, os.path.join(checkpoint_dir, 'cp'), global_step=i)
        i += 1
        k_i +=1

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop and write final checkpoint
    saver.save(sess, os.path.join(checkpoint_dir, 'final'), global_step=i)
    coord.request_stop()

#
# Shut everything down
#
coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()
