"""
The input pipeline in this file takes care of loading video datasets.
Videos are stored as JPEG files of horizontally stacked frames.

The pipeline takes care of normalizing, cropping and making all videos
the same frame length.
Videos are randomized and put into batches in a multi-threaded fashion.
"""

import tensorflow as tf
import os
import data.celeba



class InputPipelineImg(object):
    def __init__(self, root_dir, index_file, read_threads, batch_size,
                 num_epochs=1, video_frames=32, reshape_size=64):
        """
        :param root_dir: root directory containing the index_file and all the videos
        :param index_file: list of video paths relative to root_dir
        :param read_threads: number of threads used for parallel reading
        :param batch_size: size of the batches to output
        :param num_epochs: number of epochs, use None to make infinite
        :param video_frames: number of frames every video should have in the end
                             if a video is shorter than this repeat the last frame
        :param reshape_size: videos frames are stored as 126x126 images, reshape them to
                             reduce the dimensionality
        """
        self.read_threads = read_threads
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.reshape_size = reshape_size

        with open(os.path.join(root_dir, index_file)) as f:
            content = f.readlines()

        content = [root_dir + '/' + x.strip() for x in content]

        self._filename_queue = tf.train.string_input_producer(content, shuffle=True, num_epochs=num_epochs)

        # self._filename = self._filename_queue.strip().decode('ascii')

    def __read_image(self):
        """
        read one image of the filename queue and return it a image (JPG)
        of horizontally stacked frames
        """
        file_reader = tf.WholeFileReader()

        # print(self._filename_queue)

        _, image_data = file_reader.read(self._filename_queue)
        image = tf.cast(tf.image.decode_jpeg(image_data, channels=3), tf.float32)

        return image

    def __preprocess(self, video):
        """
        takes a image of horizontally stacked video frames and transforms
        it to a tensor of shape:
        [self.video_frames x self.reshape_size x self.reshape_size x 3]
        """
        shape = tf.shape(video)
        frames = tf.reshape(video, [-1, shape[1], shape[1], 3])

        # resize the image
        if shape[1] != self.reshape_size:
            frames = tf.image.resize_images(frames, [self.reshape_size, self.reshape_size])

        return tf.subtract(tf.div(frames, 127.5), 1.0)


    def input_pipeline(self):
        # vid_list = [self.__preprocess(self.__read_video()) for _ in range(self.read_threads)]
        # print vid_list
        img_list = self.__preprocess(self.__read_image())

        # print img_list

        image_batch = tf.train.batch([img_list], batch_size=self.batch_size,
                                     # TODO: check if read_threads here actually speeds things up
                                     num_threads=self.read_threads, capacity=self.batch_size * 4, enqueue_many=True,
                                     shapes=[self.reshape_size, self.reshape_size, 3])
        return image_batch



# class InputPipelineImg(object):
#     def __init__(self, root_dir, index_file, read_threads, batch_size,
#                  num_epochs=1, video_frames=32, reshape_size=64):
#         """
#         :param root_dir: root directory containing the index_file and all the videos
#         :param index_file: list of video paths relative to root_dir
#         :param read_threads: number of threads used for parallel reading
#         :param batch_size: size of the batches to output
#         :param num_epochs: number of epochs, use None to make infinite
#         :param video_frames: number of frames every video should have in the end
#                              if a video is shorter than this repeat the last frame
#         :param reshape_size: videos frames are stored as 126x126 images, reshape them to
#                              reduce the dimensionality
#         """
#         self.read_threads = read_threads
#         self.batch_size = batch_size
#         self.video_frames = video_frames
#         self.reshape_size = reshape_size
#         self.dir = root_dir
#
#         self.train_gen, dev_gen = data.celeba.load(self.batch_size, data_dir=self.dir)
#
#     def _inf_train_gen(self):
#         while True:
#             train_images = tf.random_shuffle(self.train_gen())
#             for (images,) in self.train_gen(train_images):
#                 # yield images
#                 yield tf.subtract(tf.div(tf.cast(images, tf.float32), 127.5), 1.0)
#
#
#     def __read_video(self):
#         gen = self._inf_train_gen()
#         return gen.next()
#
#
#
#     def input_pipeline(self):
#         # vid_list = [self.__preprocess(self.__read_video()) for _ in range(self.read_threads)]
#
#         img_list = self.__read_video()
#
#         # img_list = gen
#
#         img_batch = tf.train.batch([img_list], batch_size=self.batch_size,
#                                      # TODO: check if read_threads here actually speeds things up
#                                      num_threads=self.read_threads, capacity=self.batch_size * 4, enqueue_many=True,
#                                      shapes=[self.reshape_size, self.reshape_size, 3])
#
#         return img_batch
