# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

import scipy.io as sio
import numpy as np

FLAGS = flags.FLAGS


flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")
flags.DEFINE_bool("graphvlad", False,
                  "graphical vlad")
flags.DEFINE_bool("simplegraphvlad", False,
                  "graphical vlad")
flags.DEFINE_bool("addvlad", False,
                  "add vlad")
flags.DEFINE_bool("gruvlad", False,
                  "gru vlad")
flags.DEFINE_bool("acvlad", False,
                  "ac vlad")
flags.DEFINE_bool("localvlad", False,
                  "local vlad")
flags.DEFINE_bool("glulightvlad", False,
                  "GLU light vlad")
flags.DEFINE_bool("convglulightvlad", False,
                  "Conv GLU light vlad")
flags.DEFINE_bool("sqrtvlad", False,
                  "sqrt normalization vlad")
flags.DEFINE_bool("bilinear", False,
                  "bilinear pooling")
flags.DEFINE_bool("nonlocalvlad", False,
                  "nonlocal vlad")

flags.DEFINE_bool("beforeNorm", False,
                  "nonlocal before norm")
flags.DEFINE_bool("afterNorm", False,
                  "nonlocal after norm")

flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 16384,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 2048,
                    "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("dbof_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("dbof_var_features", 0,
                     "Variance features on top of Dbof cluster layer.")

flags.DEFINE_string("dbof_activation", "relu", 'dbof activation')

flags.DEFINE_bool("softdbof_maxpool", False, 'add max pool to soft dbof')

flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                   "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                   "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                   "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")


flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")



flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("fv_relu", True,
                     "ReLU after the NetFV hidden layer.")


flags.DEFINE_bool("fv_couple_weights", True,
                     "Coupling cluster weights or not")
 
flags.DEFINE_float("fv_coupling_factor", 0.01,
                     "Coupling factor")


flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_cells_video", 1024, "Number of LSTM cells (video).")
flags.DEFINE_integer("lstm_cells_audio", 128, "Number of LSTM cells (audio).")



flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_cells_video", 1024, "Number of GRU cells (video).")
flags.DEFINE_integer("gru_cells_audio", 128, "Number of GRU cells (audio).")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                     "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                     "Random sequence input for gru.")
flags.DEFINE_bool("gru_backward", False, "BW reading for GRU")
flags.DEFINE_bool("lstm_backward", False, "BW reading for LSTM")


flags.DEFINE_bool("fc_dimred", True, "Adding FC dimred after pooling")

flags.DEFINE_integer("ConvLayers", 1, "Number of conv layers for ConvSquare_Moe.")
flags.DEFINE_integer("Conv_temporal_field", 2, "Receptive field for convolution.")

class LightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class Bilinear_pooling():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):
        cluster_weights_1 = tf.get_variable("cluster_weights_1",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        cluster_weights_2 = tf.get_variable("cluster_weights_2",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation_1 = tf.matmul(reshaped_input, cluster_weights_1)
        activation_2 = tf.matmul(reshaped_input, cluster_weights_2)
        
        if self.add_batch_norm:
          activation_1 = slim.batch_norm(
              activation_1,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn_1")
          activation_2 = slim.batch_norm(
              activation_2,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn_2")
        else:
          cluster_biases_1 = tf.get_variable("cluster_biases_1",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          cluster_biases_2 = tf.get_variable("cluster_biases_2",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          # tf.summary.histogram("cluster_biases", cluster_biases)
          activation_1 += cluster_biases_1
          activation_2 += cluster_biases_2
        
        # activation = tf.nn.softmax(activation)

        activation_1 = tf.reshape(activation_1, [-1, self.max_frames, self.cluster_size])
        activation_2 = tf.reshape(activation_2, [-1, self.max_frames, self.cluster_size])
       
        activation_1 = tf.transpose(activation_1,perm=[0,2,1])
        
        # reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation_1,activation_2)
        
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.cluster_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class GLULightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])

        # Gated Linear Unit
        linear_reshaped_input = slim.fully_connected(reshaped_input, self.cluster_size, activation_fn=None, scope='GLU_linear')
        gated_reshaped_input = slim.fully_connected(reshaped_input, self.cluster_size, activation_fn=tf.sigmoid, scope='GLU_gated')
        glu_reshape_input = linear_reshaped_input*gated_reshaped_input

        gated_reshaped_input = tf.reshape(gated_reshaped_input,[-1,self.max_frames,self.cluster_size])
        vlad = tf.matmul(activation, gated_reshaped_input)
        
        vlad = tf.transpose(vlad, perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.cluster_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class ConvGLULightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):

        # Conv
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames, 1, self.feature_size])
        conv1_output = tf.contrib.layers.conv2d(reshaped_input, self.feature_size, [4,1], stride=[1,1], scope='conv1')
        conv2_output = tf.contrib.layers.conv2d(reshaped_input, self.feature_size, [4,1], stride=[1,1], scope='conv2')
        reshaped_input = tf.reshape(conv2_output,[-1, self.feature_size])

        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])

        # Gated Linear Unit
        linear_reshaped_input = slim.fully_connected(reshaped_input, self.cluster_size, activation_fn=None, scope='GLU_linear')
        gated_reshaped_input = slim.fully_connected(reshaped_input, self.cluster_size, activation_fn=tf.sigmoid, scope='GLU_gated')
        glu_reshape_input = linear_reshaped_input*gated_reshaped_input

        gated_reshaped_input = tf.reshape(gated_reshaped_input,[-1,self.max_frames,self.cluster_size])
        vlad = tf.matmul(activation, gated_reshaped_input)
        
        vlad = tf.transpose(vlad, perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.cluster_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class LocalLightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)

        vlad = tf.nn.l2_normalize(vlad,2)
        # local_weights = tf.get_variable("local_weights",
        #       [self.feature_size, self.cluster_size],
        #       initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        vlad = tf.reshape(vlad,[-1,self.feature_size])
        vlad = slim.fully_connected(vlad, self.cluster_size, scope='local_connection')
        vlad = tf.reshape(vlad,[-1,self.cluster_size, self.cluster_size])

        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.cluster_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class AC_VLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        
        c_vlad = tf.matmul(vlad, tf.transpose(vlad,perm=[0,2,1]))
        c_vlad = tf.transpose(c_vlad,perm=[0,2,1])
        c_vlad = tf.nn.l2_normalize(c_vlad,1)

        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        c_vlad = tf.reshape(c_vlad, [-1, self.cluster_size*self.cluster_size])
        vlad = tf.concat([vlad, c_vlad], -1)
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class NetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        

        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class NetVLAD_NonLocal():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        
        if FLAGS.afterNorm:
          vlad = tf.nn.l2_normalize(vlad,1) # [b,f,c]
        vlad = tf.transpose(vlad,perm=[0,2,1])

        nonlocal_theta = tf.get_variable("nonlocal_theta",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        nonlocal_phi = tf.get_variable("nonlocal_phi",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        nonlocal_g = tf.get_variable("nonlocal_g",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        nonlocal_out = tf.get_variable("nonlocal_out",
              [self.cluster_size, self.feature_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)))

        vlad_theta = tf.matmul(vlad, nonlocal_theta)
        vlad_phi = tf.matmul(vlad, nonlocal_phi)
        vlad_softmax = tf.nn.softmax(self.feature_size**-.5 * tf.matmul(vlad_theta, tf.transpose(vlad_phi,perm=[0,2,1])))
        vlad_g = tf.matmul(vlad, nonlocal_g)
        vlad_g = tf.matmul(vlad_g, nonlocal_out)
        vlad = vlad + vlad_g

        vlad = tf.transpose(vlad,perm=[0,2,1])
        if FLAGS.beforeNorm:
          vlad = tf.nn.l2_normalize(vlad,1) # [b,f,c]

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class sqrtNetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        
        vlad = tf.sign(vlad)*tf.sqrt(tf.abs(vlad))
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class AddNetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        

        vlad = tf.nn.l2_normalize(vlad,1)

        # vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.reduce_sum(vlad, 2)
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class GRUNetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):
        gru_size = FLAGS.gru_cells
        number_of_layers = FLAGS.gru_layers
        random_frames = FLAGS.gru_random_sequence
        iterations = FLAGS.iterations

        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        

        vlad = tf.nn.l2_normalize(vlad,1)

        # vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.transpose(vlad,perm=[0,2,1]) # [b, c, f]

        stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

        with tf.variable_scope("RNN"):
          outputs, state = tf.nn.dynamic_rnn(stacked_GRU, vlad,
                                             dtype=tf.float32)

        # vlad = tf.reduce_sum(vlad, 2)
        state = tf.nn.l2_normalize(state,1)

        return state


class GraphicalNetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        vlad = tf.nn.l2_normalize(vlad,1)

        graph_weights = tf.get_variable("graph_weights",
              [self.feature_size, self.feature_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        vlad_proj = tf.transpose(vlad,perm=[0,2,1])
        vlad_proj = tf.reshape(vlad_proj, [-1, self.feature_size])
        vlad_proj = tf.matmul(vlad_proj, graph_weights) #[batch, cluster, feature]
        vlad_proj = tf.reshape(vlad_proj, [-1, self.cluster_size, self.feature_size])
        G_vlad = tf.matmul(vlad_proj, tf.transpose(vlad_proj,perm=[0,2,1]))
        G_vlad = tf.nn.softmax(G_vlad, dim=2)

        graph_weights2 = tf.get_variable("graph_weights2",
              [self.feature_size, self.feature_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        vlad = tf.matmul(G_vlad, tf.transpose(vlad,perm=[0,2,1]))
        vlad = tf.reshape(vlad, [-1, self.feature_size])
        vlad = tf.matmul(vlad,graph_weights2)
        vlad = tf.reshape(vlad, [-1, self.cluster_size, self.feature_size])
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class GraphicalNetVLAD_simple():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        vlad = tf.nn.l2_normalize(vlad,1)

        graph_weights = tf.get_variable("graph_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        vlad_proj = tf.transpose(vlad,perm=[0,2,1])
        vlad_proj = tf.reshape(vlad_proj, [-1, self.feature_size])
        vlad_proj = tf.matmul(vlad_proj, graph_weights) #[batch, cluster, feature]
        vlad_proj = tf.reshape(vlad_proj, [-1, self.cluster_size, self.cluster_size])
        G_vlad = tf.matmul(vlad_proj, tf.transpose(vlad_proj,perm=[0,2,1]))
        G_vlad = tf.nn.softmax(G_vlad, dim=2)

        # graph_weights2 = tf.get_variable("graph_weights2",
        #       [self.feature_size, self.feature_size],
        #       initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        vlad = tf.matmul(G_vlad, tf.transpose(vlad,perm=[0,2,1]))
        # vlad = tf.reshape(vlad, [-1, self.feature_size])
        # vlad = tf.matmul(vlad,graph_weights2)
        # vlad = tf.reshape(vlad, [-1, self.cluster_size, self.feature_size])
        # vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)


        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class NetVLAGD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        gate_weights = tf.get_variable("gate_weights",
            [1, self.cluster_size,self.feature_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        gate_weights = tf.sigmoid(gate_weights)

        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])

        vlagd = tf.matmul(activation,reshaped_input)
        vlagd = tf.multiply(vlagd,gate_weights)

        vlagd = tf.transpose(vlagd,perm=[0,2,1])
        
        vlagd = tf.nn.l2_normalize(vlagd,1)

        vlagd = tf.reshape(vlagd,[-1,self.cluster_size*self.feature_size])
        vlagd = tf.nn.l2_normalize(vlagd,1)

        return vlagd




class GatedDBoF():
    def __init__(self, feature_size,max_frames,cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation,1)
        
        activation_max = tf.reduce_max(activation,1)
        activation_max = tf.nn.l2_normalize(activation_max,1)


        dim_red = tf.get_variable("dim_red",
          [cluster_size, feature_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
 
        cluster_weights_2 = tf.get_variable("cluster_weights_2",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights_2", cluster_weights_2)
        
        activation = tf.matmul(activation_max, dim_red)
        activation = tf.matmul(activation, cluster_weights_2)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn_2")
        else:
          cluster_biases = tf.get_variable("cluster_biases_2",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases_2", cluster_biases)
          activation += cluster_biases

        activation = tf.sigmoid(activation)

        activation = tf.multiply(activation,activation_sum)
        activation = tf.nn.l2_normalize(activation,1)

        return activation



class SoftDBoF():
    def __init__(self, feature_size,max_frames,cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation,1)
        activation_sum = tf.nn.l2_normalize(activation_sum,1)

        if max_pool:
            activation_max = tf.reduce_max(activation,1)
            activation_max = tf.nn.l2_normalize(activation_max,1)
            activation = tf.concat([activation_sum,activation_max],1)
        else:
            activation = activation_sum
        
        return activation



class DBoF():
    def __init__(self, feature_size,max_frames,cluster_size,activation, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.activation = activation


    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training

        cluster_weights = tf.get_variable("cluster_weights",
          [feature_size, cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases

        if activation == 'glu':
            space_ind = range(cluster_size/2)
            gate_ind = range(cluster_size/2,cluster_size)

            gates = tf.sigmoid(activation[:,gate_ind])
            activation = tf.multiply(activation[:,space_ind],gates)

        elif activation == 'relu':
            activation = tf.nn.relu6(activation)
        
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        avg_activation = utils.FramePooling(activation, 'average')
        avg_activation = tf.nn.l2_normalize(avg_activation,1)

        max_activation = utils.FramePooling(activation, 'max')
        max_activation = tf.nn.l2_normalize(max_activation,1)
        
        return tf.concat([avg_activation,max_activation],1)

class NetFV():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
     
        covar_weights = tf.get_variable("covar_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(mean=1.0, stddev=1 /math.sqrt(self.feature_size)))
      
        covar_weights = tf.square(covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights,eps)

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [self.cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        if not FLAGS.fv_couple_weights:
            cluster_weights2 = tf.get_variable("cluster_weights2",
              [1,self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        else:
            cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor,cluster_weights)

        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        fv1 = tf.matmul(activation,reshaped_input)
        
        fv1 = tf.transpose(fv1,perm=[0,2,1])

        # computing second order FV
        a2 = tf.multiply(a_sum,tf.square(cluster_weights2)) 

        b2 = tf.multiply(fv1,cluster_weights2) 
        fv2 = tf.matmul(activation,tf.square(reshaped_input)) 
     
        fv2 = tf.transpose(fv2,perm=[0,2,1])
        fv2 = tf.add_n([a2,fv2,tf.scalar_mul(-2,b2)])

        fv2 = tf.divide(fv2,tf.square(covar_weights))
        fv2 = tf.subtract(fv2,a_sum)

        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
      
        fv2 = tf.nn.l2_normalize(fv2,1)
        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2,1)

        fv1 = tf.subtract(fv1,a)
        fv1 = tf.divide(fv1,covar_weights) 

        fv1 = tf.nn.l2_normalize(fv1,1)
        fv1 = tf.reshape(fv1,[-1,self.cluster_size*self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1,1)

        return tf.concat([fv1,fv2],1)

class NetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd
    graphvlad = FLAGS.graphvlad
    addvlad = FLAGS.addvlad
    simplegraphvlad = FLAGS.simplegraphvlad
    gruvlad = FLAGS.gruvlad
    acvlad = FLAGS.acvlad
    localvlad = FLAGS.localvlad
    glulightvlad = FLAGS.glulightvlad
    convglulightvlad = FLAGS.convglulightvlad
    sqrtvlad = FLAGS.sqrtvlad
    nonlocalvlad = FLAGS.nonlocalvlad

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = LightVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAGD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif graphvlad:
      video_NetVLAD = GraphicalNetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = GraphicalNetVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif simplegraphvlad:
      video_NetVLAD = GraphicalNetVLAD_simple(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = GraphicalNetVLAD_simple(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif addvlad:
      video_NetVLAD = AddNetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = AddNetVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif gruvlad:
      video_NetVLAD = GRUNetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = GRUNetVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif acvlad:
      video_NetVLAD = AC_VLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = AC_VLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif localvlad:
      video_NetVLAD = LocalLightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = LocalLightVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif glulightvlad:
      video_NetVLAD = GLULightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = GLULightVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif convglulightvlad:
      video_NetVLAD = ConvGLULightVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = ConvGLULightVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif sqrtvlad:
      video_NetVLAD = sqrtNetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = sqrtNetVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif bilinear:
      video_NetVLAD = Bilinear_pooling(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = Bilinear_pooling(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    elif nonlocalvlad:
      video_NetVLAD = NetVLAD_NonLocal(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAD_NonLocal(128,max_frames,cluster_size/2, add_batch_norm, is_training)
    else:
      video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)

  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

    vlad = tf.concat([vlad_video, vlad_audio],1)

    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)
  
class DbofModelLF(models.BaseModel):
  """Creates a Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    relu = FLAGS.dbof_relu
    cluster_activation = FLAGS.dbof_activation

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if cluster_activation == 'glu':
        cluster_size = 2*cluster_size

    video_Dbof = DBoF(1024,max_frames,cluster_size, cluster_activation, add_batch_norm, is_training)
    audio_Dbof = DBoF(128,max_frames,cluster_size/8, cluster_activation, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBOF"):
        dbof_video = video_Dbof.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_DBOF"):
        dbof_audio = audio_Dbof.forward(reshaped_input[:,1024:])

    dbof = tf.concat([dbof_video, dbof_audio],1)

    dbof_dim = dbof.get_shape().as_list()[1] 

    hidden1_weights = tf.get_variable("hidden1_weights",
      [dbof_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(dbof, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    if relu:
      activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class GatedDbofModelLF(models.BaseModel):
  """Creates a Gated Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    fc_dimred = FLAGS.fc_dimred
    relu = FLAGS.dbof_relu
    max_pool = FLAGS.softdbof_maxpool

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_Dbof = GatedDBoF(1024,max_frames,cluster_size, max_pool, add_batch_norm, is_training)
    audio_Dbof = SoftDBoF(128,max_frames,cluster_size/8, max_pool, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBOF"):
        dbof_video = video_Dbof.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_DBOF"):
        dbof_audio = audio_Dbof.forward(reshaped_input[:,1024:])

    dbof = tf.concat([dbof_video, dbof_audio],1)

    dbof_dim = dbof.get_shape().as_list()[1] 

    if fc_dimred:
        hidden1_weights = tf.get_variable("hidden1_weights",
          [dbof_dim, hidden1_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(dbof, hidden1_weights)

        if add_batch_norm and relu:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="hidden1_bn")
        else:
          hidden1_biases = tf.get_variable("hidden1_biases",
            [hidden1_size],
            initializer = tf.random_normal_initializer(stddev=0.01))
          tf.summary.histogram("hidden1_biases", hidden1_biases)
          activation += hidden1_biases

        if relu:
          activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)
    else:
        activation = dbof

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class SoftDbofModelLF(models.BaseModel):
  """Creates a Soft Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    fc_dimred = FLAGS.fc_dimred
    relu = FLAGS.dbof_relu
    max_pool = FLAGS.softdbof_maxpool

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_Dbof = SoftDBoF(1024,max_frames,cluster_size, max_pool, add_batch_norm, is_training)
    audio_Dbof = SoftDBoF(128,max_frames,cluster_size/8, max_pool, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBOF"):
        dbof_video = video_Dbof.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_DBOF"):
        dbof_audio = audio_Dbof.forward(reshaped_input[:,1024:])

    dbof = tf.concat([dbof_video, dbof_audio],1)

    dbof_dim = dbof.get_shape().as_list()[1] 

    if fc_dimred:
        hidden1_weights = tf.get_variable("hidden1_weights",
          [dbof_dim, hidden1_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(dbof, hidden1_weights)

        if add_batch_norm and relu:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="hidden1_bn")
        else:
          hidden1_biases = tf.get_variable("hidden1_biases",
            [hidden1_size],
            initializer = tf.random_normal_initializer(stddev=0.01))
          tf.summary.histogram("hidden1_biases", hidden1_biases)
          activation += hidden1_biases

        if relu:
          activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)
    else:
        activation = dbof

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    random_frames = FLAGS.lstm_random_sequence
    iterations = FLAGS.iterations
    backward = FLAGS.lstm_backward

    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)
    if backward:
      model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1) 
 
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class GruModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of GRUs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    gru_size = FLAGS.gru_cells
    number_of_layers = FLAGS.gru_layers
    backward = FLAGS.gru_backward
    random_frames = FLAGS.gru_random_sequence
    iterations = FLAGS.iterations
    
    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)
 
    if backward:
        model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1) 
    
    stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


    
class NetFVModelLF(models.BaseModel):
  """Creates a NetFV based model.
     It emulates a Gaussian Mixture Fisher Vector pooling operations
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.fv_cluster_size
    hidden1_size = hidden_size or FLAGS.fv_hidden_size
    relu = FLAGS.fv_relu
    gating = FLAGS.gating

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_NetFV = NetFV(1024,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetFV = NetFV(128,max_frames,cluster_size/2, add_batch_norm, is_training)


    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_FV"):
        fv_video = video_NetFV.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_FV"):
        fv_audio = audio_NetFV.forward(reshaped_input[:,1024:])

    fv = tf.concat([fv_video, fv_audio],1)

    fv_dim = fv.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [fv_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    
    activation = tf.matmul(fv, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
        
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class ConvSquare(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):

      num_layers = FLAGS.ConvLayers
      temporal_field = FLAGS.Conv_temporal_field
      gating = FLAGS.gating
      add_batch_norm = FLAGS.netvlad_add_batch_norm

      if random_frames:
        num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)

      max_frames = model_input.get_shape().as_list()[1]
      feature_size = model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(model_input, [-1, feature_size])
      if add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn")
      model_input = tf.reshape(reshaped_input, [-1, max_frames, feature_size])
      input_ = model_input
      

      for i in range(num_layers):
        with tf.name_scope('conv1_%d' % i) as scope:
          kernel = tf.get_variable("kernel",
                          [temporal_field, feature_size, feature_size // 2],
                          initializer = tf.random_normal_initializer(stddev=0.01))
          conv = tf.nn.conv1d(input_, kernel, 1, padding='SAME')
          conv = tf.sqrt(tf.square(conv))
          # biases = tf.Variable(tf.constant(0.0, shape=[feature_size // 2], dtype=tf.float32),
          #                      trainable=True, name='biases')
          biases = tf.get_variable("biases",
                          [feature_size // 2],
                          initializer = tf.constant_initializer(0.0))
          bias = tf.nn.bias_add(conv, biases)
          input_ = tf.nn.relu(bias, name=scope)
          feature_size = feature_size // 2

      input_ = tf.reduce_mean(input_, 1)

      if gating:
        gating_weights = tf.get_variable("gating_weights_2",
                                         [feature_size, feature_size],
                                         initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        gates = tf.matmul(input_, gating_weights)
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")

        gates = tf.sigmoid(gates)

        input_ = tf.multiply(input_, gates)

      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)
      return aggregated_model().create_model(
          model_input=input_,
          vocab_size=vocab_size, is_training=is_training,
          **unused_params)

class twoStreamLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
      lstm_size = FLAGS.lstm_cells
      number_of_layers = FLAGS.lstm_layers
      random_frames = FLAGS.lstm_random_sequence
      iterations = FLAGS.iterations
      backward = FLAGS.lstm_backward

      if random_frames:
        num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                               iterations)

      video_input = model_input[:,:,0:1024]
      audio_input = model_input[:,:,1024:]
      video_dim = 1024
      audio_dim = 128

      # input embedding
      video_embedding_weights = tf.get_variable("video_embedding_weights",
      [video_dim, lstm_size//2],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(video_dim)))
      audio_embedding_weights = tf.get_variable("audio_embedding_weights",
      [audio_dim, lstm_size//2],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(audio_dim)))

      max_num_frames = video_input.get_shape().as_list()[1]
      video_input = tf.reshape(video_input, [-1, video_dim])
      # video_input = tf.matmul(video_input, video_embedding_weights)
      video_input = slim.fully_connected(video_input, lstm_size//2, activation_fn=tf.tanh, scope='video_embedding', weights_regularizer=slim.l2_regularizer(8e-4))
      video_input = tf.reshape(video_input, [-1, max_num_frames, lstm_size//2])

      audio_input = tf.reshape(audio_input, [-1, audio_dim])
      # audio_input = tf.matmul(audio_input, audio_embedding_weights)
      audio_input = slim.fully_connected(audio_input, lstm_size//2, activation_fn=tf.tanh, scope='audio_embedding', weights_regularizer=slim.l2_regularizer(8e-4))
      audio_input = tf.reshape(audio_input, [-1, max_num_frames, lstm_size//2])


      # if backward:
      #   model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1) 
   
      video_forward_lstm = tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
      video_backward_lstm = tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
      audio_forward_lstm = tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
      audio_backward_lstm = tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
      loss = 0.0

      with tf.variable_scope("Video_RNN"):
        # outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
        #                                    sequence_length=num_frames,
        #                                    dtype=tf.float32)
        v_outputs, v_state = tf.nn.bidirectional_dynamic_rnn(video_forward_lstm, video_backward_lstm, 
                                        video_input, sequence_length=num_frames,
                                           dtype=tf.float32)
        v_outputs = tf.concat(v_outputs, 2)

      with tf.variable_scope("Audio_RNN"):
        a_outputs, a_state = tf.nn.bidirectional_dynamic_rnn(audio_forward_lstm, audio_backward_lstm, 
                                        audio_input, sequence_length=num_frames,
                                           dtype=tf.float32)
        a_outputs = tf.concat(a_outputs, 2)

      video_att_weights = tf.get_variable("video_att_weights",
      [lstm_size*2, 1],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(lstm_size*2)))
      audio_att_weights = tf.get_variable("audio_att_weights",
      [lstm_size*2, 1],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(lstm_size*2)))

      v_outputs_reshape = tf.reshape(v_outputs, [-1, lstm_size*2])
      video_attention = tf.matmul(v_outputs_reshape, video_att_weights)
      video_attention = tf.reshape(video_attention, [-1, max_num_frames, 1])
      video_attention = tf.nn.softmax(video_attention, dim=1)
      v_outputs = tf.multiply(v_outputs, video_attention)
      v_outputs = tf.reduce_sum(v_outputs, 1)

      a_outputs_reshape = tf.reshape(a_outputs, [-1, lstm_size*2])
      audio_attention = tf.matmul(a_outputs_reshape, audio_att_weights)
      audio_attention = tf.reshape(audio_attention, [-1, max_num_frames, 1])
      audio_attention = tf.nn.softmax(audio_attention, dim=1)
      a_outputs = tf.multiply(a_outputs, audio_attention)
      a_outputs = tf.reduce_sum(a_outputs, 1)

      lstm_out = tf.concat([v_outputs, a_outputs], 1)
      lstm_out_up = slim.fully_connected(lstm_out, lstm_size*8, scope='up_proj')
      lstm_out_hidden = slim.fully_connected(lstm_out_up, lstm_size*4, activation_fn=tf.tanh, scope='hidden')

      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)

      return aggregated_model().create_model(
          model_input=lstm_out_hidden,
          vocab_size=vocab_size,
          is_training=is_training,
          **unused_params)


class TRN(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
      iterations = FLAGS.iterations
      gating = FLAGS.gating
      add_batch_norm = FLAGS.netvlad_add_batch_norm
      random_frames = FLAGS.lstm_random_sequence

      if random_frames:
        num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)

      max_frames = model_input.get_shape().as_list()[1]
      feature_size = model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(model_input, [-1, feature_size])
      if add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn")
      model_input = tf.reshape(reshaped_input, [-1, max_frames, feature_size])
      input_ = model_input
      
      with tf.name_scope('TRN_2') as scope:
        kernel = tf.get_variable("kernel_2",
                        [2, feature_size, feature_size // 2],
                        initializer = tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv1d(input_, kernel, 1, padding='SAME')
        conv = tf.sqrt(tf.square(conv))
        # biases = tf.Variable(tf.constant(0.0, shape=[feature_size // 2], dtype=tf.float32),
        #                      trainable=True, name='biases')
        biases = tf.get_variable("biases_2",
                        [feature_size // 2],
                        initializer = tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        TRN_2_output = tf.nn.relu(bias, name=scope)
        TRN_2_output = tf.reduce_mean(TRN_2_output, 1)

      with tf.name_scope('TRN_4') as scope:
        kernel = tf.get_variable("kernel_4",
                        [4, feature_size, feature_size // 2],
                        initializer = tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv1d(input_, kernel, 1, padding='SAME')
        conv = tf.sqrt(tf.square(conv))
        # biases = tf.Variable(tf.constant(0.0, shape=[feature_size // 2], dtype=tf.float32),
        #                      trainable=True, name='biases')
        biases = tf.get_variable("biases_4",
                        [feature_size // 2],
                        initializer = tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        TRN_4_output = tf.nn.relu(bias, name=scope)
        TRN_4_output = tf.reduce_mean(TRN_4_output, 1)

      with tf.name_scope('TRN_8') as scope:
        kernel = tf.get_variable("kernel_8",
                        [8, feature_size, feature_size // 2],
                        initializer = tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv1d(input_, kernel, 1, padding='SAME')
        conv = tf.sqrt(tf.square(conv))
        # biases = tf.Variable(tf.constant(0.0, shape=[feature_size // 2], dtype=tf.float32),
        #                      trainable=True, name='biases')
        biases = tf.get_variable("biases_8",
                        [feature_size // 2],
                        initializer = tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        TRN_8_output = tf.nn.relu(bias, name=scope)
        TRN_8_output = tf.reduce_mean(TRN_8_output, 1)

      avg_TRN = TRN_2_output + TRN_4_output + TRN_8_output
      avg_TRN = avg_TRN / 3.0

      if gating:
        gating_weights = tf.get_variable("gating_weights_2",
                                         [feature_size, feature_size],
                                         initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        gates = tf.matmul(avg_TRN, gating_weights)
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")

        gates = tf.sigmoid(gates)

        avg_TRN = tf.multiply(avg_TRN, gates)

      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)
      return aggregated_model().create_model(
          model_input=avg_TRN,
          vocab_size=vocab_size, is_training=is_training,
          **unused_params)

class CDCModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of GRUs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    gru_size = FLAGS.gru_cells
    number_of_layers = FLAGS.gru_layers
    backward = FLAGS.gru_backward
    random_frames = FLAGS.gru_random_sequence
    iterations = FLAGS.iterations
    add_batch_norm = FLAGS.netvlad_add_batch_norm
    
    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    if add_batch_norm:
      reshaped_input = tf.reshape(model_input, [-1, feature_size])
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")
      model_input = tf.reshape(reshaped_input, [-1, max_frames, 1, feature_size])

      conv1_output = tf.contrib.layers.conv2d(model_input, feature_size*4, [8,1], stride=[1,1], scope='conv1')
      conv1_output = tf.contrib.layers.max_pool2d(conv1_output, [8,1], stride=[8,1], scope='pool1')
      conv2_output = tf.contrib.layers.conv2d(conv1_output, feature_size*4, [8,1], stride=[1,1], scope='conv2')
      conv2_output = tf.contrib.layers.max_pool2d(conv2_output, [8,1], stride=[8,1], scope='pool2')
      deconv1_output = tf.contrib.layers.conv2d_transpose(conv2_output, feature_size*4, [8,1], stride=[8,1], scope='conv1_trans')
      deconv2_output = tf.contrib.layers.conv2d_transpose(deconv1_output, feature_size, [8,1], stride=[8,1], scope='conv2_trans')
 
      model_input = tf.reshape(reshaped_input, [-1, max_frames, feature_size])
      deconv2_output = tf.reshape(deconv2_output, [-1, feature_size])
      deconv2_output = slim.batch_norm(
          deconv2_output,
          center=True,
          scale=True,
          is_training=is_training,
          scope="deconv_bn")
      deconv2_output = tf.reshape(deconv2_output, [-1, max_frames, feature_size])
      model_input = tf.concat([model_input, deconv2_output], 2)
    
    stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)