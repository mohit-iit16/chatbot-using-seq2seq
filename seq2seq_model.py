# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""
#model file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from six.moves import xrange
import numpy as np
import tensorflow as tf

import data_utils

class Seq2Seq:
    def __init__(self, source_vocab_size, target_vocab_size, bucket, 
                 layer_size, num_layers, grad_clip, batch_size, learning_rate, 
                 lr_decay_factor, num_samples= 512):
        
        self.source_vocab_size= source_vocab_size
        self.target_vocab_size= target_vocab_size
        self.bucket= bucket
        self.batch_size= batch_size
        self.learning_rate= tf.Variable(float(learning_rate), name= 'learning_rate', trainable= False)
        self.learning_rate_op= self.learning_rate.assign(self.learning_rate*lr_decay_factor)
        self.global_step=tf.Variable(0, trainable=False)
        
        output_projection=None
        softmax_loss_function= None
        
        if num_samples>0 and num_samples<self.target_vocab_size:
        
            w_t= tf.get_variable("projected_weight", [self.target_vocab_size, layer_size] )
            w= tf.transpose(w_t)
            b= tf.get_variable("projected_bias", [self.target_vocab_size])
            output_projection= (w, b)
            
            def sampled_loss(labels, inputs):
                labels= tf.reshape(labels, [-1,1])
                local_wt= tf.cast(w_t, tf.float32)
                local_b= tf.cast(b, tf.float32)
                
                logits= tf.cast(inputs, tf.float32)
                
                return tf.nn.sampled_softmax_loss(weights=local_wt, biases=local_b
                                                  , labels=labels, inputs= logits,
                                                   num_sampled= num_samples, 
                                                   num_classes=self.target_vocab_size )
                
            softmax_loss_function= sampled_loss
            def single_cell():
                
                return tf.contrib.rnn.GRUCell(layer_size)
            
            if num_layers>1:
                cell= tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
                
                
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                        encoder_inputs, decoder_inputs, cell, 
                        num_encoder_symbols= source_vocab_size, 
                        num_decoder_symbols= target_vocab_size, 
                        embedding_size= layer_size, 
                        output_projection= output_projection, 
                        feed_previous= do_decode)
                
            self.encoder_inputs= []
            self.decoder_inputs=[]
            self.target_weights= []
            
            for i in xrange(bucket[-1][0]):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="encoder{0}".format(i)))
                
            for i in xrange(bucket[-1][1]+1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                           name="decoder{0}".format(i)))
                
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], 
                                                          name="weights{0}".format(i)))
                
            targets= [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs)-1)]
            
            
            
            with tf.variable_scope('rnn2'):
                self.outputs, self.losses= tf.contrib.legacy_seq2seq.model_with_buckets(
                                          self.encoder_inputs, self.decoder_inputs, targets,
                                          self.target_weights, bucket,
                                          lambda x, y: seq2seq_f(x, y, False),
                                          softmax_loss_function=softmax_loss_function)
                
            
            params= tf.trainable_variables()
            
            
            self.gradient_norms=[]
            self.updates=[]
            opt= tf.train.AdamOptimizer(self.learning_rate)
            for b in xrange(len(bucket)):
                    gradients=tf.gradients(self.losses[b], params)
                    clipped_gradients, norm= tf.clip_by_global_norm(gradients, grad_clip)
                    
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), 
                                                            global_step= self.global_step))
                    
            self.saver=tf.train.Saver(tf.global_variables())
            
        def step(self, session, encoder_inputs, decoder_inputs, target_weights, 
                 bucket_id):
            
            
            encoder_size, decoder_size= self.bucket[bucket_id]
            if len(encoder_inputs)!= encoder_size:
                raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
            if len(decoder_inputs)!= decoder_size:
                raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
            if len(target_weights)!= decoder_size:
                raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))
                
            input_feed={}
            
            for l in xrange(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
                
            for l in xrange(decoder_size):
                input_feed[self.decoder_inputs[l].name]= decoder_inputs[l]
                input_feed[self.target_weights[l].name]= target_weights[l]
                
                
            
            last_target= self.decoder_inputs[decoder_size.name]
            input_feed[last_target]=np.zeros([self.batch_size], dtype= np.int32)
            
            output_feed= [self.updates[bucket_id], self.gradient_norms[bucket_id],
                               self.losses[bucket_id]]
                
            
            
            outputs= session.run(output_feed, input_feed)
            
            return outputs[1], outputs[2], None
            
            
        def get_batch(self, data, bucket_id):
            encoder_size, decoder_size= self.bucket[bucket_id]
            
            encoder_inputs, decoder_inputs=[], []
            
            for _ in xrange(self.batch_size):
                encoder_input, decoder_input= random.choice(data[bucket_id])
                
                encoder_pad= [data_utils.PAD_ID]*(encoder_size-len(encoder_inputs))
                encoder_inputs.append(list(reversed(encoder_input+encoder_pad)))
                
                decoder_pad_size= decoder_size-decoder_inputs
                decoder_inputs.append([data_utils.GO_ID]+decoder_inputs+[data_utils.PAD_ID*decoder_pad_size])
                
            batch_encoder_inputs, batch_decoder_inputs, batch_weights=[],[],[]
            
            for length_idx in xrange(encoder_size):
                batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] 
                                            for batch_idx in xrange(self.batch_size)], dtype= np.int32))
                    
            for lenght_idx in xrange(decoder_size):
                batch_decoder_inputs.append(
                                            np.array([decoder_inputs[batch_idx][length_idx]
                                            for batch_idx in xrange(self.batch_size)], dtype=np.int32))
                    
                batch_weight= np.ones(self.batch_size, dtype= np.int32)
                for batch_idx in xrange(self.batch_size):
                    if length_idx<decoder_size-1:
                        target= decoder_inputs[batch_idx][lenght_idx+1]
                    if length_idx==decoder_size-1 or target==data_utils.PAD_ID:
                        batch_weight[batch_idx]= 0.0
                
                batch_weights.append(batch_weight)
            return batch_encoder_inputs, batch_decoder_inputs, batch_weights
                        
                    
            
            
            
                 
            
        
        
    
    
    
    
    