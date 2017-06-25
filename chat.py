from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  
import tensorflow as tf

import data_utils
import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("enc_vocabulary_size", 40000, "Input vocabulary size.")
tf.app.flags.DEFINE_integer("dec_vocabulary_size", 40000, "Output vocabulary size")
tf.app.flags.DEFINE_string("data_dir", "/Users/mohitkumar/seq2seq", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/Users/mohitkumar/seq2seq/ckpt/", "Training directory.")
tf.app.flags.DEFINE_string("encoder_input_data", "/Users/mohitkumar/seq2seq/enc.txt", "Training data.")
tf.app.flags.DEFINE_string("decoder_input_data", "/Users/mohitkumar/seq2seq/dec.txt", "Training data.")
tf.app.flags.DEFINE_string("encoder_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("decoder_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", True,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean('test', False, "true for testing [false] for training")



FLAGS= tf.app.flags.FLAGS
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def main(_):
    if FLAGS.test:
        test()
    else:
        train()
        


def create_model(session):
        
    model= seq2seq_model.Seq2Seq(FLAGS.enc_vocabulary_size, FLAGS.dec_vocabulary_size, 
                                 _buckets, FLAGS.size, FLAGS.num_layers, 
                                 FLAGS.max_gradient_norm, FLAGS.batch_size, 
                                 FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, 
                                )
    ckpt=tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("reading model parameters from training directory")
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("creating model in the training directory")
        session.run(tf.global_variables_initializer())
    return model

def read_data(encoder_input_dir, decoder_input_dir, max_size=None):
    data_set=[[] for _ in _buckets]
    with tf.gfile.GFile(encoder_input_dir, mode="r") as enc_input:
        with tf.gfile.GFile(decoder_input_dir, mode="r") as dec_input:
            encoder, decoder= enc_input.readline(), dec_input.readlines()
            counter=0
            while encoder and decoder and (not max_size or counter<max_size):
                counter+=1
                if counter % 100000 ==0:
                    print("reading line:", counter)
                    sys.stdout.flush()
                encoder_ids=[int(x) for x in encoder.split()]
                decoder_ids=[int(x) for x in decoder.split()]
                decoder_ids.append(data_utils.EOS_ID)
                
                for bucket_id , (encoder_size, decoder_size) in enumerate(_buckets):
                    if len(encoder_ids)<encoder_size and len(decoder_ids)<decoder_size:
                        data_set[bucket_id].append([encoder_ids, decoder_ids])
                        break
                encoder, decoder= enc_input.readline(), dec_input.readlines()
    return data_set

        
def train():
    print('running train')
    "train a model on chat data"
    
    encoder_input = None
    decoder_input = None
    encoder_dev = None
    decoder_dev = None
    if FLAGS.encoder_input_data and FLAGS.decoder_input_data:
        encoder_input = FLAGS.encoder_input_data
        decoder_input = FLAGS.decoder_input_data
        encoder_dev = encoder_input
        decoder_dev = decoder_input
        encoder_input, decoder_input, encoder_dev, decoder_dev, _,_= data_utils.prepare_chat_data(
                FLAGS.data_dir, encoder_input, decoder_input, encoder_dev, 
                decoder_dev, FLAGS.enc_vocabulary_size, FLAGS.enc_vocabulary_size)
    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model=create_model(sess)
        print("reading training and development data (limit: %d)." %FLAGS.max_train_data_size )
        dev_set= read_data(encoder_dev, decoder_dev)
        train_set= read_data(encoder_input, decoder_input, FLAGS.max_train_data_size)
        train_bucket_sizes=[len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        
        train_bucket_scale= [sum(train_bucket_sizes[:i+1])/train_total_size for i in xrange(len(train_bucket_sizes))]
        
        step_time, loss= 0.0, 0.0
        current_step=0
        previous_losses=[]
        
        while True:
            #random_number= np.random.random_integers(0,high=4)
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_bucket_scale))
                       if train_bucket_scale[i] > random_number_01])
            start_time= time.time()
            
            enc_inp, dec_inp, target_weights= model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, enc_inp, dec_inp,
                                   target_weights, bucket_id)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            
            if current_step % FLAGS.steps_per_checkpoint==0:
                perplexity= math.exp(float(loss)) if loss<300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path= os.path.join(FLAGS.train_dir, "chat.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                
                step_time, loss=0.0,0.0
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    enc_inp, dec_inp, target_weights = model.get_batch(
                                                dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, enc_inp, dec_inp,
                                       target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()
                
def decode():
    with tf.Session() as sess:
    # Create model and load parameters.
        model = create_model(sess)
        model.batch_size = 1  # We decode one sentence at a time.
    
        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.from" % FLAGS.from_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.to" % FLAGS.to_vocab_size)
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)
    
        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
          # Get token-ids for the input sentence.
          token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
          # Which bucket does it belong to?
          bucket_id = len(_buckets) - 1
          for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
              bucket_id = i
              break
          else:
            logging.warning("Sentence truncated: %s", sentence)
    
          # Get a 1-element batch to feed the sentence to the model.
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
          # Get output logits for the sentence.
          _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
          # This is a greedy decoder - outputs are just argmaxes of output_logits.
          outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
          # If there is an EOS symbol in outputs, cut them at that point.
          if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
          # Print out French sentence corresponding to outputs.
          print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
          print("> ", end="")
          sys.stdout.flush()
          sentence = sys.stdin.readline()
      
def test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)
            
            
    
    
if __name__ == "__main__":
  tf.app.run()
        
        
    
    
    
    