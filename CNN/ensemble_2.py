#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers2
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size2",64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir2", "ensemble/234/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train2", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement2", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement2", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def do_predict(x_raw):
    

# Map data into vocabulary
    x_raw2 = list()
    x_raw2.append(x_raw)
    
    vocab_path2 = os.path.join(FLAGS.checkpoint_dir2, "..", "vocab")
    vocab_processor2 = learn.preprocessing.VocabularyProcessor.restore(vocab_path2)
    x_test2 = np.array(list(vocab_processor2.transform(x_raw2)))


    graph2 = tf.Graph()
    with graph2.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement2,
          log_device_placement=FLAGS.log_device_placement2)
        sess2 = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        with sess2.as_default():
        # Load the saved meta graph and restore variables
            saver2 = tf.train.import_meta_graph("model-234.meta")
        #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver2.restore(sess2, "model-234")

        # Get the placeholders from the graph by name
            input_x2 = graph2.get_operation_by_name("input_x").outputs[0]
        #    print("input_x : ",input_x2)
        #input_y = graph.get_operation_by_name("input_y").outputs[0]
        #print("input_y done")
            dropout_keep_prob = graph2.get_operation_by_name("dropout_keep_prob").outputs[0]
         #   print("dropout_done")

        # Tensors we want to evaluate
            predictions2 = graph2.get_operation_by_name("output/predictions").outputs[0]
        #    print("prediction_done",predictions2)
        # Generate batches for one epoch
            batches2 = data_helpers2.batch_iter(list(x_test2), FLAGS.batch_size2, 1, shuffle=False)
         #   print("batches_done")
        # Collect the predictions here
            all_predictions2 = []

            for x_test_batch in batches2:
                batch_predictions2 = sess2.run(predictions2, {input_x2: x_test_batch, dropout_keep_prob: 1.0})
                print(x_test_batch, batch_predictions2)
                all_predictions2 = np.concatenate([all_predictions2, batch_predictions2])
#             
        
        return all_predictions2
