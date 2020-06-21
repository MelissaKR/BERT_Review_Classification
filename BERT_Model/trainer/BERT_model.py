
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import re
import os
import json
import argparse
from tensorflow.python.platform import tf_logging
import sys as _sys
import datetime

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

    
CLASSES = {'health': 0, 'quality': 1, 'product':2}  # label-to-int mapping
DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'
labels = [0, 1, 2]
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
MAX_SEQ_LENGTH = 128  # Sentences will be truncated/padded to this length
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

def load_review_data(train_data_path, eval_data_path):
    
    """
        Parses raw tsv containing body of reviews and returns train and eval dataframes 
        containing body of reviews and their labels
            - train_data_path: string, path to tsv containing training data.
                can be a local path or, S3 url (s3://...), or GCS url (gs://...)
            - eval_data_path: string, path to tsv containing eval data.
                can be a local path or a S3 url (s3://...), or GCS url (gs://...)
          Returns:
              train and eval dataframes containing integer labels and body of reviews
    """
    
    # Parse CSV using pandas
    column_names = ('label', 'text')
    df_train = pd.read_csv(train_data_path, names=column_names, sep='\t')
    df_eval = pd.read_csv(eval_data_path, names=column_names, sep='\t')
    
    # Convert labels from text to int
    df_train['label'] = df_train['label'].map(CLASSES)
    df_eval['label'] = df_eval['label'].map(CLASSES)
    
    return df_train, df_eval



def bert_preprocess(df_train, df_eval):
    
    """
        Transforms data into a format understandable by BERT, by creating InputExample's 
        using the constructor provided in the BERT library.
            - guid: globally unique ID for bookkeeping. Not used here
            - text_a: the text we want to classify, i.e. DATA_COLUMN
            - text_b: used when training a model to understand the relationship between sentences. 
                Doesn't apply here, so it is left blank.
            - label: the label or class of the review (0, 1, or 2)
        Returns:
            train_InputExamples and eval_InputExamples
            
    """
    
    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = df_train.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), 
                                                                     axis = 1)

    eval_InputExamples = df_eval.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), 
                                                                   axis = 1)
    
    return train_InputExamples, eval_InputExamples



def bert_tokenizer():
    
    """
        Gets the vocab file and casing info from the BERT Hub module, and
        returns BERT tokenizer
    """
    
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.compat.v1.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])
      
    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, 
                                           do_lower_case=do_lower_case)


def get_features(train_InputExamples, eval_InputExamples, tokenizer):
    
    """
        Converts InputExamples to InputFeatures understandable by BERT, using 
        "convert_examples_to_features" provided in BERT library, and BERT tokenizer
        
        Returns:
            train_features and eval_features
    """
    
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, 
                                                                      labels, 
                                                                      MAX_SEQ_LENGTH, 
                                                                      tokenizer)
    eval_features = bert.run_classifier.convert_examples_to_features(eval_InputExamples, 
                                                                     labels, 
                                                                     MAX_SEQ_LENGTH, 
                                                                     tokenizer)
    return train_features, eval_features



def BERT_model(input_ids, input_mask, segment_ids, label_ids, num_labels,
               dropout_rate,predict):
    """
        Creates a classification model, by loading the BERT tf-hub module to extract the 
        computation graph, and fine-tuning BERT by createing a trainable layer to adapt BERT 
        to the specific classification task.
        
        Returns:
            loss, predicted labels, and log softmax probabilities
    """
    
    ### Define BERT module from BERT_MODEL_HUB, inputs and outputs
    module = hub.Module(BERT_MODEL_HUB,
                        trainable=True)
    
    inputs = dict(input_ids=input_ids,
                  input_mask=input_mask,
                  segment_ids=segment_ids)
    
    outputs = module(inputs=inputs,
                     signature="tokens",
                     as_dict=True)
    
    ### Use "pooled_output" for classification tasks on an entire sentence.
    output_layer = outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    ### Create Trainable BERT layer
    output_weights = tf.compat.v1.get_variable("output_weights", 
                                     [num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.compat.v1.get_variable("output_bias", 
                                  [num_labels], 
                                  initializer=tf.zeros_initializer())

    with tf.compat.v1.variable_scope("loss"):
        output_layer = tf.nn.dropout(output_layer, rate=dropout_rate)   
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # one-hot encoding labels
        one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        
        # Return predicted labels and probabilities in prediction mode
        if predict:
            return (predicted_labels, log_probs)

        # Compute loss in Train and Eval modes
        instance_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(instance_loss)
        return (loss, predicted_labels, log_probs)
    

def metric_fn(label_ids, predicted_labels):
    """
        Adds evaluation metrics to EstimatorSpec when mode == tf.estimator.ModeKeys.EVAL
        
        Returns:
            evaluation accuracy, precision, and recall
    """
    accuracy = tf.compat.v1.metrics.accuracy(label_ids, predicted_labels)
    recall = tf.compat.v1.metrics.recall(label_ids,predicted_labels)
    precision = tf.compat.v1.metrics.precision(label_ids,predicted_labels) 
                
    return {"eval_accuracy": accuracy,
            "precision": precision,
            "recall": recall}


def model_fn_builder(num_labels, learning_rate, dropout_rate,num_train_steps,num_warmup_steps):
    """Returns 'model_fn' used in the Estimator"""
    
    def model_fn(features, labels, mode, params):  

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        predict = (mode == tf.estimator.ModeKeys.PREDICT)
    
        ### TRAIN and EVAL
        if not predict:
            (loss, predicted_labels, log_probs) = BERT_model(input_ids, 
                                                             input_mask, 
                                                             segment_ids, 
                                                             label_ids, 
                                                             num_labels,
                                                             dropout_rate,
                                                             predict)
            
            train_optimizer = bert.optimization.create_optimizer(loss, 
                                                                 learning_rate, 
                                                                 num_train_steps, 
                                                                 num_warmup_steps, 
                                                                 use_tpu=False)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_optimizer)
            
            else:
                eval_metrics = metric_fn(label_ids, predicted_labels)
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        ### Predict
        else:
            (predicted_labels, log_probs) = BERT_model(input_ids, 
                                                       input_mask, 
                                                       segment_ids, 
                                                       label_ids, 
                                                       num_labels,
                                                       dropout_rate,
                                                       predict)

            predictions = {'probabilities': log_probs,
                           'labels': predicted_labels}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the model function
    return model_fn


def serving_input_fn():
    """
        Defines the features to be passed to the model during inference
        Can pass in string text directly. 

        Returns: 
            tf.estimator.export.build_raw_serving_input_receiver_fn
    """
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='segment_ids')
    serve_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()

    return serve_fn



def train_and_evaluate(model_dir, hparams):
    """
        Main orchestrator. Responsible for calling all other functions in BERT_model.py

            model_dir: string, file path where training files will be written
            hparams: dict, command line parameters passed from task.py
    
        Returns: 
            Starts training and evaluation
    """
    
    df_train, df_eval = load_review_data(hparams['train'], hparams['eval'])
    train_InputExamples, eval_InputExamples = bert_preprocess(df_train, df_eval)
    tokenizer = bert_tokenizer()
    train_features, eval_features = get_features(train_InputExamples, eval_InputExamples, tokenizer)
    
    ### Compute number of train and warmup steps from batch size
    num_train_steps = int(len(train_features) / hparams['batch_size'] * hparams['num_train_epochs'])
    num_warmup_steps = int(num_train_steps * hparams['warmup_proportion'])

    ### Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                        save_summary_steps=SAVE_SUMMARY_STEPS,
                                        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    
    model_fn = model_fn_builder(num_labels=len(labels),
                                learning_rate=hparams['learning_rate'],
                                dropout_rate = hparams['dropout_rate'],
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params={"batch_size": hparams['batch_size']})
    
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    
    ### Create input functions for training and evaluating. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(features=train_features,
                                                          seq_length=MAX_SEQ_LENGTH,
                                                          is_training=True,
                                                          drop_remainder=False)
    
    eval_input_fn = run_classifier.input_fn_builder(features=eval_features,
                                                    seq_length=MAX_SEQ_LENGTH,
                                                    is_training=False,
                                                    drop_remainder=False)
    
    train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn,
                                        max_steps=num_train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, 
                                      steps=None,
                                      exporters=exporter,
                                      start_delay_secs=10,
                                      throttle_secs=10)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
