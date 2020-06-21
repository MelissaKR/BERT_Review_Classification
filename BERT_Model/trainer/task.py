
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from . import BERT_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    parser.add_argument(
        '--model_dir',
        type=str, 
        required=True
    )
    parser.add_argument(
        '--train',
        type=str, 
        required=True
    )
    parser.add_argument(
        '--eval',
        type=str, 
        required=True
    )
    parser.add_argument(
        '--num_train_epochs',
        default = 1.0,
        type= float
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float
    )
    parser.add_argument(
        '--dropout_rate',
        help='percentage of input to drop at Dropout layers',
        default=.2,
        type=float
    )   
    parser.add_argument(
        '--warmup_proportion',
        help = 'Warmup is a period of time when learning rate is small and gradually increases--usually helps training',
        default=0.1,
        type=float
    )

    args, _ = parser.parse_known_args()
    hparams = args.__dict__
    model_dir = hparams.pop('model_dir')
    
    BERT_model.train_and_evaluate(model_dir, hparams)
