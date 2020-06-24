# BERT_Review_Classification
Using the petfood review data from "Amazon Pet Products Reviews" and critical reviews scraped from Chewy.com (detailed in [SmartRev](https://github.com/MelissaKR/SmartRev) project), a BERT classification model using TF Hub BERT module with TensorFlow is structured and deployed on Google AI Platform, with special thanks to [this notebook](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb).

## Training on Google AI Platform

The scripts for the model can be found in BERT_Model/trainer Python package. Environmental variables, such as the directory of the model output and location of train and evaluation data, and hyperparameters, including batch size and number of training epochs, passed to the training job are taken by `task.py` and relayed to `BERT_mode.py` that contains the TensorFlow Estimator model.

Due to the large size of the model, the training job needs to be in parallel across multiple GPUs. The following command can be used to submit the training job to Google AI platform:
 
```bash
OUTDIR=gs://${BUCKET}/trained_model
JOBNAME=reviews_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud --verbosity=debug ai-platform jobs submit training $JOBNAME \
 --region=$REGION \
 --module-name=trainer.task \
 --package-path=${PWD}/BERT_Model/trainer \
 --job-dir=$OUTDIR \
 --scale-tier=custom \
 --master-machine-type=n1-standard-4 \
 --master-accelerator count=4,type=nvidia-tesla-p100 \
 --worker-count=5 \
 --worker-machine-type=n1-standard-4 \
 --worker-accelerator count=4,type=nvidia-tesla-p100 \
 --parameter-server-count 3 \
 --parameter-server-machine-type=n1-standard-4 \
 --runtime-version=1.15 \
 --\
 --model_dir=$OUTDIR \
 --train=gs://${BUCKET}/train.tsv \
 --eval=gs://${BUCKET}/eval.tsv \
 --batch-size=64 \
 --num_train_epochs=10.0
```

## Deployment

The trained model can be deployed on Google AI Platform, using the following `gcloud` commands:

### Creating the Model

```bash
gcloud ai-platform models create ${MODEL_NAME} --regions $REGION
```

### Creating Model Version
```bash
gcloud ai-platform versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=1.15
```

where `MODEL_NAME` is the name of the model, `MODEL_VERSION` is the assigned version of the model, and `MODEL_LOCATION` is the location that the model is saved by the `serving_input` function, e.g. 
```
gsutil ls gs://${BUCKET}/MODEL_DIRECTORY/export/exporter/ | tail -1
```

## Results

With the above setting, the evaluation accuracy of the model at the final training step was 0.44, with `Precision=0.79` and `Recall=1.0`. Compared with the results from the LSTM model structured in [SmartRev](https://github.com/MelissaKR/SmartRev) project, it can be concluded that, in this case, a Recurrent Neural Network model is a better option, both in terms of computational costs, and model performance. 

 
