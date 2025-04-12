# logisticregression-model-training
This repo has data to train a supervised logisticregression model


## Create pipeline yaml by running 

`kfp dsl compile --py wine_quality_pipeline.py --output wine_quality_pipeline.yaml`

## Created PVC called my-pvc and used in the dsl python files which have componets and pipeline . 

1 . Create PVC called **my-pvc** as I have used that in my pipeline .

2 . To create a pipeline you can also use 

`python kubeflow-dsl-pipeline.py`


## Kubeflow pipeline view 

![Kubeflow Pipeline](https://github.com/devops-mlops-self-projects/logisticregression-model-training/blob/main/images/kubeflow-pipeline.PNG)

## Model training steps

![Training Model](https://github.com/devops-mlops-self-projects/logisticregression-model-training/blob/main/images/train-model.png)
