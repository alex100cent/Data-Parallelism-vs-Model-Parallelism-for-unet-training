## Data Parallelism vs Model Parallelism for Unet training
Training speed of Unet has been compared for two distributed training implementations: 
model parallelism and data parallelism.  

You can find all the details in [this article](https://medium.com/deelvin-machine-learning/model-parallelism-vs-data-parallelism-in-unet-speedup-1341bc74ff9e).

### Dataset
[Dataset link](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets). 
You only need the `ds2` folder.    
To get `train.csv` run following command:
```
python3 create_train_csv.py --data-root ds2
```

### Create environment
```
conda env create -f env.yml
```

### Run single gpu training
```
python3 single_gpu_train.py --data-root ds2 --train-csv train.csv --epochs 30 --batch-size 14
```

### Run data parallel training
```
python3 -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 data_parallel.py --data-root ds2 --train-csv train.csv --epochs 30 --batch-size 14
```

### Run model parallel training
```
python3 model_parallel.py --batch-size 30 --split-size 6 --data-root ds2/ --train-csv train.csv --epochs 30
```