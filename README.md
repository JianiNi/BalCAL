# Balancing Two Classifiers via A Simplex ETF Structure for Model Calibration
We provide the code for reproducing the results of "Balancing Two Classifiers via A Simplex ETF Structure for Model Calibration". 

As a preliminary, you should create three folders in the repo directory named "data", "eval_path_files" and "experimental_results". You will need these later. 

You can train a base WRN 28-10 for a dataset, run the following command but replacing \<DATASET\> with CIFAR10, SVHN or CIFAR100:

```
python3 src/experiments/00_train_models.py \
    --model_name WRN \
    --loss CE \
    --epochs 600 \
    --accelerator gpu \
    --seed <SEED> \
    --dataset <DATASET> \
    --batch_size 256
```

The trained model found with best validation should be saved in ./experiment_results/\<DATASET\>_WRN/checkpoints. Now to run BalCAL, run the following command-line command:

```
python3 src/experiments/00_train_models.py \
    --model BalCAL \
    --loss CE \
    --epochs 600 \
    --accelerator gpu \
    --seed <SEED> \
    --dataset <DATASET> \
    --batch_size 256 \
    --delta <Delta>
```

where \<DATASET\> must be the same dataset the WRN provided in <PATH_TO_TRAINED_WRN> was trained on. 
 \<SEED\> is a seed integer (we run 1 in BalCAL experiments).


To evaluate the models run (note that you must manually download CIFAR10C and CIFAR100C and place them as directed in the evaluation script):

```
python3 src/experiments/01_eval_models.py \
    --save_file_name <SAVEFILE_NAME> \
    --model_name_file <MODEL_PATHS>
```

where \<SAVEFILE_NAME\> is the name of the file the evaluation metrics are saved in and \<MODEL_PATHS\> is the name of a txt file in the /eval_path_files/ directory containing one or more (local) paths
to a model that one wishes to evaluate.

To evaluate the WRN but with temperature scaling, run:

```
python3 src/experiments/01_eval_models.py \
    --save_file_name <SAVEFILE_NAME> \
    --model_name_file <WRN_TXT_PATH> \
    --temperature_scale
```

where \<WRN_TXT_PATH\> should point to a folder containing the path to the trained WRN.

## References
Our code is mainly based on [`TST and V-TST`](https://github.com/MJordahn/Decoupled-Layers-for-Calibrated-NNs). Thanks to the authors!