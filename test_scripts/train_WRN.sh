export PYTHONPATH=$PWD
echo "!!Training model!!"

python3 src/experiments/00_train_models.py \
    --loss CE \
    --epochs 600 \
    --accelerator gpu \
    --seed 1 \
    --dataset CIFAR10 \
    --model_name WRN \
    --batch_size 256