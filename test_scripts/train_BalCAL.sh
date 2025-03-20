export PYTHONPATH=$PWD
echo "!!Training model!!"

python3 src/experiments/00_train_models.py \
    --model BalCAL \
    --loss CE \
    --epochs 600 \
    --accelerator gpu \
    --seed 1 \
    --dataset CIFAR10 \
    --batch_size 256 \
    --delta 0.95