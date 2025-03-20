export PYTHONPATH=$PWD
echo "!!Training model!!"

python3 src/experiments/01_eval_models.py \
    --save_file_name <SAVEFILE_NAME> \
    --model_name_file <MODEL_PATHS>