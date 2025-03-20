from src.lightning_modules.One_Stage import *
import numpy as np
from src.utils.eval_utils import *
import torch.nn as nn
import os
from argparse import ArgumentParser
import json
import argparse
from src.utils.utils import *
import torch
from src.lightning_modules.Two_Stage import *
from lightning.pytorch import seed_everything

parser = ArgumentParser()
parser.add_argument("--save_file_name", type=str, default="")
parser.add_argument("--model_name_file", type=str, default="")
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--alpha", type=float, default=0.8)
parser.add_argument("--lamd", type=float, default=0.8)
parser.add_argument("--delta", type=float, default=1.0)
parser.add_argument("--temperature_scale", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

seed = 1
torch_seed = torch.Generator()
torch_seed.manual_seed(1)
torch_seed.manual_seed(seed)


if args.save_file_name == "":
    raise Exception("Oops you did not provide a save_file_name!")
if args.model_name_file == "":
    raise Exception("Oops you did not provide a model_name_file!")
batch_size=512

model_paths = open("./eval_path_files/"+args.model_name_file+'.txt', "r")

if torch.cuda.is_available():
    device = 'cuda:0'
    device_auto = "gpu"
else:
    device = 'cpu'
    device_auto = "cpu"

if os.path.isfile('./eval_table_metrics/'+args.save_file_name+'.json'):
    f = open('./eval_table_metrics/'+args.save_file_name+'.json', 'r') 
    results = json.load(f) 
else:
    results = {}

for model_path in model_paths.read().splitlines():
    seed_everything(seed, workers=True)
    model_path = model_path.strip()
    model_name = model_path.split("/")[-1].replace(".ckpt", "")
    print("===============================")
    print(model_name)
    model_name = model_name+'_'+str(args.delta)
    dataset = model_name.split("_")[0]
    model_type = model_name.split("_")[1]
    if args.temperature_scale:
        model_name = "Temp-"+model_name

    model_name = model_name.strip()
    results_keys = {key.strip() for key in results.keys()}
    if model_name not in results_keys:
        print("model_name not in results.keys()")
        ood_done = False
        in_done = False
        shift_done = False
        train_done = False
        results[model_name] = {}
    else:
        print("model_name in results.keys()")
        ood_done = True
        in_done = True
        shift_done = True
        train_done = True
        if 'clean_accuracy' not in results[model_name].keys():
            in_done = False
        if 'SHIFT ECE' not in results[model_name].keys():
            shift_done = False
        if 'OOD' not in results[model_name].keys():
            ood_done = False
            in_done = False
        if 'Train NLL' not in results[model_name].keys():
            train_done = False
        if ood_done and in_done and shift_done and train_done:
            print("SKIPPING")
            print(model_name)
            continue
    model = load_model(name=model_type, dataset=dataset, path=model_path, device=device)

    model.eval() 
    model.return_z = False
    model = model.to(device)
    model.device = device

    if args.temperature_scale:
        model = temperature_scale_model(model, dataset, batch_size)
 
    if not train_done:
        print("==================eval_train_data==================")
        nll_value = eval_train_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['Train nll'] = nll_value
    
    if not in_done:
        print("==================eval_test_data==================")
        ece_calc, mce_calc, aece_calc, cece_calc, acc, conf, nll_value, brier_score, OOD_y_preds_logits, OOD_labels = eval_test_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['clean_accuracy'] = acc.to("cpu").numpy().tolist()
        results[model_name]['clean_confidence'] = conf.to("cpu").numpy().tolist()
        results[model_name]['ECE'] = ece_calc.to("cpu").numpy().tolist()*100
        results[model_name]['MCE'] = mce_calc.to("cpu").numpy().tolist()*100
        results[model_name]['AECE'] = aece_calc.to("cpu").numpy().tolist()*100
        results[model_name]['CECE'] = cece_calc.to("cpu").numpy().tolist()*100
        results[model_name]['nll'] = nll_value
        results[model_name]['brier'] = brier_score

    if not ood_done:
        print("==================eval_ood_data==================")
        print("#"+model_name)
        ood_datasets, auroc_calc, fpr_at_95_tpr_calc, confidences, entropies = eval_ood_data(model, dataset=dataset, batch_size=batch_size, device=device, OOD_y_preds_logits=OOD_y_preds_logits, OOD_labels=OOD_labels, num_samples=args.num_samples)
        results[model_name]['OOD'] = 'True'
        for i, ood_dataset in enumerate(ood_datasets):
            results[model_name]["OOD_" + ood_dataset] = {}
            results[model_name]["OOD_" + ood_dataset]['OOD AUROC'] = auroc_calc[i]
            results[model_name]["OOD_" + ood_dataset]['OOD FPR95'] = fpr_at_95_tpr_calc[i]
            results[model_name]["OOD_" + ood_dataset]['confidence'] = confidences[i].to("cpu").numpy().tolist()*100
            results[model_name]["OOD_" + ood_dataset]['entropy'] = entropies[i].to("cpu").numpy().tolist()*100

    if not shift_done:
        print("==================eval_shift_data==================")
        ece_calc, mce_calc, acc, conf, corruption_ece_dict, corruption_mce_dict, corruption_acc_dict, corruption_conf_dict = eval_shift_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['SHIFT ECE'] = ece_calc.to("cpu").numpy().tolist()*100
        results[model_name]['SHIFT MCE'] = mce_calc.to("cpu").numpy().tolist()*100
        results[model_name]['SHIFT ACCURACY'] = acc.to("cpu").numpy().tolist()
        results[model_name]['SHIFT Confidence'] = conf.to("cpu").numpy().tolist()
        for key in corruption_ece_dict.keys():
            results[model_name]["SHIFT Intensity: " + str(key)] = {}
            results[model_name]["SHIFT Intensity: " + str(key)]['Acc'] = corruption_acc_dict[key]
            results[model_name]["SHIFT Intensity: " + str(key)]['Conf'] = corruption_conf_dict[key]
            results[model_name]["SHIFT Intensity: " + str(key)]['ECE'] = corruption_ece_dict[key]
            results[model_name]["SHIFT Intensity: " + str(key)]['MCE'] = corruption_mce_dict[key]

    print("==================save==================")
    with open('./eval_table_metrics/'+args.save_file_name+'.json', 'w') as fp:
        json.dump(results, fp)
