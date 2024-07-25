import setproctitle
import pytorch_lightning as pl
import sys
import time
import os
from argparse import ArgumentParser
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".")
from scripts.script_utils import trainer_setup, test, train, get_cfgs
from utils.select_free_gpu import select_free_gpu
from datasets.triBPA import triBPA_datawork
from scripts.script_utils import model_dict, transform_dict

'''
    get args
'''
parser = ArgumentParser()
parser.add_argument("--model", choices=["GeoNGNN"], default="GeoNGNN")
parser.add_argument("--devices", nargs="+", type=int, default=None)
parser.add_argument("--data_dir", default="~/datasets/triBPA")
parser.add_argument("--version", default="NO_VERSION")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--ckpt", default=None)
parser.add_argument("--merge", nargs="+", type=str, default=None)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--proj_name", default=None)

args = parser.parse_args()
# print log
print(f"{'-'*10}ARGS{'-'*10}")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

from lightningModule.MD_module import MD_module



dataset_name="triBPA"
data_name = "triBPA"
model_name = args.model
data_dir = args.data_dir
resume = args.resume
checkpoint_path = args.ckpt
skip_train = args.skip_train
version = args.version
devices = args.devices
merge_list = args.merge

accelerator = "gpu"
if devices is None:
    devices = [select_free_gpu()]
elif devices == [-1]:
    devices = None
    accelerator = "cpu"


'''
    get hparams
'''
config_path = "hparams/{}_{}.yaml".format(model_name, dataset_name)
specific_config_path = "hparams/specific/{}_{}_specific.yaml".format(model_name, dataset_name)
if not os.path.exists(specific_config_path):
    specific_config_path = None
config = get_cfgs(config_path, merge_list, specific_config_path, data_name)

scheduler_config = config.scheduler_config
optimizer_config = config.optimizer_config
model_config = config.model_config

# trainer_config
trainer_config = config.trainer_config
validation_interval = trainer_config.validation_interval
log_every_n_steps = 100
early_stopping_patience = trainer_config.early_stopping_patience
max_epochs = trainer_config.max_epochs

# global_config
global_config = config.global_config
seed = config.global_config.seed

# data_config
data_config = config.data_config
train_batch_size = data_config.train_batch_size
val_batch_size = data_config.val_batch_size
test_batch_size = data_config.test_batch_size




print(f"{'-'*10}MODEL CONFIG{'-'*10}")
print(model_config)
print(f"{'-'*10}OPTIMIZER CONFIG{'-'*10}")
print(optimizer_config)
print(f"{'-'*10}SCHEDULER CONFIG{'-'*10}")
print(scheduler_config)
print(f"{'-'*10}DATA CONFIG{'-'*10}")
print(data_config)
print(f"{'-'*20}")


    
pl.seed_everything(seed)

'''
    prepare data
'''
predict_force = True
datawork = triBPA_datawork
'''
    prepare model and transform
'''

model = model_dict[model_name]

transform = transform_dict[model_name]
    


setproctitle.setproctitle("GeoNGNN@{}-{}-{}-{}".format(model_name, dataset_name, data_name, version))




train_dl, val_dl, test_dl, global_y_mean, global_y_std = datawork(
    root=data_dir,
    name=data_name,
    batch_size=[train_batch_size, val_batch_size, test_batch_size],
    subgraph_cutoff=model_config.subg_cutoff,
    cutoff=model_config.cutoff,
    extend_r=model_config.extend_r,
    transform=transform,
    max_neighbor=model_config.max_neighbor,
)

import torch
from torch_scatter import scatter
from tqdm import tqdm

neighbor_num = torch.empty([0], dtype=torch.long, device=f"cuda:{devices[0]}")
subg_neighbor_num = torch.empty([0], dtype=torch.long, device=f"cuda:{devices[0]}")

for data in tqdm(iter(train_dl)):

    row, col = data.edge_index.to(f"cuda:{devices[0]}")
    neighbor_num_add = scatter(torch.ones_like(row).float(), row, dim=0, reduce='sum', dim_size=row.max()+1)
    neighbor_num = torch.cat([neighbor_num, neighbor_num_add])

    row, col = data.subg_edge_index.to(f"cuda:{devices[0]}")
    try:
        subg_neighbor_num_add = scatter(torch.ones_like(row).float(), row, dim=0, reduce='sum', dim_size=row.max()+1)
        subg_neighbor_num = torch.cat([subg_neighbor_num, subg_neighbor_num_add])
    except:
        print("No subgraph")
        
        
# To calculate the meta information
# largest_distance = -1.
# for data in tqdm(iter(train_dl)):
#     largest_distance_ = (data.pos.unsqueeze(0) - data.pos.unsqueeze(1)).norm(dim=-1).max().item()
#     largest_distance = max(largest_distance, largest_distance_)
# for data in tqdm(iter(val_dl)):
#     largest_distance_ = (data.pos.unsqueeze(0) - data.pos.unsqueeze(1)).norm(dim=-1).max().item()
#     largest_distance = max(largest_distance, largest_distance_)
# for i in  range(3):
#     for data in tqdm(iter(test_dl[i])):
#         largest_distance_ = (data.pos.unsqueeze(0) - data.pos.unsqueeze(1)).norm(dim=-1).max().item()
#         largest_distance = max(largest_distance, largest_distance_)
        
# print(largest_distance)
    
    
C = neighbor_num.mean().item() ** - model_config.C_power
subg_C = subg_neighbor_num.mean().item() ** - model_config.C_power

if subg_C > 1.0:
    subg_C = 1.0

print(f"{'-'*10}C={C}{'-'*10}")
print(f"{'-'*10}subg_C={subg_C}{'-'*10}")
if (skip_train or resume) and checkpoint_path is not None:
    module_instance = MD_module.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False
        )
else:
    module_instance = MD_module(
        model_config=model_config, 
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        global_y_std=global_y_std, 
        global_y_mean=global_y_mean,
        C=C,
        subg_C=subg_C,
        model=model,
        data_config=data_config, # not used, but for saving hparams
        global_config=global_config, # not used, but for saving hparams
    )



log_path = "{}_log/{}/{}".format(
    model_name, 
    dataset_name, 
    data_name
    ) 

trainer = trainer_setup(
    log_path=log_path,
    version=version,
    early_stopping_patience=early_stopping_patience,
    max_epochs=max_epochs,
    validation_interval=validation_interval,
    devices=devices,
    log_every_n_steps=log_every_n_steps,
    accelerator=accelerator,
    use_wandb=args.use_wandb,
    proj_name=args.proj_name if args.proj_name is not None else f"{model_name}_{dataset_name}",
    data_name=data_name,
    num_sanity_val_steps=-1 if resume else 2,
    monitor=optimizer_config.monitor,
    mode=optimizer_config.mode,
)





if not skip_train:
    start_time = time.time()
    train(
        trainer=trainer,
        module=module_instance,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        resume=resume,
        ckpt_path=checkpoint_path,
    )
    end_time = time.time()





    
for i in range(len(test_dl)):
    tdl = test_dl[i]
    names = ["1", "2", "3"]
    test_loss = test(
        trainer=trainer,
        module=module_instance,
        test_dataloader=tdl,
        skip_train=skip_train,
        ckpt_path=checkpoint_path,
        test_dataset_name=names[i]
    )

