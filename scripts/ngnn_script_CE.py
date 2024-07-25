import setproctitle
import pytorch_lightning as pl
import sys
import os
from argparse import ArgumentParser
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".")
from scripts.script_utils import trainer_setup, test, train, get_cfgs
from utils.select_free_gpu import select_free_gpu
from datasets.CE import CE_datawork
from scripts.script_utils import model_dict, transform_dict

'''
    get args
'''

parser = ArgumentParser()
parser.add_argument("--model", choices=["GeoNGNN"], default="GeoNGNN")
parser.add_argument("--dname", default="r12-0")
parser.add_argument("--combine", action="store_true")
parser.add_argument("--devices", nargs="+", type=int, default=None)
parser.add_argument("--version", default="NO_VERSION")
parser.add_argument("--merge", nargs="+", type=str, default=None)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--proj_name", default=None)

args = parser.parse_args()
# print log
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

from lightningModule.CE_module import CE_module
module = CE_module


model_name = args.model
dataset_name = "CE"
data_name = args.dname
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


pl.seed_everything(seed)

'''
    prepare data
'''
model = model_dict[model_name]
transform = transform_dict[model_name]

setproctitle.setproctitle("GeoNGNN@{}-{}-{}-{}".format(model_name, dataset_name, data_name, version))


train_dl, val_dl, test_dl, _, _ = CE_datawork(
    name=data_name,
    subgraph_cutoff=model_config.subg_cutoff,
    cutoff=model_config.cutoff,
    extend_r=model_config.extend_r,
    combine=args.combine,
    transform=transform,
    max_neighbor=model_config.max_neighbor,
)

'''
    prepare module
'''

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

C = neighbor_num.mean().item() ** - model_config.C_power
subg_C = subg_neighbor_num.mean().item() ** - model_config.C_power

if subg_C > 1.0:
    subg_C = 1.0

print(f"{'-'*10}C={C}{'-'*10}")
print(f"{'-'*10}subg_C={subg_C}{'-'*10}")


module_instance = module(
    model=model,
    model_name=model_name,
    model_config=model_config, 
    optimizer_config=optimizer_config,
    scheduler_config=scheduler_config,
    C=C,
    subg_C=subg_C,
    global_config=global_config, # not used, but for saving hparams
)

'''
    prepare trainer
'''


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
)


'''
    train and test
'''

train(
    trainer=trainer,
    module=module_instance,
    train_dataloader=train_dl,
    val_dataloader=val_dl,
)


test_loss = test(
    trainer=trainer,
    module=module_instance,
    test_dataloader=test_dl,
)
