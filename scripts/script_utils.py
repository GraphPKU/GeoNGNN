from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from yacs.config import load_cfg
from typing import List, Optional
from layers.GeoNGNN import GeoNGNN
from datasets.transform_collate import subg_transform

model_dict = {
    "GeoNGNN": GeoNGNN,
}


transform_dict = {
    "GeoNGNN": subg_transform,
}

def trainer_setup(
    log_path: str, 
    version: Optional[str] = None, 
    enable_model_summary: bool = True,
    early_stopping_patience: int = 20,
    validation_interval: int = 5,
    devices: List[int] = [0],
    max_epochs: int = 3000,
    log_every_n_steps: int = 10,
    accelerator: str = "gpu",
    use_wandb: bool = False,
    proj_name: Optional[str] = None,
    data_name: Optional[str] = None,
    num_sanity_val_steps: int = 0,
    monitor: Optional[str] = "val_loss/val_loss",
    mode: Optional[str] = "min",
    ):

    # config logger
    if use_wandb:
        assert proj_name is not None, "proj_name should not be None when use_wandb is True"
        assert data_name is not None, "data_name should not be None when use_wandb is True"
        run_name = "{}-{}".format(data_name, version)
        run_name = run_name.replace(" ", "-")
        logger = WandbLogger(
            save_dir="logs",
            project=proj_name,
            name=run_name,
            log_model=True,
            version=run_name,
        )
        
    else:
        logger = TensorBoardLogger(
            "logs", 
            name=log_path, 
            version=version
            )
    
    # callback functions

    earlystopping = EarlyStopping(monitor=monitor, patience=early_stopping_patience, mode=mode)
    modelcheckpoint = ModelCheckpoint(monitor=monitor, filename="{epoch}-{step}", mode=mode)
    
    assert len(devices) == 1, "Current version only supports single GPU small-batch training"
    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=max_epochs, 
        accelerator=accelerator, 
        devices=devices, 
        log_every_n_steps=log_every_n_steps, 
        callbacks=[modelcheckpoint, earlystopping],
        check_val_every_n_epoch=validation_interval,
        enable_model_summary=enable_model_summary,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=num_sanity_val_steps
        )    
    return trainer


def train(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    resume: bool = False,
    ckpt_path: Optional[str] = None,
    ):
    if not resume:
        ckpt_path = None

    # train
    trainer.fit(
        model=module, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader, 
        ckpt_path=ckpt_path,
        )

def test(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    test_dataloader: DataLoader,
    skip_train: bool = False,
    ckpt_path: Optional[str] = None,
    test_dataset_name: Optional[str] = None
    ):
    
    if not skip_train:
        ckpt_path = "best"
        
    module.test_dataset_name = test_dataset_name
    
    test_loss = trainer.test(
        module, 
        dataloaders=test_dataloader,
        ckpt_path=ckpt_path
        )
    
    return test_loss



def get_cfgs(
    original_config_path: str, 
    merge_list: Optional[dict] = None, 
    specific_config_path: Optional[str] = None, 
    data_name: Optional[str] = None
    ):
    
    # load config
    with open(original_config_path, "r") as f:
        cfg = load_cfg(f)

    if specific_config_path is not None:
        assert data_name is not None
        with open(specific_config_path, "r") as f:
            s_cfg = load_cfg(f).get(data_name)
        if s_cfg is not None:
            cfg.merge_from_other_cfg(s_cfg)
        else:
            print(f"No specific config for {data_name}")
    
    if merge_list is not None:
        cfg.merge_from_list(merge_list)

        
    return cfg

