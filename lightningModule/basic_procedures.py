import torch
from utils.GradualWarmupScheduler import GradualWarmupScheduler

                    
def configure_optimizers_(self):
    optimizer_config = self.optimizer_config
    scheduler_config = self.scheduler_config
    
    # initialize AdamW
    optimizer = torch.optim.AdamW(
        params=self.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        amsgrad=True,
        )


    # initialize schedulers
    if scheduler_config.type == "cosine":
        COSINE = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=scheduler_config.cosine_T_max, 
            )
        COSINE_warmup = GradualWarmupScheduler(
            optimizer=optimizer, 
            multiplier=1., 
            total_epoch=scheduler_config.warmup_epoch, 
            after_scheduler=COSINE
            )
        lr_scheduler_configs = {
            "scheduler": COSINE_warmup,
            "interval": "epoch",
        }

    else:
        RLROP = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=scheduler_config.RLROP_factor,
            patience=scheduler_config.RLROP_patience,
            threshold=scheduler_config.RLROP_threshold,
            cooldown=scheduler_config.RLROP_cooldown,
            mode=optimizer_config.mode,
        )

        EXLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_config.EXLR_gamma)
        EXLR_warmup = GradualWarmupScheduler(
            optimizer=optimizer, 
            multiplier=1., 
            total_epoch=scheduler_config.warmup_epoch, 
            after_scheduler=EXLR
            )

        lr_scheduler_configs = []
        for sched in [RLROP, EXLR_warmup]:
            lr_scheduler_config = {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": optimizer_config.monitor,
            }
            lr_scheduler_configs.append(lr_scheduler_config)


    return [optimizer], lr_scheduler_configs
            
