import pytorch_lightning as pl
from utils.activation_fns import activation_fn_map
from utils.EMA import ExponentialMovingAverage
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from lightningModule.basic_procedures import configure_optimizers_


class CE_module(pl.LightningModule):
    def __init__(
        self, 
        model,
        C: float,
        subg_C: float,
        model_config: dict,
        optimizer_config: dict,
        scheduler_config: dict,
        global_y_std: float = 1.,
        global_y_mean: float = 0.,
        **kwargs
        ):
        super().__init__()
        
        self.save_hyperparameters()
        self.model = model(
                z_hidden_dim=model_config.z_hidden_dim,
                hidden_dim=model_config.hidden_dim,
                ef_dim=model_config.ef_dim,
                rbf=model_config.rbf,
                max_z=model_config.max_z,
                outer_rbound_upper=model_config.cutoff + 2.0,
                inner_rbound_upper=model_config.subg_cutoff + 2.0,
                activation_fn=activation_fn_map[model_config.activation_fn],
                inner_layer_num=model_config.inner_layer_num,
                outer_layer_num=model_config.outer_layer_num,
                inner_cutoff=model_config.subg_cutoff,
                outer_cutoff=model_config.cutoff,
                predict_force=False,
                global_y_mean=global_y_mean,
                global_y_std=global_y_std,
                C=C,
                subg_C=subg_C,
                ablation_innerGNN=model_config.ablation_innerGNN,
                extend_r=model_config.extend_r,
            )
        
        # ema configs
        self.ema_model = ExponentialMovingAverage(self, decay=optimizer_config.ema_decay, device=self.device)
        

        # optimizer and scheduler configs
        self.automatic_optimization = False
        
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        self.force_ratio = optimizer_config.force_ratio
        self.gradient_clip_val = optimizer_config.gradient_clip_val
        self.warmup_end = False
        self.warmup_epoch = scheduler_config.warmup_epoch

        

    def forward(self, batch_data):
        return self.model(batch_data)
        
    def configure_optimizers(self):
        return configure_optimizers_(self)

    def training_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]

        pred_y = self(batch)
        y = batch.y.unsqueeze(-1)
        loss = self.cal_bce_loss(pred_y, y)
        
        # log multiple loss
        self.log('train_loss/train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True, batch_size=batch_size)
        
            
        # optimize manually
        if loss == loss:
            self.opt_step(loss)
        else:
            print("nan loss")
        
        # log
        self.log("learning_rate/lr_rate_AdamW", self.optimizers().param_groups[0]["lr"], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]

        pred_y = self(batch) # (B, 1)
        y = batch.y.unsqueeze(-1) # (B, 1)
        
        acc = self.cal_acc(pred_y, y)
        loss = self.cal_bce_loss(pred_y, y)
        

        # log multiple loss
        self.log('val_loss/acc', acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('val_loss/val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)

        return acc
        
        
    def test_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]

        pred_y = self(batch) # (B, 1)
        y = batch.y.unsqueeze(-1) # (B, 1)
        
        acc = self.cal_acc(pred_y, y)
        
        self.log(f'test_loss/acc', acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)

        return acc

        
    def on_validation_epoch_end(self):
        
        if self.warmup_end:
            for scheduler in self.lr_schedulers():
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.trainer.callback_metrics[self.optimizer_config.monitor])
                    
        
    def cal_acc(self, pred_y_raw, y):
        pred_y_sig = torch.nn.functional.sigmoid(pred_y_raw)
        pred_type = pred_y_sig > 0.5 # (B, 1)
        y_type = y==1.0 # (B, 1)
        
        assert pred_type.shape == y_type.shape
        
        acc = (pred_type == y_type).sum().float() / pred_type.shape[0]
        
        return acc
    
    def cal_bce_loss(self, pred_y_raw, y):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y_raw, y)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema_model.update_parameters(self) 
        

    def on_train_epoch_end(self):
        
        if self.warmup_end == False:
            if self.current_epoch >= self.warmup_epoch:
                self.warmup_end = True
            
        for scheduler in self.lr_schedulers():
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()



    def on_validation_epoch_end(self):
        if self.warmup_end:
            for scheduler in self.lr_schedulers():
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.trainer.callback_metrics[self.optimizer_config.monitor])

    def opt_step(self, loss) -> None:
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        clip_grad_norm_(self.parameters(), self.gradient_clip_val)
        self.optimizers().step()