import random
import os
import wandb
from torch.utils.tensorboard import SummaryWriter
import torch
import glob

        
class Logger(object):
    def __init__(self, config):
        self.config = config
        self.logger = []
        if 'tensorboard' in config.logger:
            logdir = os.path.join(config.save_dir, 'logs')
            self.logger.append(TensorboardLogger(logdir))
        if 'wandb' in config.logger:
            self.logger.append(WandbLogger(project='video2sound', 
                                           name=config.save_dir.split('/')[-1], 
                                           directory=os.path.join(config.save_dir, 'wandb'),
                                           config=config))
            print('wandb.run.dir', wandb.run.dir)

    def log_training(self, reduced_loss_dict, test_loss_dict, learning_rate, duration, epoch):
        assert list(reduced_loss_dict.keys()) == list(test_loss_dict.keys())
        for logger in self.logger:
            logger.log_training(reduced_loss_dict, test_loss_dict, learning_rate, duration, epoch)
    
    def save_checkpoint(self, model, epoch):
        save_directory = self.config.save_dir
        lr = model.optimizers[0].param_groups[0]['lr']
        for name in model.model_names:
            filepath = os.path.join(save_directory, "checkpoint_{:0>6d}_{}.pt".format(epoch, name))
            print("Saving {}_module and optimizer state at epoch {} to {}".format(
                name, epoch, filepath))
            net = getattr(model, name)
            if torch.cuda.is_available():
                torch.save({"epoch": epoch,
                            "learning_rate": lr,
                            "optimizer_{}".format(name): net.module.cpu().state_dict()}, filepath)
                net.to(model.device)
            else:
                torch.save({"epoch": epoch,
                            "learning_rate": lr,
                            "optimizer_{}".format(name): net.cpu().state_dict()}, filepath)

            """delete old model"""
            model_list = glob.glob(os.path.join(save_directory, "checkpoint_*_*"))
            model_list.sort()
            # if len(model_list) > 3:
            #     for model_path in model_list[:-3]:
            #         cmd = "rm {}".format(model_path)
            #         print(cmd)
            #         os.system(cmd)
        return "_".join(model_list[-1].split("_")[-2:])


class WandbLogger(object):
    def __init__(self, project, name, directory, config):
        self.run = wandb.init(project=project, 
                              name=name, 
                              dir=directory,
                              config=config)
        
    def log_training(self, reduced_loss_dict, test_loss_dict, learning_rate, duration, epoch):
        log_dict = {}
        
        for loss_type, loss_value in reduced_loss_dict.items():
            log_dict[f"Train_Loss_{loss_type}"] = loss_value
            log_dict[f"Test_Loss_{loss_type}"] = test_loss_dict[loss_type]
        log_dict["Learning_Rate"] = learning_rate
        log_dict["duration"] = duration
        
        wandb.log(log_dict)


class TensorboardLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardLogger, self).__init__(logdir)

    def log_training(self, reduced_loss_dict, test_loss_dict, learning_rate, duration, epoch): # iteration
        # (model, reduced_loss, learning_rate, duration, iteration)
        for loss_type, loss_value in reduced_loss_dict.items():
            self.add_scalar(f"training.loss.{loss_type}", loss_value, epoch)
            self.add_scalar(f"testing.loss.{loss_type}", test_loss_dict[loss_type], epoch)
        self.add_scalar("learning_rate", learning_rate, epoch)
        self.add_scalar("duration", duration, epoch)

    