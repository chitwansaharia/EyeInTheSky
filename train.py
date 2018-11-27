import numpy as np
import parser
from tensorboardX import SummaryWriter
from utils import *
from comet_ml import Experiment
from sat_loader import SatelliteDataset
import sklearn
from sklearn import metrics
from models import unet
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None,
                    help="Name of the model to be trained")
parser.add_argument("--batch-size", type=int, default=10,
                    help="Batch size used while training/validating")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate for training (default: 1e-3)")
parser.add_argument("--epochs", type=int, default=1000,
                    help="number of epochs to be trained (default: 20)")
parser.add_argument("--max-grad-norm", type=float, default=10,
                    help="maximum gradient clipping norm")
parser.add_argument("--optimizer", default="adadelta",
                    help="type of optimizer to use for training")
parser.add_argument("--model-type", default="",
                    help="type of model to use")
parser.add_argument("--resume", action="store_true",
                    help="resume training of a saved model")
parser.add_argument("--num-channels", type=int, default=4,
                    help="number of channels of imput image to use (3 or 4)")
parser.add_argument("--crop-dim", type=int, default=256,
                    help="dimension of the cropped image")
parser.add_argument("--log-interval", type=int, default=10,
                    help="logging interval")
parser.add_argument("--decay-rate", type=float, default=0.96,
                    help="decay rate for learning rate")
parser.add_argument("--decay-step", type=int, default=2,
                    help="decay step for learning rate")

args = parser.parse_args()
print(args)

experiment = Experiment(api_key="7bEW5h9UoEOpQQyNLpt36lY66", project_name=args.model)
experiment.log_multiple_params({
                                "batch-size" : args.batch_size,
                                "model" : args.model,
                                "lr" : args.lr,
                                "epochs" : args.epochs,
                                "max-grad-norm" : args.max_grad_norm,
                                "optimizer" : args.optimizer,
                                "model-type" : args.model_type,
                                "num-channels" : args.num_channels
                                })

def run_epoch(model, epoch, data_loader, device, mode="train", writer=None):
    # Define the iterator for the given mode

    if mode == "train":
        print("Training")
    else:
        model.eval()
        print("Validating")

    total_loss = 0
    total_grad_norm = 0
    total_score = 0
    i = 0

    for image, mask in data_loader:
        image, true_mask = image.to(device), mask.to(device).long()
        if mode is not "train":
            with torch.no_grad():
                out_mask = model(image)
        else:
            out_mask = model(image)
        loss = loss_criterion(out_mask, true_mask)

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            grad_norm = sum(p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            total_grad_norm += grad_norm.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_grad_norm += grad_norm.item()

        pred_labels = torch.argmax(out_mask, 1).view(-1)
        total_score += sklearn.metrics.cohen_kappa_score(pred_labels.cpu().numpy(), true_mask.view(-1).cpu().numpy())

        total_loss += loss.item()


        if i % args.log_interval == 0:
            tb_writer.add_scalar("{}_loss".format(mode), total_loss/(i+1))
            if mode == "train":
                tb_writer.add_scalar("{}_grad_norm".format(mode), total_grad_norm/(i+1))
            tb_writer.add_scalar("{}_kappa_score".format(mode), total_score/(i+1))
            print("Epoch {} | Batch {} |  Loss {} | Grad Norm {} | Kappa Score {}".format(epoch+1,
                                                                                          i+1,
                                                                                          total_loss/(i+1),
                                                                                          total_grad_norm/(i+1),
                                                                                          total_score/(i+1)))

        i += 1

    model.train()
    return total_loss/i, total_score/i

if __name__ == "__main__":
    args = parser.parse_args()
    num_classes = 9
    done_epochs = 0
    best_metric = 0

    train_x_dir = 'sat'
    train_y_dir = 'gt'
    val_x_dir = 'valid_sat'
    val_y_dir = 'valid_gt'
    root_dir = 'data'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Creating logging and saved models directory
    create_if_not_exists()

    model_path = "{}/{}.pt".format("saved_models", args.model)
    log_path = "{}/{}".format("tb_logs", args.model)

    tb_writer = SummaryWriter(log_path)

    # Dataset Loader
    train_loader = torch.utils.data.DataLoader(
                                SatelliteDataset(train_x_dir, train_y_dir, root_dir, args.crop_dim, args.num_channels),
                                batch_size = args.batch_size,
                                shuffle = True
                                )
    val_loader = torch.utils.data.DataLoader(
                                SatelliteDataset(val_x_dir, val_y_dir, root_dir, args.crop_dim, args.num_channels),
                                batch_size = args.batch_size,
                                shuffle = False
                                )

    model = unet.UNet(args.num_channels, num_classes)

    optimizer = torch.optim.Adam(model.parameters(),
                           lr=args.lr
                           )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=args.decay_rate)

    loss_criterion = nn.CrossEntropyLoss()

    # Resuming the training from a saved model
    if args.resume:
        if not os.path.exists(model_path):
            print("Saved model not found")
        else:
            # Load the pretrained model
            saved_data = torch.load(model_path)
            model.load_state_dict(saved_data["model_state_dict"])
            done_epochs = saved_data["epochs"]
            best_metric = saved_data["best_metric"]
            optimizer.load_state_dict(saved_data["opt_state_dict"])

    if torch.cuda.is_available():
        model.cuda()

    tb_writer = SummaryWriter(log_path)

    for epoch in range(done_epochs, done_epochs + args.epochs):
        experiment.log_current_epoch(epoch)
        lr_scheduler.step()

        # Training
        train_loss, train_metric = run_epoch(model, epoch, train_loader, device, "train", tb_writer)
        experiment.log_metric("train_loss",train_loss, step=epoch)
        experiment.log_metric("train_metric",train_metric, step=epoch)
        tb_writer.add_scalar("train_loss_per_epoch", train_loss, epoch)
        tb_writer.add_scalar("train_metric_per_epoch", train_metric, epoch)

        # Validating
        valid_loss, valid_metric = run_epoch(model, epoch, val_loader, device, "valid", tb_writer)
        experiment.log_metric("valid_loss",valid_loss, step=epoch)
        experiment.log_metric("valid_metric",valid_metric, step=epoch)
        tb_writer.add_scalar("valid_loss_per_epoch", valid_loss, epoch)
        tb_writer.add_scalar("valid_metric_per_epoch", valid_metric, epoch)

        if valid_metric > best_metric:
            best_metric = valid_metric
            torch.save({
            'epochs': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'best_metric' : best_metric
            }, model_path)





