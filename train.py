import numpy as np
import parser
from tensorboardX import SummaryWriter
from utils import *
from sat_loader import SatelliteDataset
import sklearn
from sklearn import metrics
from models import unet
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None,
                    help="Name of the model to be trained")
parser.add_argument("--data_dir", default=os.getcwd(),
					help="directory to store logging and model data")
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
parser.add_argument("--class-number", type=int, default=1,
                    help="class number to train a model for")
parser.add_argument("--train-per-class", action="store_true",
                    help="train a network for a single class")
parser.add_argument("--seed", type=int, default=0,
                    help="setting random seed for the experiment")
parser.add_argument("--contrast-enhance", action="store_true",
                    help="use contrast enhancement on the image")
parser.add_argument("--gaussian-blur", action="store_true",
                    help="perform gaussian blur on the one-hot mask")


args = parser.parse_args()
print(args)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_class_weight(class_):
    class_ratio = class_ratios[class_]
    class_weight = [(1+epsilon)/(class_ratio+epsilon), (1+epsilon)/(1-class_ratio+epsilon)]
    class_weight = [weight/sum(class_weight) for weight in class_weight]
    return class_weight[0]

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
    total_accuracy = 0
    total_preds = 0

    i = 0

    for image, mask, label in data_loader:
        image, true_mask, true_label = image.to(device), mask.to(device), label.to(device)

        if not args.gaussian_blur:
            true_mask = true_mask.long()
        else:
            true_mask = true_mask.float()

        if args.train_per_class:
            true_mask = (true_mask == args.class_number).long()

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

        score = sklearn.metrics.cohen_kappa_score(pred_labels.detach().cpu().numpy(), true_label.long().view(-1).cpu().numpy())

        total_accuracy += (pred_labels == true_label.long().view(-1)).sum().item()
        total_preds += pred_labels.numel()

        if not math.isnan(score):
            total_score += score

        total_loss += loss.item()

        if i % args.log_interval == 0:
            tb_writer.add_scalar("{}_loss".format(mode), total_loss/(i+1))
            if mode == "train":
                tb_writer.add_scalar("{}_grad_norm".format(mode), total_grad_norm/(i+1))
            tb_writer.add_scalar("{}_kappa_score".format(mode), total_score/(i+1))
            print("Epoch {} | Batch {} |  Loss {} | Grad Norm {} | Kappa Score {} | Accuracy {}".format(epoch+1,
                                                                                          i+1,
                                                                                          total_loss/(i+1),
                                                                                          total_grad_norm/(i+1),
                                                                                          total_score/(i+1),
                                                                                          total_accuracy/total_preds))
        i += 1

    model.train()
    return total_loss/i, total_score/i

if __name__ == "__main__":
    args = parser.parse_args()
    seed(args.seed)
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
    create_if_not_exists(args)

    model_path = "{}.pt".format(os.path.join(args.data_dir, 'saved_models', args.model))
    log_path = os.path.join(args.data_dir, 'tb_logs', args.model)

    tb_writer = SummaryWriter(log_path)

    # Dataset Loader
    train_loader = torch.utils.data.DataLoader(
                                SatelliteDataset(train_x_dir, train_y_dir, root_dir, args.crop_dim, args.num_channels, args.contrast_enhance, args.gaussian_blur),
                                batch_size = args.batch_size,
                                shuffle = True
                                )
    val_loader = torch.utils.data.DataLoader(
                                SatelliteDataset(val_x_dir, val_y_dir, root_dir, args.crop_dim, args.num_channels, args.contrast_enhance, args.gaussian_blur),
                                batch_size = args.batch_size,
                                shuffle = False
                                )

    if args.train_per_class:
        loss_criterion = nn.CrossEntropyLoss(torch.tensor([0.2, 0.8]).to(device))
        model = unet.UNet(args.num_channels, 2)
    elif not args.gaussian_blur:
        model = unet.UNet(args.num_channels, num_classes)
        # weights = [1]*num_classes
        # weights[2] = 4
        # weights[7] = 4
        # weights[1] = 0.4
        # print(weights)
        # loss_criterion = nn.CrossEntropyLoss(weight = torch.tensor(weights).to(device))
        loss_criterion = nn.CrossEntropyLoss()
    else:
        model = unet.UNet(args.num_channels, num_classes)
        loss_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                           lr=args.lr
                           )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=args.decay_rate)

    # Resuming the training from a saved model
    if args.resume:
        if not os.path.exists(model_path):
            print("Saved model not found")
            sys.exit()
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
        lr_scheduler.step()

        # Training
        train_loss, train_metric = run_epoch(model, epoch, train_loader, device, "train", tb_writer)
        tb_writer.add_scalar("train_loss_per_epoch", train_loss, epoch)
        tb_writer.add_scalar("train_metric_per_epoch", train_metric, epoch)

        # Validating
        valid_loss, valid_metric = run_epoch(model, epoch, val_loader, device, "valid", tb_writer)
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





