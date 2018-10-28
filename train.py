import numpy as np
import parser
from tensorboardX import SummaryWriter
from utils import *
from comet_ml import Experiment

parser = ArgumentParser()
parser.add_argument("--model", default=None,
                    help="Name of the model to be trained")
parser.add_argument("--batch-size", default=5,
                    help="Batch size used while training/validating")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate for training (default: 1e-3)")
parser.add_argument("--epochs", type=int, default=20,
                    help="number of epochs to be trained (default: 20)")
parser.add_argument("--max-grad-norm", type=float, default=10,
                    help="maximum gradient clipping norm")
parser.add_argument("--optimizer", default="adadelta",
                    help="type of optimizer to use for training")
parser.add_argument("--model-type", default="",
                    help="type of model to use")
parser.add_argument("--resume", action="store_true",
                    help="resume training of a saved model")

print(args)

experiment = Experiment(api_key="7bEW5h9UoEOpQQyNLpt36lY66", project_name=args.model)
experiment.log_multiple_params({
                                "batch-size" : args.batch_size,
                                "model" : args.model,
                                "lr" : args.lr,
                                "epochs" : args.epochs,
                                "max-grad-norm" : args.max_grad_norm,
                                "optimizer" : args.optimizer,
                                "model-type" : args.model_type
                                })

def run_epoch(epoch, mode="train"):
    # Define the iterator for the given mode

    if mode == "train":
        print("Training")
    else:
        # Put model to eval mode
        print("Validating")





if __name__ == "__main__":
    args = parser.parse_args()
    done_epochs = 0

    # Creating logging and saved models directory
    create_if_not_exists()

    model_path = "{}/{}.pt".format("saved_models", args.model)
    log_path = "{}/{}".format("tb_logs", args.model)

    # Creating the model's logging folder
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Resuming the training from a saved model
    if args.resume:
        if not os.path.exists(model_path):
            print("Saved model not found")
        else:
            # Load the pretrained model

    writer = SummaryWriter(log_path)

    best_metric = 0
    for epoch in range(done_epochs, done_epochs + args.epochs):
        experiment.log_current_epoch(epoch)

        # Training
        experiment.log_metric("train_loss",train_loss, step=epoch)
        experiment.log_metric("train_metric",train_metric, step=epoch)


        # Validating
        experiment.log_metric("valid_loss",valid_loss, step=epoch)
        experiment.log_metric("valid_metric",valid_metric, step=epoch)

        if valid_metric > best_metric:
            best_metric = valid_metric
            # Save Model




