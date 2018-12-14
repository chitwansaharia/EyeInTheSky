import os

def create_if_not_exists(args):
    model_dir = os.path.join(args.data_dir, 'saved_models')
    log_dir = os.path.join(args.data_dir, 'tb_logs')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)