import os

def create_if_not_exists():
    if not os.path.exists("tb_logs"):
        os.mkdir("tb_logs")
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")