import wandb

def init_wandb(project_name, run_name):
    wandb.init(project=project_name, name=run_name, config={
        "input_size": (256, 256),
        "epochs": 25,
        "batch_size": 8,
        "loss": "BCE + Dice",
        "optimizer": "Adam",
    })
