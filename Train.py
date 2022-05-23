# Import libraries
import time
from datetime import datetime
from random import random

from utils.rich_logger import *
from Validation import *


def log_initiation():
    run_name = "Run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "CIFAR10"
    device = "GPU"
    net_name = "ResNet152"
    optimizer = "Adam"
    criterion = "CrossEntropy"
    net_in_channels = 3
    net_num_classes = 200
    epochs = 3
    learning_rate = 0.001
    batch_size = 8
    n_train = 1045
    val_percent = 0.1 * 100
    n_val = 116
    n_eval = 200
    save_checkpoint = True
    amp = False
    WandB_usage = True
    train_dataset_size = 1161
    test_dataset_size = 200
    notes = "This is a test note with 12, 'loop' False"
    rich_print(f'''\n[INFO]: Training Settings:
        DataSet:                <{dataset_name}>
        Device:                 "{device}"
        Model:                  <{net_name}>
        Image Channel:          {net_in_channels}
        Model Output (Ch):      {net_num_classes}
        Epochs:                 {epochs}
        Batch Size:             {batch_size}
        Learning Rate:          {learning_rate}
        Training Size:          {n_train}
        Validation Size:        {n_val}
        Validation %:           {val_percent}%
        Evaluation Size:        {n_eval}
        Checkpoints:            {save_checkpoint}
        Mixed Precision:        {amp}
        optimizer:              {optimizer}
        criterion:              {criterion}
        ------------------------------------------------------
        wandb:                  {WandB_usage}
        Tensorboard:            {True}
        Train Dataset Sample:   {train_dataset_size}
        Test Dataset Sample:    {test_dataset_size}
        ------------------------------------------------------
        Notes: {notes}''')
    rich_print(f'\n[INFO]: Start training as "{run_name}" ...\n')
    
    return


def trainer(
    epochs: int = 3,
    batch_size: int = 8,
    val_repeat: int = 2,
    **kwargs):
    ...

    # Initialize the logger (just something to show the configuration) Change it to your own logger
    log_initiation()


    # Data Size
    n_train = 1045
    

    # Counting all steps to the end of the training
    global_step = 0

    # total data size with batch size (You can remove it when you use dataloader from pytorch)
    total = n_train // batch_size
    total += 1 if n_train % batch_size else 0

    for epoch in range(epochs):
        # Create a progress bar
        train_task_id = train_progress.add_task(description="Starting...", total=n_train, epoch = epoch, epochs = epochs)

        # Counting the epoch information
        epoch_step = 0
        # epoch loss = sum(each step_loss) then divide by total epoch steps
        epoch_loss = 0
        # step loss = loss of each set of train and feeding network, 
        step_loss = 0

        # Replace this with your own data loader. like: for data1, data2 in train_loader:
        for _ in range(total):
            
            # Testing the algorithm with random data (Remove it)
            time.sleep(0.005)
            epoch_loss = random()
            step_loss = random()
            global_step += 1
            epoch_step += 1
            
            # Update the progress bar in each step
            update_progress(bar_type = 0, progress_bar = train_progress, progress_id = train_task_id, 
                    advance = batch_size, epoch_loss = epoch_loss, step_loss = step_loss)

            val_loss, val_step_loss, val_acc = validation(epoch, epoch_step, n_train, batch_size, val_repeat)

        stop_progress(progress_bar = train_progress, progress_id = train_task_id, visible = False)
        result_progress(bar_type = 0, progress_bar = train_progress, progress_id = train_task_id, 
                epoch_loss = epoch_loss, step_loss = step_loss)


def validation(epoch: int, epoch_step: int, n_train: int, batch_size: int, val_repeat: int = 2, **kwargs):
    # Evaluation round

    # Let's See if is it evaluation time or not
    val_point = find_validation_point(n_train, batch_size, val_repeat)

    if epoch_step in val_point:
        ...
        # rich_print(f"Validation Points: {val_point}")

        val_loss, step_loss , val_acc = validate(epoch, val_point.index(epoch_step), val_repeat, **kwargs)

        return val_loss, step_loss, val_acc

    return None, None, None
