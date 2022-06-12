# Import libraries
from random import random, randint
import time

from utils.rich_logger import *



def find_validation_point(n_train, batch_size, val_repeat):
    n_train_batch = n_train // batch_size
    last_point = (n_train_batch + 1) if n_train % batch_size else n_train_batch
    return [last_point if item == val_repeat else ((n_train_batch//val_repeat) * item) for item in range(1, val_repeat + 1)]



def validate(epoch, val_round, val_repeat):
    n_val = 116

    valid_task_id = valid_progress.add_task(description = "Starting...", total = n_val, epoch = epoch, val_round = val_round)

    # Replace this with your own data loader. like: for data1, data2 in valid_loader:
    for _ in range(n_val):

        # Testing the algorithm with random data (Remove it)
        time.sleep(0.025)
        val_loss = random()
        step_loss = random()
        val_acc = randint(0, 100)
        
        update_progress(bar_type = 1, progress_bar = valid_progress, progress_id = valid_task_id, advance= 1)
    
    stop_progress(progress_bar = valid_progress, progress_id = valid_task_id, visible = False)
    result_progress(bar_type = 1, progress_bar = valid_progress, progress_id = valid_task_id, 
            val_loss = val_loss, step_loss = step_loss, 
            val_acc = val_acc, val_repeat = val_repeat)

    return val_loss, step_loss , val_acc 


