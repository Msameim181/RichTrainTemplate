# Import libraries
from random import random, randint
import time

from utils.rich_logger import *


def evaluate():
    n_eval = 200

    eval_task_id = eval_progress.add_task(description = "Starting...", total = n_eval)

    # Replace this with your own data loader. like: for data1, data2 in valid_loader:
    for _ in range(n_eval):

        # Testing the algorithm with random data (Remove it)
        time.sleep(0.025)
        eval_loss = random()
        step_loss = random()
        eval_acc = randint(0, 100)
        
        update_progress(bar_type = 2, progress_bar = eval_progress, progress_id = eval_task_id, 
                advance = 1 , eval_loss = eval_loss, eval_acc = eval_acc)
    
    stop_progress(progress_bar = eval_progress, progress_id = eval_task_id, visible = False)
    result_progress(bar_type = 2, progress_bar = eval_progress, progress_id = eval_task_id, 
            eval_loss = eval_loss, step_loss = step_loss, eval_acc = eval_acc)


    return eval_loss, step_loss , eval_acc 

