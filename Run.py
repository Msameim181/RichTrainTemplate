# Import libraries
from rich.live import Live

from utils.rich_logger import *
from Train import *
from Evaluation import *


if __name__=='__main__':
    with Live(progress_group()):


        trainer()


        evaluate()