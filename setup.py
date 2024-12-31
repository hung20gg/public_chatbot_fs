import argparse
import os
import sys

from ETL.connector import setup_everything


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'no', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', default=False, type=str2bool)
    parser.add_argument('--reset_vector_db', default=False, type=str2bool)
    parser.add_argument('--openai', default=False, type=str2bool)
    parser.add_argument('--local', default=False, type=str2bool)
    parser.add_argument('--ignore_rdb', default=False, type=str2bool)
    parser.add_argument('--source', default="cafef", type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    # args to dict
    args = vars(args)
    print(args)
    
    setup_everything(args)
    
    sys.exit(0)