import pandas as pd
import json
from io import StringIO
import numpy as np
import bert_score
from sacrebleu import sentence_chrf
from utils.eval import *
from utils.utils import *
from utils.table_utils import *
import torch
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', 
        type=str,
        required=True,
        help='Input eval json'
    )
    parser.add_argument(
        '--output_path', 
        type=str,
        required=True,
        help='Output collated eval json'
    )
    args = parser.parse_args()

    with open(args.input_path, 'r') as f:
        eval_dict = json.load(f)

    macro_avgs = calculate_all_macro_avgs(eval_dict, eval_exact_match=True, eval_bert=True, eval_chrf=True)

    with open(args.output_path, 'w') as f:
        json.dump(macro_avgs, f, indent=6)