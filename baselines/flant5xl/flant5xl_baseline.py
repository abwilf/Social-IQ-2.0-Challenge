from alex_utils import *
import wandb
import os

import argparse
import torch
import numpy as np
import json
import pandas as pd

import openai
import random
import webvtt
from tqdm import tqdm
import time
from joblib import Parallel, delayed
from transformers import T5Tokenizer, T5ForConditionalGeneration

import gc
import torch


def computeFlanT5XL(data, transcript=True, classes=2):
    
    i,row = data
    message = ""
    
    try: 
        if(transcript):
            captions = [caption.text for caption in webvtt.read(os.path.join(args.transcript_path, row["vid_name"] + '.vtt'))]
            transcript = '\n'.join(captions) 
            message = "Context:\"{}\"\n".format(transcript)
        
        message += "Question:\"{}\"\nAnswer Choices:\n".format(row["q"])
        
        for i in range(classes):
            name = "a" + str(i)
            message += "({})\"{}\"\n".format(i, row[name])
            
        message += "Respond with only the index of the more likely correct answer to the question:"
        
        input_ids = tokenizer(message, return_tensors="pt").input_ids.to("cuda")

        outputs = model.generate(input_ids)

        reply = tokenizer.decode(outputs[0])
        
        del input_ids
        torch.cuda.empty_cache()
        
        return reply
        
    except Exception as e:
        print(e)


def compute_correctness(df_new):
    correct = 0
    total = 0

    for _,row in tqdm(df_new.iterrows(),total=len(df_new), position=0, leave=True):
        if(row['result'] != None):
            if(row['answer_idx'] in [int(c) for c in row['result'] if c.isdigit()]):
                correct += 1
            total += 1

    return (1.0 * correct/total)
    
        
def main():
    
    torch.cuda.empty_cache()
    gc.collect()

    global args
    global tokenizer
    global model
    
    arg_defaults = [
        ('--_tags', str, 'debug'), # NOTE: required if you use deploy_sweeps. please do not remove the _. Use 'debug' if you don't want wandb to sync.
        
        ('--seed', int, 42),
        ('--wdb_project', str, ''), # defaults to chdir, but will be overwritten if called as part of a sweep
        ('--wdb_entity', str, 'socialiq'),

        ('--transcript_path', str, ''),
        ('--num_classes', int, 2),
        ('--transcript', bool, True),
        ('--cache_dir', str, ''),
        ('--dataset_path', str, ''),
        ('--output_dir', str, ''),
    ]
    
    parser = None
    args = process_defaults(arg_defaults, parser_in=parser)
    
    if 'debug' not in args._tags:
        wandb.init(
            project=args.wdb_project,
            entity=args.wdb_entity, 
            config=vars(args),
            tags=args._tags.split(','),
        )
        
    # get flan t5 xl model 
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir = args.cache_dir)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", cache_dir = args.cache_dir)
    
    # get data 
    matched_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path),lines=True)
    
    # Execute the flan t5 xl function 
    results = []
    for i,row in tqdm(matched_df.iterrows(),total=len(matched_df), position=0, leave=True):
        res = computeFlanT5XL((i,row), args.transcript, args.num_classes)
        results.append(res)
        
    
    # Updating the dataset
    matched_df["result"] = results
    os.makedirs(args.output_dir, exist_ok=True)
    matched_df.to_json(os.path.join(args.output_dir, 'qa_flant5xl.json'),orient='records',lines=True)
    
    
    # Compute correctness
    accuracy = compute_correctness(matched_df)
    print(accuracy)
    
    
    if 'debug' not in args._tags:
        wandb.log({'train_loss': 0.1})

    

if __name__=='__main__':
    main()