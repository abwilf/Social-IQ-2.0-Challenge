from alex_utils import *
import wandb
import os

import argparse
import pandas as pd
import os
import json
import openai
import random
import webvtt
from tqdm import tqdm
import time
from joblib import Parallel, delayed

API_KEY = os.getenv('CHATGPT_API')
NUM_RETRIES = 5

def compute_correctness(df):
    correct = 0
    total = 0

    for _,row in tqdm(df.iterrows(),total=len(df), position=0, leave=True):
        if(row['result'] != None):
            if(row['answer_idx'] in [int(c) for c in row['result'] if c.isdigit()]):
                correct += 1
            total += 1

    return (1.0 * correct/total)


def chatGPT(m):
    for i in range(NUM_RETRIES):
        try:
            res=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=m,
            )
            return res.choices[0].message.content
        except Exception as e:
            time.sleep(30)
            res=None
            print("Error:", e)


def computeChatGPT(data):
    
    i, row, transcript, classes, api_key = data
    openai.api_key = api_key
 
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

        final = {"role": "user", "content": message},

        reply = chatGPT(final)
        
        return reply
        
    except Exception as e:
        print(e)


        
def main():
    
    global args
    
    arg_defaults = [
        ('--_tags', str, 'debug'), # NOTE: required if you use deploy_sweeps. please do not remove the _. Use 'debug' if you don't want wandb to sync.
        
        ('--seed', int, 42),
        ('--wdb_project', str, ''), # defaults to chdir, but will be overwritten if called as part of a sweep
        ('--wdb_entity', str, 'socialiq'),

        ('--transcript_path', str, ''),
        ('--num_classes', int, 2),
        ('--transcript', bool, True),
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


    # get original SIQ data 
    df = pd.read_json(args.dataset_path,lines=True)


    # Set the number of cores to use
    num_cores = 4

    # Execute the function on the data in parallel using joblib
    results = Parallel(n_jobs=num_cores)(delayed(computeChatGPT)((i,row,args.transcript, args.num_classes, API_KEY)) for i,row in df.iterrows())


    # Save results 
    df["result"] = results
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_json(os.path.join(args.output_dir, 'qa_chatgpt.json'),orient='records',lines=True)


    # compute accuracy
    accuracy = compute_correctness(df)
    print(accuracy)
    
    if 'debug' not in args._tags:
        wandb.log({'train_loss': 0.1})


if __name__=='__main__':
    main()