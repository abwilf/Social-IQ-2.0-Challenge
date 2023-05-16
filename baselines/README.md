## How to run the baselines on any qa_test/train/val.json file: 

1. You must need a ChatGPT key to use the ChatGPT baseline. Please set an environment variable CHATGPT_API=your_api_key. 

2. Update the sweep defined in the yml file with your own file paths:

```
program: chatgpt_baseline.py 
method: grid
parameters:
  transcript_path:
      value: /baselines/transcript # TODO: update with the path to the video transcripts
  dataset_path:
      value: /socialiq_data/final_matchings/five_class/qa_test.json # TODO: update with the path of the input json file 
  output_dir:
      value: /socialiq_data/final_matchings/five_class/chatgpt_baseline # TODO: update with the directory of where output json file will be
  transcript:
      value: True # TODO: update with whether we are using transcripts or not (video text context or no context)
  num_classes:
      value: 4 # TODO: update with the number of classes (number of answers for each question) 
  seed:
    value: 42
  wdb_entity:
    value: socialiq-s2023
  _tags:
    value: debug
```

3. Create the sweep
```
wandb sweep chatgpt.yml
```
You'll see an output like this:

```
wandb: Creating sweep from: chatgpt.yml
wandb: Creating sweep with ID: gjv2phc4
wandb: View sweep at: https://wandb.ai/socialiq-s2023/uncategorized/sweeps/gjv2phc4
wandb: Run sweep agent with: wandb agent socialiq-s2023/uncategorized/gjv2phc4
```

Run the agent command in the terminal and click on the link to take you to the dashboard.

4. The accuracy of the model will be printed out onto the terminal

## MERLOT Baseline
See MERLOT repo at https://github.com/sherylm77/merlot_reserve/tree/main for instructions.
