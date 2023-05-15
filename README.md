# The Social-IQ 2.0 Challenge
The Social-IQ 2.0 Challenge will be co-hosted with the Artificial Social Intelligence Workshop at ICCV '23, and will comprise a challenge with paper submissions on the Social-IQ 2.0 task. The dataset contains over 1,000 videos, annotated with over 6,000 4-class multiple choice questions. State-of-the-Art model performances on this task are low, with ChatGPT, FLAN-T5-XL, and RoBERTa scoring in the high 50%'s accuracy. There are $1,200 in total prizes.

If you plan to participate, please fill out this [form](https://forms.gle/ZVTAvNunBQUa9ncJ6) with your email address so we can keep you up to date with any relevant updates about the challenge.

## Important Dates 📅
- [x] Challenge is released 📅  May 15
- [ ] Challenge and paper submission sites open, test set released 📅  mid-June
- [ ] Paper submissions and final challenge submissions due 📅 July 21
- [ ] Acceptance decisions issued 📅 August 4
- [ ] Camera ready paper due 📅 August 11
- [ ] ICCV workshop 📅 one of the days in October 4-6, exact date TBA

## The Challenge 🏔️
To participate in the challenge, you will need to:
1. Submit your model's predictions on the test set to the challenge submission site (coming soon)
2. Submit your paper to the paper submission site (also coming soon)

You can use the validation data to evaluate your model's performance before submitting. Once you have found your best performing model, you will submit it to the online submission site to see your performance on the held-out test set. The test set and the submission site will be posted in mid-June. **You can only submit five times**, so use your submissions wisely!

You may use pretrained models but **may not use any additional data not included in the Social-IQ 2.0 dataset release**. In each focus, we ask that you do not do any additional training or adaptation on the test set. Please only use the validation set for evaluating, and submit predictions on the test set.

## Paper Submissions 📤
For your submission to be considered for the challenge, **please submit a paper describing your approach, and share your github with the challenge organizers so they can ensure your results are reproducible.** Along with your paper and challenge submission, please let us know which focus (described below) you would like your paper to be considered for using this link TODO. We will release the paper submission site in June. Please format your papers according to [ICCV's submission guidelines](https://iccv2023.thecvf.com/submission.guidelines-361600-2-20-16.php). Papers can be up to 6 pages (with no cap on additional pages for references and appendices).

## Paper Decisions 🎉
Paper decisions will be based on four criteria:
1. the contribution / novelty of the approach
2. the reproducibility of the results (if deemed not reproducible by our program committee members, the paper will be rejected)
3. the clarity of the writing
4. the challenge performance (*with respect to the focus area*)

**Even if your approach does not have high performance, it can still be accepted if it is interesting, clearly described, and makes a valuable contribution to one of the three focus areas!** We are curious how you will compose models few-shot, what kinds of features you will extract and how you will reason and fuse over them, and how you will learn robust and generalizable representations.

In addition, we believe that negative results are important information for the community. In your submission, we would be interested in seeing as well what you tried that did not work, so future research can learn from your experiments. 

## Awards 🏆 and Prizes 💰
There will be over $1,200 in total prizes. We will give the following awards, and depending on the submissions, may create additional awards as well.
- **Best Paper Award**: assigned based on the criteria above
- **Best Paper Honorable Mention**
- **Challenge Winner**: goes to the paper with the highest performance on the held out test set through the submission portal

## Focuses 🔍
In order to encourage diverse, innovative research into solving the Social-IQ 2.0 task, you will also describe which "focus" area your paper falls within.

### Few-Shot Focus 🎯
This focus is intended to encourage research into composing foundation models in a zero- or few-shot way, motivated by works such as [Socratic Models](https://socraticmodels.github.io/). As foundation models improve, we believe this is a very interesting area of research to explore in challenging domains such as artificial social intelligence. This focus is intended to be accessible to anyone with the ability to compose models zero-shot. It is ok to use API-accessed models such as ChatGPT and Bard.

For this focus, we encourage you to compose models zero- or few-shot *without* training on Social-IQ 2.0 data.

### Fusion and Reasoning Focus 🧠
In this focus, we hope to foster approaches that make creative use of latent feature representations with relatively small amounts of training (e.g. a few hours on a single GPU).  The goal here is to enable researchers to integrate features that may not be immediately composable zero shot (e.g. latent vector representations instead of text). We encourage you to think about *which features you can extract* and *how you can fuse or reason over those features* cleverly, using relatively small amounts of training / fusion layers to combine feature representations. For this focus, we expect that the number of *learnable parameters* you train will be relatively low (<< 1e+9). An interesting paper to look at for inspiration on this focus is the [Neural State Machine](https://arxiv.org/pdf/1907.03950.pdf).

### Representation Learning Focus 🎛️
While research into few-shot and low-resource settings is important, we also want to encourage novel training procedures that scale to high resource amounts. For this focus, you may use as much compute as you would like, provided (as with all challenge submissions) that you only use data in the Social-IQ 2.0 dataset release for your training. Please reach out to us if you intend to use specialized hardware such as TPU's, so we can ensure your code is reproducible. If you would like to use TPU's, we recommend applying for [this program from Google](https://sites.research.google/trc/about/); their generous support allowed us to run the MERLOT-Reserve baselines for this challenge. You may choose to research a purely self-supervised setting (e.g. continued pretraining + partially frozen finetuning, as in [this paper](https://arxiv.org/pdf/2208.01036.pdf)) or in a supervised setting such as MERLOT's finetuning results.

## The Social-IQ 2.0 Dataset
We provide scripts to download the videos and question / answer files that comprise the task. Videos are drawn from youtube videos in three categories: general youtube videos containing socially rich situations, youtube clips from movies, and clips containing socially rich situations in passenger vehicles.

The `siq2` folder looks like this:
```
siq2
├── audio
│   ├── mp3 # will contain mp3 files
│   └── wav # will contain wav files
├── qa # contains question / answer labelled data; will contain unlabelled qa_test.json file when submission site is released
│   ├── qa_train.json 
│   └── qa_val.json
├── split.json # contains which video ids are in train/val/test splits for the different subsets: youtubeclips, movieclips, car clips
├── transcript # will contain .vtt transcript files
├── trims.json # contains start times for the videos; videos are 60 seconds long from that start second
└── video # will contain .mp4 video files
```

To download the data, first install all required dependencies:
```
conda create -n siq2 python=3.8 -y && conda activate siq2
pip install -r requirements.txt
sudo apt update && sudo apt install ffmpeg # or however you need to install ffmpeg; varies per machine
```

Then, run the following to download the dataset
```
python download_all.py # this takes about 4 hours to download and 60GB to store the whole dataset. This goes down to 30GB if you run with flag --no_frames
```
This will update `siq2/current_split.json`, which will describe the videos in the train/val/test splits. There is also `siq2/original_split.json`, which contains all the video ID's we have questions and answers for, regardless of whether they're available at the moment on youtube or not.

**A finer point about the dataset**: Because you will download videos from youtube directly, the set of videos that constitute the dataset may change slightly between the release and conclusion of the challenge. **We will treat the "final" dataset as the set of videos downloaded from youtube one week before the challenge concludes**. If you download the dataset now and report your results later, that is ok – we will simply discard predictions made on test set videos that are no longer available when we determine final testing accuracies, and you will not be penalized for training or validating on videos that are no longer available by the end of the challenge.

