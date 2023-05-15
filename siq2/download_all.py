# Given trims and original split, downloads siq2 dataset audio (mp3 and wav), video, transcripts, and frames.
# author: Sheryl Mathew
import json
import os
import youtube_utils
import random
from moviepy.editor import *
import subprocess
import webvtt
import datetime
import tempfile
from tqdm import tqdm
import argparse
import pathlib
join = os.path.join

def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Command-line arguments')
    parser.add_argument('--data_dir', type=str, default='siq2', help='Data directory')
    parser.add_argument('--no_frames', action='store_true', default=False, help='Disable frames')
    args = parser.parse_args()
    return args

args = parse_args()
trims_path = join(args.data_dir, 'trims.json')
original_split_path = join(args.data_dir, 'original_split.json')
current_split_path = join(args.data_dir, 'current_split.json')
transcript_path = join(args.data_dir, 'transcript')
video_path = join(args.data_dir, 'video')
mp3_path = join(args.data_dir, 'audio', 'mp3')
wav_path = join(args.data_dir, 'audio', 'wav')

mkdirp(transcript_path)
mkdirp(video_path)
mkdirp(mp3_path)
mkdirp(wav_path)

with open(trims_path) as f:
    trims = json.load(f)

with open(original_split_path) as f:
    split = json.load(f)

trim_not_found = []
videos_not_found = []
transcript_not_found = []

def find_active_videos(ids):
    found_vids = []
    for id in tqdm(ids, desc='Downloading video data for Social-IQ 2.0 Dataset'):
        if id not in trims:
            trim_not_found.append(id)
            continue

        trim_time = trims[id]

        try:
            video_found = True
            transcript_found = True
            
            temp_folder = tempfile.TemporaryDirectory()

            # download full video from youtube if available
            full_video = youtube_utils.download_video(id, temp_folder.name, False)
            if full_video == None:
                video_found = False
                videos_not_found.append(id)
                continue

            # trim mp4
            if not os.path.exists(join(video_path, id + ".mp4")):
                clip = VideoFileClip(full_video)
                clip1 = clip.subclip(trim_time, 60+trim_time)
                clip1.write_videofile(join(video_path, id + ".mp4"),codec='libx264')

            # convert to mp3
            if not os.path.exists(join(mp3_path, id + ".mp3")):
                input_name = join(video_path, id + ".mp4")
                output_name = join(mp3_path, id + ".mp3")
                subprocess.call('ffmpeg -i {video} -ar 22050 -ac 1 {out_name}'.format(video=input_name, out_name=output_name), shell=True)
            
            # convert mp3 to wav
            if not os.path.exists(join(wav_path, id + ".wav")):
                input_name = join(mp3_path, id + ".mp3")
                output_name = join(wav_path, id + ".wav")
                subprocess.call('ffmpeg -i {video} -ar 22050 -ac 1 {out_name}'.format(video=input_name, out_name=output_name), shell=True)

            # download transcript
            if not os.path.exists(join(transcript_path, id + ".vtt")):
                transcript, info = youtube_utils.download_transcript(id, temp_folder.name)
                try:
                    transcript = webvtt.read(join(temp_folder.name, id + ".v2.en.vtt"))
                except:
                    try:
                        transcript = webvtt.read(join(temp_folder.name, id + ".v2.en-manual.vtt"))
                    except:
                        transcript_not_found.append(id)
                        continue
                trimmed_transcript = webvtt.WebVTT()

                # adjust start/end times to start at 0 instead of the trimmed time
                for caption in transcript:
                    start_time = datetime.datetime.strptime(caption.start, '%H:%M:%S.%f')
                    end_time = datetime.datetime.strptime(caption.end, '%H:%M:%S.%f')
                    start_time = start_time.replace(year=2000,month=1,day=1)
                    end_time = end_time.replace(year=2000,month=1,day=1)
                    
                    sec = int(str(trim_time).split(".")[0])
                    ms = int(str(trim_time).split(".")[1])*1000 # convert millisecond to microsecond
                    min = 0
                    if sec >= 60:
                        min = sec // 60
                        sec = sec % 60
                    trim_time_start = datetime.datetime.combine(datetime.date(year=2000,month=1,day=1), datetime.time(minute=min, second=sec, microsecond=ms))
                    trim_time_end = datetime.datetime.combine(datetime.date(year=2000,month=1,day=1), datetime.time(minute=min, second=sec, microsecond=ms)) + datetime.timedelta(days=0, minutes=0, seconds=60)

                    # remove extra time stamps between <>
                    caption.text = caption.text.replace("<.*?>", "")
                    if start_time >= trim_time_start and end_time <= trim_time_end:
                        str_start = str(start_time - trim_time_start)
                        str_end = str(end_time - trim_time_start)
                        if len(str_start.split(":")[0]) == 1:
                            str_start = "0" + str_start
                        if len(str_end.split(":")[0]) == 1:
                            str_end = "0" + str_end
                        if "." not in str_start:
                            str_start += ".000"
                        if "." not in str_end:
                            str_end += ".000"
                        caption.start = str_start
                        caption.end = str_end
                        trimmed_transcript.captions.append(caption)
                trimmed_transcript.save(join(transcript_path, id + ".vtt"))
                temp_folder.cleanup()

            # download frames
            frame_dir = join(args.data_dir, "frames")
            if not os.path.exists(os.path.dirname(id)) and not args.no_frames:
                os.makedirs(join(frame_dir, id), exist_ok=True)
                vid_path = join(video_path, id + ".mp4")
                output = join(frame_dir, id, id+"_%03d.jpg")
                subprocess.call('ffmpeg -i {video} -r 3 -q:v 1 {out_name}'.format(video=vid_path, out_name=output), shell=True)
            
            if video_found and transcript_found:
                found_vids.append(id)
        except:
            temp_folder.cleanup()
            continue

    return found_vids

yt_clips = split['subsets']['youtubeclips']
mv_clips = split['subsets']['movieclips']
car_clips = split['subsets']['car']
new_split = {
        "subsets":
            {    
                "youtubeclips": {"train": find_active_videos(yt_clips['train']), "val": find_active_videos(yt_clips['val']), "test": find_active_videos(yt_clips['test'])}, 
                "movieclips": {"train": find_active_videos(mv_clips['train']), "val": find_active_videos(mv_clips['val']), "test": find_active_videos(mv_clips['test'])}, 
                "car": {"train": find_active_videos(car_clips['train']), "val": find_active_videos(car_clips['val']), "test": find_active_videos(car_clips['test'])}
            }
        }

# put updated split into json file
with open(current_split_path, "w") as f:
    f.write(json.dumps(new_split))

if trim_not_found != []:
    print("could not find trims for:", trim_not_found)
if videos_not_found != []:
    print("could not download videos for:", videos_not_found)
if transcript_not_found != []:
    print("could not download transcripts for:", transcript_not_found)
