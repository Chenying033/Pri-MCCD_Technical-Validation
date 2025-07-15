# coding=utf-8
import os
import cv2
import shutil
import time
import numpy as np
from moviepy.editor import AudioFileClip
from funasr import AutoModel


VIDEO_PATH = '../data'
EXTRACT_FOLDER = '../data/frames' 
EXTRACT_FREQUENCY = 10  


# 主操作
def extract_frames(video_path):
    
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    frames_list = []
    # 循环遍历视频中的所有帧
    while True:
        
        _, frame = video.read()
        if frame is None:
            break
       
        if frame_count % EXTRACT_FREQUENCY == 0:
            print(frame.shape)
            frames_list.append(frame)
        frame_count += 1  

    frames = np.stack(frames_list, axis=0)
    print(frames.shape)
    return frames


def extract_audio(video_path):
    my_audio_clip = AudioFileClip(video_path)
    my_audio_clip.write_audiofile(video_path.replace(".mp4", ".wav"))


def extract_audio_feat(wav_file):
    model = AutoModel(model="../asr_model")
    res = model.generate(wav_file, output_dir="../outputs", granularity="frame", extract_embedding=True)
    feats = res[0]['feats']

    np.savez(wav_file.replace('.wav', '.npz'), **{'feat': feats})
    # print(feats.shape)


def main(data_dir):
    # extract_frames(VIDEO_PATH)

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files if f.endswith('.mp4')]
    for file in files:
        extract_audio(file)

        extract_audio_feat(file.replace('.mp4', '.wav'))


if __name__ == '__main__':
    main('../data/train')
    main('../data/valid')
    main('../data/test')

    # feat = np.load('data/video_1.npz')['feat']
    # print(feat.shape)

