from __future__ import print_function, division
import os
import subprocess
import sys


def video_process(dir_path, dst_path, video_names):
    length = len(video_names)
    for idx, video_name in enumerate(video_names):
        print(idx+1, '/', length, video_name)
        dst_video_path = os.path.join(dst_path, video_name)
        if not os.path.exists(dst_video_path):
            os.mkdir(dst_video_path)

        video_file_path = os.path.join(dir_path, video_name + extension)
        if not os.path.exists(os.path.join(dst_video_path, 'image_000001.jpg')):
            subprocess.call('rm -r \"{}\"'.format(dst_video_path), shell=True)
            # print('remove {}'.format(dst_video_path))
            os.mkdir(dst_video_path)
        else:
            continue
        cmd = 'ffmpeg -i \"{}\" -threads 1 -q:v 0 \"{}/image_%06d.jpg\" -loglevel panic'.format(video_file_path, dst_video_path)
        # print(cmd)
        subprocess.call(cmd, shell=True)
        # print('\n')


dir_path = sys.argv[1]
dst_path = sys.argv[2]

extension = '.mp4' 
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
video_names = [video_name.split('.')[0] for video_name in os.listdir(dir_path)]
video_process(dir_path, dst_path, video_names)
