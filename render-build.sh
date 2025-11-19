#!/usr/bin/env bash
set -x

# Download static ffmpeg build
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg.tar.xz
tar -xf ffmpeg.tar.xz
mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/
mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/
chmod +x /usr/local/bin/ffmpeg
chmod +x /usr/local/bin/ffprobe

# Install python dependencies
pip install -r requirements.txt
