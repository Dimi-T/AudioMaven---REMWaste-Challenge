import os
import yt_dlp
from moviepy import VideoFileClip


def download_video(url, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    try:
        if "youtube.com" in url or "youtu.be" in url:

            with yt_dlp.YoutubeDL() as ydl:
                info_dict = ydl.extract_info(url, download=False)
                video_title = info_dict.get('title', 'video')

            output_path = os.path.join(output_dir, f"{video_title}.mp4")

            ydl_opts = {
                "format": "bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        else:
            raise ValueError("Please send a valid YouTube url.")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error downloading video: {e}")


def extract_audio(video_url, out_dir=os.getcwd()):

    mp4_path = download_video(video_url, out_dir)
    wav_path = os.path.splitext(mp4_path)[0] + ".wav"
    with VideoFileClip(mp4_path) as video:
        audio = video.audio
        audio.write_audiofile(wav_path, codec='pcm_s16le')
    os.remove(mp4_path)

    return wav_path