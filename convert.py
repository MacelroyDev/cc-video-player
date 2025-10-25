from PIL import Image
import numpy as np
from scipy.signal import resample, butter, lfilter
import cv2,nfp,dfpwm,ffmpeg,sys,os,io
import soundfile as sf

TARGET_SAMPLERATE = 48000 # Target sample rate for computer craft

CUTOFF_FREQ = 4000 
FILTER_ORDER = 5

def butter_lowpass_filter(data, cutoff, fs, order):
    """Applies a Butterworth Low-Pass Filter to the audio data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert.py <input video> <resolution> <fps>")
        exit(1)

    video = sys.argv[1]
    resolution = tuple(map(int, sys.argv[2].split("x")))
    fps = int(sys.argv[3])

    cap = cv2.VideoCapture(video)

    # Convert audio to dfpwm
    data = ffmpeg.input(video).output("pipe:1", format='wav').run(capture_stdout=True, capture_stderr=True)[0]
    data, original_samplerate = sf.read(io.BytesIO(data)) #renamed to original_samplerate for clarity

    # Convert to mono
    if data.ndim > 1:
        # Average audio channels to get mono
        data = data.mean(axis=1)

    # Resample audio data to target rate
    if original_samplerate != TARGET_SAMPLERATE:
        num_orignal_samples = len(data)
        num_target_samples = int(num_orignal_samples*(TARGET_SAMPLERATE/original_samplerate))

        data = resample(data, num_target_samples)

        samplerate_to_use = TARGET_SAMPLERATE
    else:
        samplerate_to_use = TARGET_SAMPLERATE


    data = butter_lowpass_filter(data, CUTOFF_FREQ, samplerate_to_use, FILTER_ORDER)

    data = np.clip(data, -1.0,1.0)

    audio_data = dfpwm.convert_audio(data, samplerate_to_use)
    with open('audio.dfpwm', 'wb') as f:
        f.write(audio_data.getvalue())

    # Calculate frame skip
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(original_fps / fps)

    # Read frames
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % frame_skip == 0:
                frames.append(frame)
            frame_count += 1
        else:
            break

    # Resize frames
    frames = [cv2.resize(frame, resolution) for frame in frames]

    # Convert frames to nfp
    nfp_frames = []
    for frame in frames:
        nfp_frames.append(nfp.img_to_nfp(Image.fromarray(frame)))

    # Write nfp frames to file
    with open("video.nfv", "wt") as f:
        f.write(f"{resolution[0]} {resolution[1]} {fps}\n")
        f.write("\n".join(nfp_frames))