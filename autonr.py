from scipy.io import wavfile
import noisereduce as nr
import numpy as np
from noisereduce.utils import int16_to_float32, float32_to_int16
from librosa.feature import rms
import argparse


#
# Find longest section of audio where the energy is below mean - thresh * stddev
#
def find_some_background_noise(rate, y):
    thresh = 0.0  # A window consistently
    hop_length = 256
    frame_length = 2048
    energy = rms(y=y, frame_length=frame_length, hop_length=hop_length)
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    # find longest sequence of background noise
    longest_noise_n_frames = 0
    current_noise_n_frames = 0
    longest_noise_start_frame = -1
    current_noise_start_frame = -1

    # print(energy.T, mean_energy, std_energy)
    is_noise = energy < mean_energy - thresh * std_energy
    is_noise = is_noise[0]
    # print(is_noise)

    hop_length_secs = hop_length / rate
    frame_length_secs = frame_length / rate
    for i in range(len(is_noise)):
        # print(f'Frame at {hop_length_secs * i: 6.4}s')
        current_frame_is_noise = is_noise[i]
        if current_noise_n_frames > 0:  # Finding  noise
            if current_frame_is_noise:
                current_noise_n_frames += 1
            else:  # end of noise frames, reset counter and save start frame and length is > longest
                # print(f'Found end of noise at {i}')

                if current_noise_n_frames > longest_noise_n_frames:
                    print(
                        f'Found noise of length {current_noise_n_frames} {current_noise_n_frames * hop_length_secs: 3.3} at {hop_length_secs * current_noise_start_frame: 3.3}s')

                    longest_noise_n_frames = current_noise_n_frames
                    longest_noise_start_frame = current_noise_start_frame

                current_noise_n_frames = 0

        else:  # Not finding noise
            if current_frame_is_noise:
                # print(f'Found noise at {i}')
                current_noise_start_frame = i
                current_noise_n_frames = 1

    pad_frames = 0
    return y[(longest_noise_start_frame + pad_frames) * hop_length:(longest_noise_start_frame + longest_noise_n_frames - pad_frames) * hop_length]


def load_wav(file_name):
    return wavfile.read(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectral gating noise reduction')
    parser.add_argument('--input_file', '-i', type=str, help='Noisy File', required=True)
    parser.add_argument('--output_file', '-o', type=str, help='Denoised file', required=True)
    parser.add_argument('--threshold', '-t', type=float, help='Threshold', default=1.5)

    args = parser.parse_args()

    r, y = load_wav(args.input_file)
    y = int16_to_float32(y)
    noise = find_some_background_noise(r, y)
    wavfile.write(f'{args.output_file}.noise_section.wav', r, noise)
    reduced_noise = nr.reduce_noise(
        n_std_thresh=args.threshold,
        n_fft=1024,
        win_length=1024,
        audio_clip=y,
        noise_clip=noise,
        use_tensorflow=False,
        verbose=False,
    )
    wavfile.write(args.output_file, r, np.array(reduced_noise))
