import librosa
import numpy as np
from scipy.signal import get_window
import scipy.fftpack as fft
import matplotlib.pyplot as plt

import scipy.io
from tqdm import tqdm
from python_speech_features import logfbank
import os

class TIDIGIT_Converter():
    """This class is for converting TIDIGIT data to mfsc features"""
    def __init__(self):
        self.mfsc_conv = MFSC_Converter()


    def convert_tidigit_own(self, PATH, item_name, sample_rate, timeframes, freq_bins):
        """
        Converts tidigit data with custom implementation
        :param PATH: path to the Tidigit data file
        :param item_name: Name of the variable to be loaded
        :param sample_rate: Sample rate for the data
        :param timeframes: Amount of timeframes in result
        :param freq_bins: Amount of frequency bins in result
        :return: Resulting mfsc features as numpy array
        """
        mat = scipy.io.loadmat(PATH)
        samples = mat[item_name]
        audios = [None] * len(samples)
        sample_rates = [sample_rate] * len(samples)
        for index, item in tqdm(enumerate(samples)):
            audios[index] = [value for sublist in item[0] for value in sublist]

        results = self.mfsc_conv.all_mfsc_own(audios, sample_rates, timeframes,
                           freq_bins)
        return results

    def convert_tidigit_lib(self, PATH, item_name, sample_rate, timeframes, freq_bins):
        """
                Converts tidigit data with existing implementation
                :param PATH: path to the Tidigit data file
                :param item_name: Name of the variable to be loaded
                :param sample_rate: Sample rate for the data
                :param timeframes: Amount of timeframes in result
                :param freq_bins: Amount of frequency bins in result
                :return: Resulting mfsc features as numpy array
                """
        mat = scipy.io.loadmat(PATH)
        samples = mat[item_name]
        audios = [None] * len(samples)
        sample_rates = [sample_rate] * len(samples)
        for index,item in tqdm(enumerate(samples)):
            audios[index] = item[0]
        results = self.mfsc_conv.all_mfsc_lib(audios, sample_rates, timeframes,
                                freq_bins)
        return results

class TIMIT_Converter():
    """This class is for converting TIMIT data to mfsc features"""

    #TODO: Further implement this class
    def __init__(self):
        self.mfsc_conv = MFSC_Converter()

    def convert_timit_own(self, PATH, timeframes, freq_bins):
        sound_list, s_rate_list = self.get_all_lists(PATH)
        results = self.mfsc_conv.all_mfsc_own(sound_list, s_rate_list, timeframes,freq_bins)
        return results

    def convert_timit_lib(self, PATH, timeframes, freq_bins):
        sound_list, s_rate_list = self.get_all_lists(PATH)
        results = self.mfsc_conv.all_mfsc_lib(sound_list, s_rate_list,
                                              timeframes, freq_bins)
        return results

    def get_all_lists(self, PATH):
        wav_names = []
        for root, dirs, files in os.walk(PATH):
            for file in files:
                if file.endswith(".wav"):
                    wav_names.append(os.path.join(root,file))
        return self.wav_to_list(wav_names)


    def wav_to_list(self, wav_files):
        """
        Converts wav format files to a list of sounds
        :param wav_files: A list of the wav filenames
        :return: list of sounds
        """
        audio_list = [None]*len(wav_files)#np.zeros(len(wav_files))
        s_rate_list = np.zeros(len(wav_files))
        for ind, filename in tqdm(enumerate(wav_files)):
            audio, sample_rate = librosa.load(filename)
            audio_list[ind] = audio
            s_rate_list[ind] = sample_rate
        return audio_list, s_rate_list

class MFSC_Converter():
    """This class is for converting audio data to mfsc features"""
    def __init__(self):
        pass

    def all_mfsc_own(self, sound_list, s_rate_list, timeframes, freq_bins):
        """
        Converts sounds to mfsc features based on a custom implementation
        :param sound_list: List of sound lists
        :param s_rate_list: list of sample rates
        :param timeframes: amount of timeframes in result
        :param freq_bins: amount of frequency bins in result
        :return: list of mfsc feature for each sound as numpy array
        """
        n_sounds = len(sound_list)
        all_results = n_sounds * [None]
        for i in tqdm(range(n_sounds)):
            all_results[i] = self.one_mfsc(sound_list[i], s_rate_list[i],
                                      timeframes, freq_bins)
        return np.array(all_results)

    def all_mfsc_lib(self, sound_list, s_rate_list, timeframes, freq_bins):
        """
        Converts sounds to mfsc features based on an existing implementation
        :param sound_list: List of sound lists
        :param s_rate_list: list of sample rates
        :param timeframes: amount of timeframes in result
        :param freq_bins: amount of frequency bins in result
        :return: list of mfsc feature for each sound as numpy array
        """
        n_sounds = len(sound_list)
        all_results = n_sounds * [None]
        for i in tqdm(range(n_sounds)):
            #Old version:
            duration = len(sound_list[i])/s_rate_list[i]
            winstep = duration/timeframes
            #Notes: winlen = winstep+2*overlap. Winstep = winstep-overlap
            #Possibly use identical calculations to making frames?
            #Winlen should be same as fft, winstep should be window length
            #NOTE: Change calculation of winstep too.
            all_results[i] = logfbank(sound_list[i] , samplerate = s_rate_list[i] ,
                                    winlen = 1.1*winstep, winstep = winstep, nfilt=freq_bins, nfft=1024)

            #New adaptable version
            #DOES NOT WORK FULLY, USES TRUNCATION DUE TO LOW NFFT
            #frame_len = int(len(sound_list[i]) / (timeframes - 1))
            #FFT_size = frame_len + (len(sound_list[i]) / (
            #        timeframes - 1))  # This is the new calculated version
            #FFT_size = int(FFT_size / 2) * 2
            #all_results[i] = logfbank(sound_list[i] , samplerate = s_rate_list[i] ,
            #                         winlen = FFT_size, winstep = frame_len, nfilt=freq_bins, nfft=FFT_size)
        return np.array(all_results)


    def one_mfsc(self, sound, s_rate, timeframes, freq_bins):
        """
        Converts one sound to mfsc features based on a custom implementation
        :param sound: sound to be converted, as a list
        :param s_rate: sample rate of the sound
        :param timeframes: amount of timeframes in result
        :param freq_bins: amount of frequency bins in result
        :return: list of mfsc feature for each sound as numpy array
        """
        sound = sound / np.max(np.abs(sound))  # First, we normalize
        frames, fft_len = self.create_frames(sound, timeframes)
        window = get_window("hann", fft_len, fftbins=True)
        windowed = frames * window
        windowedT = windowed.T
        fft_sound = np.empty((int(1 + fft_len // 2), windowedT.shape[1]),
                             dtype=np.complex64, order='F')

        for n in range(fft_sound.shape[1]):
            fft_sound[:, n] = fft.fft(windowedT[:, n], axis=0)[
                              :fft_sound.shape[0]]

        fft_soundT = fft_sound.T
        sound_strength = np.square(np.abs(fft_soundT))

        lowf = 0
        highf = s_rate / 2

        filt_places, m_freqs = self.get_filt_places(lowf, highf, freq_bins, fft_len,
                                               s_rate)
        filters = self.make_filters(filt_places, fft_len)

        e_val = 2.0 / (m_freqs[2:freq_bins + 2] - m_freqs[:freq_bins])
        filters *= e_val[:, np.newaxis]

        sound_filt = np.dot(filters, np.transpose(sound_strength))
        sound_log = 10.0 * np.log10(sound_filt)

        return sound_log

    def create_frames(self, sound, n_frames):
        """
        Split a sound into n different, slightly overlapping, frames
        :param sound: sound to be split, as a list
        :param n_frames: amount of frames to be created
        :return: a list of frames, and the size of a frame
        """
        frame_len = int(len(sound) / (n_frames - 1))
        FFT_size = frame_len + (len(sound) / (
                    n_frames - 1))  # This is the new calculated version
        FFT_size = int(FFT_size / 2) * 2
        sound = np.pad(sound, int(FFT_size / 2), mode='reflect')
        frames = np.zeros((n_frames, FFT_size))
        for n in range(n_frames):
            frames[n] = sound[n * frame_len:n * frame_len + FFT_size]

        return frames, FFT_size

    def f_mel(self, f):
        """
        Converts a frequency to a mel-scale value
        :param f: frequency value
        :return: mel-scale value
        """
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_f(self, mel):
        """
        Converts a mel-scale value to a frequency
        :param mel: mel-scale value
        :return: frequency value
        """
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def get_filt_places(self, lowF, highF, n_samples, fft_len, s_rate):
        """
        Computes filter points for filters
        :param lowF: lower frequency value
        :param highF: higher frequency value
        :param n_samples: amount of frequency bins/samples to be made
        :param fft_len: FFT-size of a frame
        :param s_rate: sample rate
        :return: list of filter points and frequencies
        """
        lowf_mel = self.f_mel(lowF)
        highf_mel = self.f_mel(highF)

        mel = np.linspace(lowf_mel, highf_mel, num=n_samples + 2)
        f = self.mel_f(mel)

        return np.floor((fft_len + 1) / s_rate * f).astype(int), f

    def make_filters(self, filt_places, fft_len):
        """
        Create filters based on filter points
        :param filt_places: filter points
        :param fft_len: FFT-size of a frame
        :return: list of filters
        """
        filters = np.zeros((len(filt_places) - 2, int(fft_len / 2 + 1)))

        for n in range(len(filt_places) - 2):
            filters[n, filt_places[n]: filt_places[n + 1]] = np.linspace(0, 1,
                    filt_places[n + 1] - filt_places[n])
            filters[n, filt_places[n + 1]: filt_places[n + 2]] = np.linspace(1,
                    0, filt_places[n + 2] - filt_places[n + 1])

        return filters

class result_handler():
    """This class is for handling results. Saving, loading, and printing"""
    def __init__(self):
        pass

    def save_file(self, filename, data):
        """
        Saves data to a file
        :param filename: filename to which data should be saved
        :param data: the data to be saved
        :return: None
        """
        np.save(filename, data)

    def load_file(self, filename):
        """
        Loads data from a file
        :param filename: filename from which to load data
        :return: loaded data
        """
        data = np.load(filename)
        return data

    def print_mfsc(self, results, mat, sample_name, index):
        """
        Prints mfsc features of a result, for TIDIGIT data
        :param results: total list of results
        :param mat: the read-in mat file
        :param sample_name: variable name in mat file
        :param index: index of the to-be-shown data
        :return: None
        """

        original_samples = mat[sample_name][:, 0]
        original_audios = [item for aud in original_samples for item in aud]

        plt.figure(figsize=(5, 5))
        plt.plot(np.linspace(0, len(original_audios[index]) / 20000,
                             num=len(original_audios[index])),
                 original_audios[index])
        plt.imshow(results[index].T, aspect='auto', origin='lower')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])



