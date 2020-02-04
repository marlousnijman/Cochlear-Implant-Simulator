# Imports
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile
import cochlear_implant_simulator as sim
import librosa
import numpy as np
import os, random
import spectrogram as spec
import torch


# Resamples the data in one directory to have a new sampling rate (new_rate) while maintaining the same length, and
# stores the resampled sound fragments to another directory.
# from_dir: directory from which sound fragments are resampled
# to_dir: directory where resampled sound fragments are saved
# new_rate: the new sampling rate
def resample(from_dir, to_dir, new_rate):
    for filename in os.listdir(from_dir):
        data, rate = librosa.load(from_dir + "/" + filename, sr=new_rate)
        data = np.around(data * 32767).astype(np.dtype("i2")) # From 32 to 16 bit representation
        wavfile.write(to_dir + "/" + filename, rate, data)


# Creates chunks of equal length from all sound fragments in a certain directory and stores them in another directory.
# from_directory: directory from which sound fragments are cut into chunks
# to_directory: directory where the resulting chunks are stored
# chunk_length_ms: length of the chunks in ms
def create_chunks(from_directory, to_directory, chunk_length_ms):
    for filename in os.listdir(from_directory):
        myaudio = AudioSegment.from_file(from_directory + "/" + filename, "wav")
        chunks = make_chunks(myaudio, chunk_length_ms)

        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk)

            if(chunk_len < chunk_length_ms):
                chunk = chunk + AudioSegment.silent(duration=(chunk_length_ms - chunk_len))

            chunk_name = filename.replace(".wav","_chunk{0}.wav".format(i))
            chunk.export(to_directory + "/" + chunk_name, format="wav")


# Reads all wav files from a certain directory, and appends zero's to the data if the length is shorter than data_len
# from_dir: directory from which wav files are loaded in
# data_len: the length the data should have, corresponding to the length of the wav file in ms
def load_data(from_dir, data_len):
    all_data = []

    for filename in sort_files(os.listdir(from_dir)):
        rate, data = wavfile.read(from_dir + "/" + filename)

        if (len(data) < data_len):
            data = np.pad(data, (0, data_len - len(data)), 'constant')
        all_data.append(data)

    return all_data


# Sorts all the files in a certain directory so that it is sorted like ["string1", string2", ..., "string10"]
# instead of ["string1", string10", ..., "string2"], i.e., it sorts on both the strings as well as the order of the
# corresponding integer.
# from_dir: directory of which the files are sorted
def sort_files(from_dir):
    index_sum = 0
    last_index = 0
    current_file = from_dir[0][:8]
    sorted_files = [None] * len(from_dir)

    for file in sorted(from_dir):
        file_name = file[:8]
        file_nr = int(file[14:].replace(".wav", ""))

        if file_name == current_file:
            sorted_files[index_sum + file_nr] = file
            last_index += 1
        else:
            current_file = file_name
            index_sum += last_index
            last_index = 1
            sorted_files[index_sum + file_nr] = file

    return sorted_files


# Converts sound data (i.e. data obtained from a wav file) obtained from a pytorch tensor into a spectrogram.
# data: pytorch tensor that is converted to a spectrogram
def sound_to_spec(data):
    data = data.detach().numpy()
    spectrogram = spec.pretty_spectrogram(data.astype('float64'), fft_size=spec.fft_size,
                                          step_size=spec.step_size, log=True, thresh=spec.spec_thresh)
    spectrogram = spectrogram / -spec.spec_thresh

    return spectrogram


# Processes a batch during the training and testing of the network, by converting all the data in the batch to
# spectrograms, and creating a torch tensor of it with the correct dimensions.
# batch: batch that is processed
def process_batch(batch):
    noisy_data = []
    clean_data = []

    for data in batch:
        noisy_data.append(sound_to_spec(data[0]))
        clean_data.append(sound_to_spec(data[1]))

    noisy_tensor = torch.tensor(noisy_data, dtype=torch.float32).view(-1, 1, 112, 256)
    clean_tensor = torch.tensor(clean_data, dtype=torch.float32).view(-1, 1, 112, 256)

    return noisy_tensor, clean_tensor


# Converts the results of the network to a spectrogram with the correct dimensions and type and adds it to an array
# containing other results.
# batch: batch from wich the resuls are converted
# results: array of already converted results
def convert_results(batch, results):
    for data in batch:
        converted_result = data.detach().numpy().reshape(112, 256)
        results.append(converted_result)

    return results


# Creates a list of indices on which to split all the testing data, such that all chunks belonging to one wav file
# end up in the same array.
# from_dir: directory where the original wav files are stored
def create_split_indices(from_dir):
    split_indices = []
    i = 0
    previous_filename = ""

    for filename in sort_files(os.listdir(from_dir)):
        filename = filename[:8]

        if i == 0:
            previous_filename = filename

        if filename == previous_filename:
            i += 1

        else:
            previous_filename = filename
            split_indices.append(i)
            i += 1

    return split_indices


# Converts a spectrogram back to sound.
# spec_array: array consisting of spectrograms that will be converted to sound
def spec_to_sound(spec_array):
    spec_array = spec_array * -spec.spec_thresh
    audio = []

    for i in range(spec_array.shape[0]):
        recovered_audio = spec.invert_pretty_spectrogram(spec_array[i], fft_size=spec.fft_size,
                                                         step_size=spec.step_size, log=True, n_iter=10)

        recovered_audio_louder = recovered_audio * 1e7

        audio.append(recovered_audio_louder)

    return audio


# Appends all sound chucks in each array to one another and saves these as wav files that are a result of testing the
# network to a new directory, with filenames that are equal to the file names used as input to the network.
# sound_chunks: array of arrays that contain the sound generated by the network
# rate: sample rate
# name_dir: directory that contains the data that was used as input to the network
# to_dir: name of the directory to which to save the wav files
def save_sound_to_dir(sound_chunks, rate, name_dir, to_dir):
    filenames =  sorted(os.listdir(name_dir))

    for i, sound_chunk in enumerate(sound_chunks):
        sound = []

        for chunk in sound_chunk:
            sound.extend(chunk)

        wavfile.write(to_dir + "/" + filenames[i], rate, np.array(sound).astype(np.dtype('i2')))


# Runs the data from one directory through the cochlear implant simulator and saves it to another directory.
# from_dir: directory from which data is simulated
# to_dir: directory where simulated data will be stored
# sin_waves: array of sine waves at different frequencies, matching to those in freqs
# freqs: frequency ranges corresponding to the channels in the to be simulated cochlear implant
def convert_to_sim_data(from_dir, to_dir, freq_ranges, samples):
    for i in range(samples):
        filename = random.choice(os.listdir(from_dir))
        rate, data = wavfile.read(from_dir + "/" + filename)
        chunks = np.split(data, range(rate, len(data), rate))
        sim_data = []
        for chunk in chunks:
            sim_chunk = sim.cis(chunk, rate, freq_ranges)
            sim_data.extend(sim_chunk)
        sim_data = np.asarray(sim_data)
        wavfile.write(to_dir + "/" + filename, rate, sim_data.astype(np.dtype("i2")))