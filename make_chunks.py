# Imports
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

# Creates chunks of equal length from all sound fragments in a certain directory and stores them in another directory.
# from_directory: directory from which sound fragments are cut into chunks
# to_directory: directory where the resulting chunks are stored
# chunk_length_ms: length of the chunks in ms
def make_chunks(from_directory, to_directory, chunk_length_ms):
    for filename in os.listdir(from_directory):
            myaudio = AudioSegment.from_file(from_directory+"/"+filename, "wav")
            chunks = make_chunks(myaudio, chunk_length_ms)

            for i, chunk in enumerate(chunks):
                chunk_len = len(chunk)
                if(chunk_len < chunk_length_ms):
                    chunk = chunk + AudioSegment.silent(duration=(chunk_length_ms-chunk_len))

                chunk_name = filename.replace(".wav","_chunk{0}.wav".format(i))
                print ("exporting", chunk_name)
                chunk.export(to_directory+"/"+chunk_name, format="wav")