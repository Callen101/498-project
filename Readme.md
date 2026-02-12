
build_recording.py downloads files from the librispeech dataset, builds the metadata, concatenates the wav file, and adds a beep to the beginning. You may need to create a /data folder in your current repo. Note that this takes a random sample so downloaded files will be different each time you run the script.

analyze recording.py takes in a recorded wav file and splits it into the original smaller wav files using the beep at the beginning for synchronization. Note that you may need to create a /data folder in your current repo.

stream_wav.py is an example of running facebooks audio seal model.