# Seismic Arrival-Time ML Pipeline for LabNet: Deep Learning for High-Frequency Seismic Data <br>
## University of Texas at Austin
Created by Isaac Xu and Dr. Chas Bolton
# Download instructions
Download LabNet.py file and run requirements.txt 
# Overview
Machine learning pipeline developed for Bolton laboratory with scripts for model training and inference as well as 
LabNet library to easily adapt code. Documentation is found [here](https://www.notion.so/LabNet-Documentation-24094d2dd03d80909cefe76507690111?source=copy_link)
# Background
These scripts were developed to adapt PhaseNet (Zhu et al. 2018) to higher-frequency laboratory data. The backbone
of these scripts were developed using the Seisbench python library and primarily uses the object VariableLengthPhaseNet
which allows for a variable input layer size - needed for higher frequency data (1-5 MHz). <br> 
The main scripts for training and inference is model_validationAN.py and phasenet_custom_trainingANR.py, the codes are located below
for reference. These are built to read from csv's produced from a trace marking script found [here](https://github.com/Is-Xu1/waveform_marking_bolton/tree/main)
# Extra scripts
We experimented with several preprocessing methods, the scripts for training and inferencing these models are stored in
ExtraScripts directory, the preprocessing codes are the following: <br>
AN - Amplitude Scaling with Window Normalization <br>
ANB - Amplitude Scaling with Whole Waveform Normalization and Bandpass Filter <br>
ANBR - Amplitude Scaling with Whole Waveform Normalization and Bandpass Filter and Randomly Augmented Pick Window Added <br>
ANR - Amplitude Scaling with Whole Waveform Normalization  and Randomly Augmented Pick Window Added <br>
ANRV - Amplitude Scaling with Whole Waveform Normalization and Randomly Augmented Pick Window Added and Validation Script Stop <br>
ANRC - Amplitude Scaling with Whole Waveform Normalization and Randomly Augmented Pick Window Added and Cirriculum Training 
