# people-moving-prediction

Prediction on the future population flow by the inflow and outflow numbers 

The latest work is in the file SRTcode\New\new.py LSTM_new.py.

Using the medieum output of the Conv in file new.py to predict the future by LSTM in LSTM_new.py.

The structure of the new.py is in model1.png, and the structure of the LSTM_new.py is in the modelLSTM.png.

Now using LSTM predicting separately(8*16=128 pixel, thus 128 different LSTM).

The Conv kernel=(3,3) except the last Conv kernel=(1,1) for the LSTM's separately predicting.
