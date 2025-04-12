# 7643-fl-momentum

DL project group

Dependencies in the requirements.txt
to run, python3 run.py

edit configuration in config.py (epochs, distirbution, etc.)

modify and update models and optimizers in models.py and optimizers.py

mess with the aggregation strategies in strategies.py, fedavg is the default one i'm using for now. see docs

** Notes 4/10**

Implement a better model like resnet.
Explore different optimizers
Adam
SGD
Explore different momentums:
Polyak
Netsorov (Aamir)
QHM (not in pytorch, but already implemented somewhere) (Alireza)
Rectified Adam (Mike)
ACC SGD (Ansel)
Do we need momentum for server side aggregation or just client side is okay?
