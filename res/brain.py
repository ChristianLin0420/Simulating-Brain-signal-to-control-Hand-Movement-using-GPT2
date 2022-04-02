

class Brain():

    def __init__(self):
        super(Brain, self).__init__()

        self.channel_names = [  "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "C3", 
                                "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz", "P4",
                                "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1",
                                "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C1", "C2", "C6", "TP7", "CP3", "CPz",
                                "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7", "PO3", "POz", "PO4", "PO8" ]

    def get_channel_names(self):
        return self.channel_names