import numpy as np
from scipy.io import loadmat, savemat


# Function for merge input data with generated data
def merge_data(dataset, gen_data):
    pom_list = []

    for i in range(len(dataset)):
        pom_list.append(dataset[i])

    for i in range(len(gen_data)):
        pom_list.append(gen_data[i])

    return np.array(pom_list)


# Class which works with file ".mat" format
class FileWorker:
    def __init__(self):
        self.data = loadmat("VarekaGTNEpochs.mat")

    # Function for prepare input data
    # are filtered of values above 100 uV
    def load_data(self):
        target_data, non_target_data = self.data["allTargetData"], self.data["allNonTargetData"]  # get target and non-target data

        # Filter noise above 100 uV
        threshold = 100.0
        target_data_result, non_target_data_result = [], []
        for i in range(target_data.shape[0]):
            if not np.max(np.abs(target_data[i])) > threshold:
                target_data_result.append(target_data[i])
        for i in range(non_target_data.shape[0]):
            if not np.max(np.abs(non_target_data[i])) > threshold:
                non_target_data_result.append(non_target_data[i])

        # Save data to numpy array
        target_data = np.array(target_data_result)
        non_target_data = np.array(non_target_data_result)
        data_set = {"target": target_data, "non_target": non_target_data}

        for data in data_set:
            print(data)
            print(data_set.get(data).shape)
        return data_set

    # Data are saved into new ".mat" file
    def save_data(self, new_gen_target, new_gen_non_target, file_name):
        self.data["allTargetData"] = new_gen_target
        self.data["allNonTargetData"] = new_gen_non_target

        print(self.data["allTargetData"].shape)
        print(self.data["allNonTargetData"].shape)

        savemat(file_name, self.data)
