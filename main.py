import numpy as np
from scipy.io import loadmat # , savemat

from GAN import GAN


def print_hi(name):
    print(f'Hi, {name}')
    print("ahoj")

    data = loadmat('VarekaGTNEpochs.mat')
    '''
    print(data["allNonTargetData"][0].shape)
    target_data, non_target_data = data['allTargetData'], data['allNonTargetData']  # get target and non-target data
    print(target_data.shape)
    print(non_target_data.shape)
    features = np.concatenate((target_data, non_target_data))
    print(features.shape)
    # Target labels are represented as (1, 0) vector, non target labels are represented as (0, 1) vector
    target_labels = np.tile(np.array([1, 0]), (target_data.shape[0], 1))  # set 'target' as (1, 0) vector
    non_target_labels = np.tile(np.array([0, 1]), (non_target_data.shape[0], 1))  # set 'non target' as (0, 1) vector
    labels = np.vstack((target_labels, non_target_labels))  # concatenate target and non target labels
    features = features.reshape((features.shape[0], 1, -1))
    labels = labels.reshape((labels.shape[0], 1, -1))
    print(f'Features shape: {features.shape}, Labels shape: {labels.shape}')
    '''
    return data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = print_hi("nema")
    gan = GAN()
    gan.train(epochs=30000, data=data, batch_size=32, save_interval=200)
    print(gan.predict(data["allNonTargetData"]).shape)

