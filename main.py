import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from GAN import GAN


def print_graph(data):
    print(data.shape)
    '''
    file_object = open('image.txt', 'a')
    for element in data["allNonTargetData"]:
        for array in element:
            for digit in array:
                file_object.write('{:.1f} | '.format(digit))
            file_object.write("\n\n")
        file_object.write("\n-------------------------------------\n\n")
    file_object.close()
    '''
    index = 0

    for element in data:
        fig, axs = plt.subplots(1, 3)

        axs[0].plot(element[0])
        axs[1].plot(element[1])
        axs[2].plot(element[2])
        fig.savefig("images/P300_%d.png" % index)
        plt.close()
        index += 1


def load_data(name):
    data = loadmat(name)

    target_data, non_target_data = data['allTargetData'], data['allNonTargetData']  # get target and non-target data

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


if __name__ == '__main__':
    dataset = load_data('VarekaGTNEpochs.mat')
    # print_graph(dataset)
    target_gan = GAN()
    # print_graph(dataset["allNonTargetData"])
    target_gan.train(epochs=50000, dataset=dataset.get("target"), batch_size=32, save_interval=1000)

    new_target_data = target_gan.predict(dataset.get("target"), 150)

    print(dataset.get("target").shape)
    print(new_target_data.shape)

    non_target_gan = GAN()

    non_target_gan.train(epochs=50000, dataset=dataset.get("non_target"), batch_size=32, save_interval=1000)

    new_non_target_data = non_target_gan.predict(dataset.get("non_target"), 150)

    print(dataset.get("non_target").shape)
    print(new_non_target_data.shape)

    target_gan.save_model("target_gen.h5")
    non_target_gan.save_model("non_target_gen.h5")
