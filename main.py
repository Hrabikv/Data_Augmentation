import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from GAN import GAN
from Data_set import Data_set


def print_graph(data):
    # print(data.shape)
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
        fig.savefig("generated/P300_%d.png" % index)
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


def merge_data(dataset, gen_data):
    pom_list = []

    for i in range(len(dataset)):
        pom_list.append(dataset[i])

    for i in range(len(gen_data)):
        pom_list.append(gen_data[i])

    return np.array(pom_list)


if __name__ == '__main__':
    percentage = sys.argv[1]

    data = Data_set()
    dataset = data.load_data()

    # print_graph(dataset)
    target_gan = GAN()
    target_gan.load_model("target_gen.h5")
    # print_graph(dataset["allNonTargetData"])
    # gen_target_data target_gan.train(epochs=50000, dataset=dataset.get("target"), name="target",
    #                  batch_size=32, save_interval=1000)
    # target_gan.save_model("target_gen.h5")
    gen_target_data = target_gan.predict(len(dataset.get("target")), int(percentage))
    print_graph(gen_target_data)
    new_target_data = merge_data(dataset.get("target"), gen_target_data)

    # print(dataset.get("target").shape)
    # print(new_target_data.shape)

    non_target_gan = GAN()
    non_target_gan.load_model("non_target_gen.h5")
    # non_target_gan.train(epochs=50000, dataset=dataset.get("non_target"), name="non_target",
    #                     batch_size=32, save_interval=1000)
    # non_target_gan.save_model("non_target_gen.h5")
    gen_non_target_data = non_target_gan.predict(len(dataset.get("non_target")), int(percentage))

    new_non_target_data = merge_data(dataset.get("non_target"), gen_non_target_data)

    # print(dataset.get("non_target").shape)
    # print(new_non_target_data.shape)
    data.save_data(new_target_data, new_non_target_data, "VarekaGTNEpochs{0}.mat".format(percentage))

