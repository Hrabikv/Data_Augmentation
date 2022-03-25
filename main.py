import sys
import matplotlib.pyplot as plt
import numpy as np
from GAN import GAN
from DataWork import FileWorker, merge_data


# Function for print graphs from input data
def print_graph(raw_data):
    # print(data.shape)
    '''
    file_object = open('image.txt', 'a')
    for element in raw_data["allNonTargetData"]:
        for array in element:
            for digit in array:
                file_object.write('{:.1f} | '.format(digit))
            file_object.write("\n\n")
        file_object.write("\n-------------------------------------\n\n")
    file_object.close()
    '''
    index = 0

    for element in raw_data:
        fig, axs = plt.subplots(1, 3)

        axs[0].plot(element[0])
        axs[1].plot(element[1])
        axs[2].plot(element[2])
        fig.savefig("images/P300_%d.png" % index)
        plt.close()
        index += 1


# Function which train GAN on input data
# after training procedure trained GAN will be saved
# returns two model of GAN after training
def training(data_set):
    # Training of data from "target" class
    target = GAN()
    target.train(epochs=50000, dataset=data_set.get("target"), name="target",
                 batch_size=32, save_interval=1000)
    target.save_model("target_gan.h5")  # Saving of trained model for "target" class
    # Training of data from "non_target" class"
    non_target = GAN()
    non_target.train(epochs=50000, dataset=data_set.get("non_target"), name="non_target",
                     batch_size=32, save_interval=1000)
    non_target.save_model("non_target_gan.h5")  # Saving of trained model "non_target" class
    return target, non_target


# Function which load saved model of GAN
# returns two models of GAN with trained generators
def load_model(target_model_name, non_target_model_name):
    target = GAN()
    target.load_model(target_model_name)
    non_target = GAN()
    non_target.load_model(non_target_model_name)
    return target, non_target


# Function for predictions of data
# predicted data are merged with input data and saved into new file
def predict(file_worker, target, non_target, percentage, data_set):
    gen_target_data = target.predict(len(data_set.get("target")), int(percentage))
    new_target_data = merge_data(data_set.get("target"), gen_target_data)
    gen_non_target_data = non_target.predict(len(data_set.get("non_target")), int(percentage))
    new_non_target_data = merge_data(data_set.get("non_target"), gen_non_target_data)
    file_worker.save_data(new_target_data, new_non_target_data, "VarekaGTNEpochs{0}.mat".format(int(percentage)))


# Function which process arguments from command line
def process_args():
    symptoms = {}
    previous = ""
    symptoms["training"] = "n"
    for arg in sys.argv:
        if previous == "p":
            symptoms["percentage"] = arg
            previous = ""
        if previous == "tg":
            symptoms["target"] = arg
            previous = ""
        if previous == "ng":
            symptoms["non_target"] = arg
            previous = ""

        if arg == "-t":
            symptoms["training"] = "y"
        if arg == "-p":
            previous = "p"
        if arg == "-tg":
            previous = "tg"
        if arg == "-ng":
            previous = "ng"

    return symptoms


# Main function of program
# entry point of the program
if __name__ == '__main__':
    args = process_args()
    file = FileWorker()
    dataset = file.load_data()
    print(args)
    target_gan = ""
    non_target_gan = ""

    if args.keys().__contains__("training"):
        if args["training"] == "y":
            target_gan, non_target_gan = training(dataset)
            print("Training is done.")

    if args.keys().__contains__("percentage"):
        if args.keys().__contains__("non_target") & args.keys().__contains__("target"):
            target_gan, non_target_gan = load_model(args["target"], args["non_target"])
            predict(file, target_gan, non_target_gan, args["percentage"], dataset)
            print("Predicting is done.")
        else:
            print("Models of generator are missing!!")
            print("Check README.txt for right parameters!!")

'''
    percentage = sys.argv[1]

    

    # print_graph(dataset)
    target_gan = GAN()
    target_gan.load_model("target_gen.h5")
    # print_graph(dataset["allNonTargetData"])
    #target_gan.train(epochs=50000, dataset=dataset.get("target"), name="target",
    #                  batch_size=32, save_interval=1000)
    # target_gan.save_model("target_gen.h5")
    gen_target_data = target_gan.predict(len(dataset.get("target")), int(percentage))

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
    file.save_data(new_target_data, new_non_target_data, "VarekaGTNEpochs{0}.mat".format(int(percentage)))
'''
