import os
import matplotlib.pyplot as plt
import numpy as np
from GAN import GAN
from DataWork import FileWorker, merge_data, load_config


# Function for print graphs from input data
def print_graph(name, raw_data):
    print("Printing {0}".format(name))
    try:
        os.makedirs(os.path.join("./", name))
    except OSError:
        print()

    index = 0

    x = []
    for i in range(-200, 1000):
        x.append(i)

    for element in raw_data:
        fig, axs = plt.subplots(1, 3)

        axs[0].plot(x, element[0])
        axs[0].set_xlabel("ms")
        axs[0].set_ylabel("uV")
        axs[0].set_title("Fz")
        axs[0].set_xticks(np.arange(-200, 1050, 500))

        axs[1].plot(x, element[1])
        axs[1].set_xlabel("ms")
        axs[1].set_title("Cz")
        axs[1].set_xticks(np.arange(-200, 1050, 500))

        axs[2].plot(x, element[2])
        axs[2].set_xlabel("ms")
        axs[2].set_title("Pz")
        axs[2].set_xticks(np.arange(-200, 1050, 500))

        fig.savefig("{0}/P300_{1}.png".format(name, index))
        plt.close()
        index += 1


# Function which train GAN on input data
# after training procedure trained GAN will be saved
# returns two model of GAN after training
def training(data_set, model, examples):
    # Training of data from "target" class
    target = GAN(model)
    target.train(epochs=50000, dataset=data_set.get("target"), name="target",
                 examples=examples, batch_size=32, save_interval=1000)
    target.save_model("target_gan_mk_{0}.h5".format(model))  # Saving of trained model for "target" class
    # Training of data from "non_target" class"
    non_target = GAN(model)
    non_target.train(epochs=50000, dataset=data_set.get("non_target"), name="non_target",
                     examples=examples, batch_size=32, save_interval=1000)
    non_target.save_model("non_target_gan_mk_{0}.h5".format(model))  # Saving of trained model "non_target" class
    return target, non_target


# Function which load saved model of GAN
# returns two models of GAN with trained generators
def load_model(target_model_name, non_target_model_name, model):
    target = GAN(model)
    target.load_model(target_model_name)
    non_target = GAN(model)
    non_target.load_model(non_target_model_name)
    return target, non_target


# Function for predictions of data
# predicted data are merged with input data and saved into new file
def predict(file_worker, target, non_target, percentage, data_set, window, graph):
    print("Predicting target class.")
    gen_target_data = target.predict(len(data_set.get("target")), int(percentage), window)
    new_target_data = merge_data(data_set.get("target"), gen_target_data)
    print("Predicting non target class.")
    gen_non_target_data = non_target.predict(len(data_set.get("non_target")), int(percentage), window)
    if graph == "T":
        print_graph("output_target", gen_target_data)
        print_graph("output_non_target", gen_non_target_data)
    new_non_target_data = merge_data(data_set.get("non_target"), gen_non_target_data)
    file_worker.save_data(new_target_data, new_non_target_data, "VarekaGTNEpochs{0}.mat".format(int(percentage)))


# Main function of program
# entry point of the program
if __name__ == '__main__':
    args = load_config()
    file = FileWorker()
    dataset = file.load_data()

    target_gan = ""
    non_target_gan = ""

    if args.keys().__contains__("-gi") & (args["-gi"] == "T"):
        print_graph("input_target", dataset.get("target"))
        print_graph("input_non_target", dataset.get("non_target"))

    if args.keys().__contains__("-t"):
        if args["-t"] != "n":
            if args.keys().__contains__("-e"):
                target_gan, non_target_gan = training(dataset, args["-t"], args["-e"])
            else:
                print("Missing argument!")
            print("Training is done.")
    else:
        print("Missing argument!")

    if args.keys().__contains__("-m"):
        if args["-m"] == "1" or args["-m"] == "2":
            if args.keys().__contains__("-tg") & args.keys().__contains__("-ng"):
                target_gan, non_target_gan = load_model(args["-tg"], args["-ng"], args["-m"])
                if args.keys().__contains__("-w"):
                    if args.keys().__contains__("-go"):
                        predict(file, target_gan, non_target_gan, args["-p"], dataset, int(args["-w"]), args["-go"])
                    print("Predicting is done.")
            else:
                print("Models of generator are missing!!")
                print("Check README.txt for right parameters!!")

    print("Everything is done. Exiting the project.")
