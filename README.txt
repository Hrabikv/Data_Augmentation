The project is written in Python version 3.8.5.

First you will need all the file from GitHub repository:

https://github.com/Hrabikv/Data_Augmentation. 

You can download it via Git or by clicking Code -> Download ZIP. After downloading is finished, extract all the file from the zip into a folder where you want to work (the best option is a folder with do not have any non-ASCII character in path).

When you have prepared thr folder with the project it will need the environment. If you are familiar with Python enough, install all necessary modules from "requirements.txt".

Otherwise, follow these steps:

	1)
	Download Anaconda3: 

	https://www.anaconda.com/

	and install it. Make sure that it is added in PATH variables in Windows. The install process can do it for you if you check 		off one checkbox in the install procedure.
	
	2)
	Open the command line. In the folder with the project ,write "cmd" into navigation line. 

	3)
	Now, create a new environment using the command:
	
	conda create -n name of your environment python=3.8.5

	"name of your environment" replace with the name that you want. This will create an empty environment that can be used to 	install dependencies.

	4)
	Open the created environment by command: 

	conda activate name of your environment

	This will switch you into conda environment.

	5)
	To install dependencies write this command:

	pip install -r requirements.txt

	This will install all module into conda environment.

Before project can be run you will need a dataset. A dataset can be downloaded on:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/G9RRLN

There is a file named "VarekaGTNEpochs.mat". After downloading move this file into the folder with the other project files.

Now, everything should be ready to run the project.

####################################################################

The project is run by command:

python mian.py

in prepared conda environment.

The project has a lot of parameters in config.txt. There are all parameters necessary for running the project. Do not delete any flags of parameters. There are parameters which need specific values. For example, there are first two parameters which indicates model which we want to train or used for augmenting. These specific values are described above each parameter. If there are not described any specific values, these parameters have no exact limitations. For example, there are parameters with flags "-tg" and "-ng" which need files with trained GANs. These files got default name after training, but you can be renamed them if you want. Because of it you must here select specific files. Same it is for next two parameters "-p" and "-w" which set two main parameters for augmenting. "-p" described size of created dataset in percents. "-w" described size of window in number of samples. Last three parameters have again specific values. In this case they are only switches whatever or not we want print examples of generated data during training, print input data or print final created data.

The training process is marked by flag "-t". This flag has three specific values. There are described in config.txt in the first section.

The augmenting process needs five parameters. The first parameter is "-m" which determines used model. The second and the third parameter are for path to trained model of GAN. Their flags are "-tg" for target GAN and "-ng" for non target GAN. The fourth parameter with flag "-p" determines size of augmented dataset. The last parameter "-w" determines size of averaging window. These parameters are in the middle section in config.txt.




