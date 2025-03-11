
"""""
Dataset configurations:
    :param DATASET_PATH -> the directory path to dataset .tar files
    :param TASK_ID -> specifies the the segmentation task ID (see the dict below for hints)
    :param IN_CHANNELS -> number of input channels
    :param NUM_CLASSES -> specifies the number of output channels for dispirate classes
    :param BACKGROUND_AS_CLASS -> if True, the model treats background as a class

"""""
DATASET_PATH = './data'
TASK_ID = 4
IN_CHANNELS = 1
NUM_CLASSES = 1
BACKGROUND_AS_CLASS = True


"""""
Training configurations:
    :param TRAIN_VAL_TEST_SPLIT -> delineates the ratios in which the dataset shoud be splitted. The length of the array should be 3.
    :param SPLIT_SEED -> the random seed with which the dataset is splitted
    :param TRAINING_EPOCH -> number of training epochs
    :param VAL_BATCH_SIZE -> specifies the batch size of the training DataLoader
    :param TEST_BATCH_SIZE -> specifies the batch size of the test DataLoader
    :param TRAIN_CUDA -> if True, moves the model and inference onto GPU
    :param BCE_WEIGHTS -> the class weights for the Binary Cross Entropy loss
"""""
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
SPLIT_SEED = 42
TRAINING_EPOCH = 100
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
TRAIN_CUDA = True
BCE_WEIGHTS = [0.004, 0.996]
