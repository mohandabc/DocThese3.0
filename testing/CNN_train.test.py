from d_CNN import CNN
from d_dataset import DataGenerator

path_dataset_1 = 'Datasetres\\Dataset\\data1\\train'
path_dataset_2 = 'Datasetres\\Dataset\\data2\\train'
train_gen = DataGenerator(path_dataset_1, path_dataset_2, seed=1, validation_split=0.2, subset='train')
validation = DataGenerator(path_dataset_1, path_dataset_2, seed=1, validation_split=0.2, subset='validation')

cnn_model = CNN()
cnn_model.set_config('epochs', 5)
cnn_model.display(graph = True)


history = cnn_model.train(train_gen, validation)
cnn_model.save()