from d_dataset import DataGenerator

path_dataset_test_1 = 'Datasetres\\Dataset\\data1\\test'
path_dataset_test_2 = 'Datasetres\\Dataset\\data2\\test'
test_gen = DataGenerator(path_dataset_test_1, path_dataset_test_2, 1)
test_gen.set_data_type('XYZ')

print(test_gen[0])