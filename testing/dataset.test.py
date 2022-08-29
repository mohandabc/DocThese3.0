from d_dataset import Dataset

dataset = Dataset(images_folder="dataset\\", output_folder="datasetres\\")
dataset.set_config("nb_processed_imgs", 100)
dataset.process()
