from d_CNN import CNN
from d_dataset import DataGenerator
import matplotlib.pyplot as plt

path_dataset_1 = 'Datasetres\\Dataset\\data1\\train'
path_dataset_2 = 'Datasetres\\Dataset\\data2\\train'
cnn_name = "Last_cnn"
train_gen = DataGenerator(path_dataset_1, path_dataset_2, seed=1, validation_split=0.2, subset='train')
validation = DataGenerator(path_dataset_1, path_dataset_2, seed=1, validation_split=0.2, subset='validation')

cnn_model = CNN()
cnn_model.set_config('epochs', 100)
cnn_model.display(name = cnn_name, graph = True)


history = cnn_model.train(train_gen, validation)
cnn_model.save(cnn_name)



import numpy as np
from keras.models import load_model

# Predict on test dataset
path_dataset_test_1 = 'Datasetres\\Dataset\\data1\\test'
path_dataset_test_2 = 'Datasetres\\Dataset\\data2\\test'
test_gen = DataGenerator(path_dataset_test_1, path_dataset_test_2)

# Load CNN if already trained
cnn_model = load_model(f'model\\{cnn_name}.h5')
res = cnn_model.predict(test_gen)


# Display results
rounded = np.argmax(res, axis=-1)
real = []
for i in test_gen:  
    real.extend(i[1].numpy())

s=0
for i in range(len(real)):
    if rounded[i] == real[i]:
        s+=1
    # print(rounded[i], "=", real[i])
print(f"result {s} / {len(rounded)} = {s/len(rounded)}")



print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()