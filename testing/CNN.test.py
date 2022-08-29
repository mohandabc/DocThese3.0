# %%
import numpy as np
from keras.models import load_model
from d_dataset import DataGenerator
from d_CNN import CNN

# Predict on test dataset
path_dataset_test_1 = 'Datasetres\\Dataset\\data1\\test'
path_dataset_test_2 = 'Datasetres\\Dataset\\data2\\test'
test_gen = DataGenerator(path_dataset_test_1, path_dataset_test_2)

# Load CNN if already trained
cnn_model = load_model('model\\model.h5')
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
    print(rounded[i], "=", real[i])
print(f"result {s} / {len(rounded)} = {s/len(rounded)}")
