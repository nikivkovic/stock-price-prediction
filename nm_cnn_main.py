import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tfpip
import logging

tf.get_logger().setLevel(logging.ERROR) # da ne bi dolazilo do warning-a u izvrsavanju

from keras import layers
from keras import Sequential
from keras.utils import image_dataset_from_directory
from keras.losses import SparseCategoricalCrossentropy

from keras.losses import CategoricalCrossentropy

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# U ovom primeru dubokog ucenja radi se klasifikacija automobila, brodova i aviona na osnovu dostupnih slika.
# Dakle, ulazni podaci su slike, a klase su: cars(automobili), ships(brodovi), airplanes(avioni).

from PIL import Image
import glob

# Predobrada slika - svodjenje svih slika na istu dimenziju, tj.  244x244.
resized_images = []
new_height = 244

def resize(im, new_height):
    width, height = im.size
    ratio = width/height
    new_width = int(ratio*new_height)
    resized_image = im.resize((new_width, new_height))
    resized_width, resized_height = resized_image.size
    box = (int(resized_width/2) - 122, 0, int(resized_width/2) + 122, 244)
    resized_image = resized_image.crop(box)
    return resized_image

for filename in glob.glob('./Dataset/train/airplanes/*'):
    img = Image.open(filename)
    img = resize(img, new_height)
    resized_images.append(img)

for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('./Dataset_resized/train/airplanes/airplane',i+1,'.jpg'))

resized_images.clear()

for filename in glob.glob('./Dataset/train/cars/*'):
    img = Image.open(filename)
    img = resize(img, new_height)
    resized_images.append(img)

for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('./Dataset_resized/train/cars/car',i+1,'.jpg'))

resized_images.clear()

for filename in glob.glob('./Dataset/train/ship/*'):
    img = Image.open(filename)
    img = resize(img, new_height)
    resized_images.append(img)

for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('./Dataset_resized/train/ships/ship',i+1,'.jpg'))

resized_images.clear()

for filename in glob.glob('./Dataset/test/airplanes/*'):
    img = Image.open(filename)
    img = resize(img, new_height)
    resized_images.append(img)

for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('./Dataset_resized/test/airplanes/airplane',i+1,'.jpg'))

resized_images.clear()

for filename in glob.glob('./Dataset/test/cars/*'):
    img = Image.open(filename)
    img = resize(img, new_height)
    resized_images.append(img)

for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('./Dataset_resized/test/cars/car',i+1,'.jpg'))

resized_images.clear()

for filename in glob.glob('./Dataset/test/ships/*'):
    img = Image.open(filename)
    img = resize(img, new_height)
    resized_images.append(img)

for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('./Dataset_resized/test/ships/ship',i+1,'.jpg'))

resized_images.clear()

main_path = './Dataset_resized/train'
test_path = './Dataset_resized/test'
img_size = (244, 244)
batch_size = 64

# Podela na trening i validacioni skup. Bitno je kako bismo izbegli preobucavanje mreze (gubitak mogucnosti generalizacije podataka). Test skup je zaseban u preuzetom dataset-u i sluzi za testiranje mreze kako samo ime kaze.
# Iz podfoldera za trening dataset-a uzimamo 80% podataka zapravo za trening, a ostalih 20% za validaciju.
Xtrain = image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

Xtest = image_dataset_from_directory(test_path,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

classes = Xtrain.class_names
# print(classes)

# Pie chart za prikaz broja podataka po klasama
set = [800, 200, 189, 800, 200, 193, 800, 200, 200] # broj podataka po klasama za trening/validaciju/test
set_labels = ['Airplanes - train', 'Airplanes - validation', 'Airplanes - test', 'Cars - train', 'Cars - validation', 'Cars - test', 'Ships - train', 'Ships - validation', 'Ships - test']
plt.figure('Podaci po klasama')
def value(x):
    return '{:.2f}%\n({:.0f})'.format(x, np.sum(set)*x/100)
plt.pie(set, labels=set_labels, autopct=value, startangle=90)
plt.axis('equal')
# Podaci su balansirani kako imamo ravnomeran broj odbiraka za svaku od klasa.

N = 10
plt.figure('Primeri podataka klasa')
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
    ]
)

N = 10
plt.figure('Primeri augmentacije')
for img, lab in Xtrain.take(1):
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.title(classes[lab[0]])
        plt.axis('off')

# U prvom i skrivenim slojevima koristili smo relu aktivacionu funkciju. U poslednjem sloju smo koristili softmax kako nam treba klasifikacija 3 vrste objekata
num_classes = len(classes)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(244, 244, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.summary() # Imamo 7,396,899 parametara

# Koristili smo SpraseCategoricalCrossentropy kriterijumsku funkciju kako imamo problem klasifikacije (i vise od 2 klase).
# Koriscen je Adam optimizator kako dobro radi na vecim dataset-ovima i kompjuterski je efikasan.
model.compile(Adam(learning_rate=0.005),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
history = model.fit(Xtrain,
                    epochs=100,
                    validation_data=Xval,
                    callbacks=[es],
                    verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Performanse
plt.figure('Performanse neuralne mreze')
plt.subplot(1, 2, 1)
plt.plot(acc)
plt.plot(val_acc)
plt.grid()
plt.legend(['Trening', 'Validacija'])
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(loss)
plt.plot(val_loss)
plt.grid()
plt.legend(['Trening', 'Validacija'])
plt.title('Loss')

# Matrica konfuzije na trening skupu
labels = np.array([])
pred = np.array([])
for img, lab in Xtrain:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import accuracy_score
print('Tačnost modela na trening skupu je: ' + str(100*accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()

# Matrica konfuzije na test skupu
labels = np.array([])
pred = np.array([])
for img, lab in Xtest:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import accuracy_score
print('Tačnost modela na test skupu je: ' + str(100*accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()

# Primer klasifikacije slika
N = 20
pred = np.array([])
plt.figure('Primer klasifikacije slika')
for img, lab in Xtest.take(1):
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))
    for i in range(N):
        plt.subplot(4, int(N/4), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[int(pred[i])])
        plt.axis('off')

plt.show()