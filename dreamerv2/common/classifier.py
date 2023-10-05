import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class Classifier():
  def __init__(self, data_path):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
      except RuntimeError as e:
        print(e)

    data_dir = data_path
    data_dir = pathlib.Path(data_dir).with_suffix('')

    self.batch_size = 16
    self.img_height = 256
    self.img_width = 256

    self.train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

    self.val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size)

    class_names = self.train_ds.class_names

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    #first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    #print(np.min(first_image), np.max(first_image))

    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                        input_shape=(self.img_height,
                                    self.img_width,
                                    3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
    )

    num_classes = len(class_names)

    self.model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
    ])



    self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  
  def train(self):
    epochs=20

    history = self.model.fit(
    self.train_ds,
    validation_data=self.val_ds,
    epochs=epochs
    )
    return
  
  def predict(self,image):
    img = image.copy()
    img = tf.expand_dims(img, 0) # Create a batch
    predictions = self.model.predict(img)
    for pred in predictions:
      score = tf.nn.softmax(pred).numpy()
      max_score = np.max(score)
      unfold = np.argmax(score) and max_score > 0.9999
      if unfold:
        return 1.0
    return 0.0

def thread_function(files):
  path = "/home/rad/.keras/256/"
  new_path = '/home/rad/Documents/ClothHangProject/episodes/normal_fixed_IK_class'

  classifier = Classifier(path)
  classifier.train()

  ep  = 0
  total = len(files)
  for file in files:
    episode = np.load(join(mypath, file), allow_pickle=True)
    observations = episode['observation']
    images = episode['image']
    depths = episode['depth']
    rewards = []
    is_firsts = episode['is_first']
    is_lasts = episode['is_last']
    is_terminals = episode['is_terminal']
    actions = episode['action']

    images256 = episode['largeimgs']

    for i in range(len(images256)):
      image = images256[i]
      cls_prediction = classifier.predict(image)
      # image = classifier.resize(image)
      rewards.append(cls_prediction)
    filename = join(new_path, file)
    print(f"{ep} / {total}")
    ep += 1
    np.savez_compressed(filename, observation=observations, image=images, depth=depths, reward=rewards, is_first=is_firsts, is_last=is_lasts, is_terminal=is_terminals,action=actions, )



if __name__ == "__main__":
  #train classifier and use it to convert rewards in dataset
  import numpy as np
  from os import listdir
  from os.path import isfile, join


  mypath = '/home/rad/Documents/ClothHangProject/episodes/normal_fixed_IK'
  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

  import threading
  import time
  import logging
  threads = list()
  num_threads = 4
  files_split = np.split(np.array(onlyfiles[:-1]), num_threads) #note: there were 458 here so make 456 for good split
  for index in range(num_threads):
      logging.info("Main    : create and start thread %d.", index)
      x = threading.Thread(target=thread_function, args=(files_split[index],))
      threads.append(x)
      time.sleep(0.5)
      x.start()

  for index, thread in enumerate(threads):
      logging.info("Main    : before joining thread %d.", index)
      thread.join()
      logging.info("Main    : thread %d done", index)