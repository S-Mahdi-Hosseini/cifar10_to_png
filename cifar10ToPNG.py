import pickle
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def unpickle_bytes(file):
  with open(file, 'rb') as fo:
     dict = pickle.load(fo, encoding='bytes')
  return dict

def unpickle_ascii(file):
  with open(file, 'rb') as fo:
     dict = pickle.load(fo, encoding='ASCII')
  return dict

def make_cifar10_png(cifar10_path = 'cifar-10-batches-py' , target_cifar10_path = 'cifar-10-batches-py'):
  '''
  cifar10_path : this is the unzipped file downloaded fron https://www.cs.toronto.edu/~kriz/cifar.html
  target_cifar10_path : this is the path your png image dataset is gonna be saved there!!!
  '''
  
  label_path = os.path.join(cifar10_path, 'batches.meta')
  label = unpickle_ascii(label_path)

  for batch in ('test_batch' , 'data_batch_1' , 'data_batch_2' , 'data_batch_3' , 'data_batch_4' , 'data_batch_5'):
    fpath = os.path.join(cifar10_path, batch)
    d = unpickle_bytes(fpath)
    
    d_decoded = {}
    
    for k, v in d.items():
        d_decoded[k.decode('utf8')] = v
    d = d_decoded
    
    for i , filename in enumerate(tqdm(d['filenames'])):
        folder = os.path.join(target_cifar10_path , 'test' if batch=='test_batch' else 'train', label['label_names'][d['labels'][i]])
        os.makedirs(folder, exist_ok=True)
        img_vector = d['data'][i]
        img = img_vector.reshape((32, 32, 3), order='F').swapaxes(0,1)
        img = Image.fromarray(img , mode = "RGB")
        img.save(os.path.join(folder,filename.decode())) 
