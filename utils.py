import os

def create_if_not_exists():
    if not os.path.exists("tb_logs"):
        os.mkdir("tb_logs")
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

# import numpy as np
# for file in os.listdir('EyeInTheSky/data/gt/'):
#  if file == '.DS_Store':
#      continue
#  image = Image.open('EyeInTheSky/data/gt/{}'.format(file))
#  image_mat = np.array(image)
#  new_image_mat = np.zeros((image_mat.shape[0], image_mat.shape[1]))
#  for i in range(image_mat.shape[0]):
#      for j in range(image_mat.shape[1]):
#          new_image_mat[i,j] = map_dict[tuple(image_mat[i,j,:])]
#  np.save('EyeInTheSky/data/gt/{}.npy'.format(file.split()[0]), new_image_mat)