# https://github.com/gatsby2016/Augmentation-PyTorch-Transforms
from Augment import myTransforms
from PIL import Image
from matplotlib import pyplot as plt
imagename = 'image_test/Red_Apple.png'
img = Image.open(imagename) # read the image

preprocess = myTransforms.HEDJitter(theta=0.05)
print(preprocess)

HEPerimg = preprocess(img)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(HEPerimg)
plt.show()

