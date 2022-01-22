'''from train.py import SegmentPeople
from pylab import imshow
import numpy as np
import cv2
import albumentations as albu'''
import torch
print(torch.version)
print(torch.cuda.device_count())


'''model = SegmentPeople.load_from_checkpoint("example.ckpt")
model.eval()
image = load_rgb("eml7crxnxftrimsmolwjegqcrp4.jpeg")
transform = albu.Compose([albu.Normalize(p=1)], p=1)
padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
x = transform(image=padded_image)["image"]
x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
with torch.no_grad():
  prediction = model(x)[0][0]
mask = (prediction > 0).cpu().numpy().astype(np.uint8)
mask = unpad(mask, pads)
imshow(mask)'''
