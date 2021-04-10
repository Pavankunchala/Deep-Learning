import io
import numpy as np
import PIL
import requests
import torch
import openpifpaf
import cv2

device = torch.device('cpu')
# device = torch.device('cuda')  # if cuda is available

print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)

image_response = requests.get('https://images.pexels.com/photos/233129/pexels-photo-233129.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=500')
pil_im = PIL.Image.open(io.BytesIO(image_response.content)).convert('RGB')
im = np.asarray(pil_im)
im = im[...,::-1]



with openpifpaf.show.image_canvas(im) as ax:
    pass
cv2.imshow('image',im)
cv2.waitKey(0)




net_cpu, _ = openpifpaf.network.Factory(checkpoint='shufflenetv2k16', download_progress=False).factory()
net = net_cpu.to(device)
decoder = openpifpaf.decoder.factory([hn.meta for hn in net_cpu.head_nets])

preprocess = openpifpaf.transforms.Compose([
    openpifpaf.transforms.NormalizeAnnotations(),
    openpifpaf.transforms.CenterPadTight(16),
    openpifpaf.transforms.EVAL_TRANSFORM,
])
data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)

loader = torch.utils.data.DataLoader(
    data, batch_size=1, pin_memory=True, 
    collate_fn=openpifpaf.datasets.collate_images_anns_meta)

annotation_painter = openpifpaf.show.AnnotationPainter()

for images_batch, _, __ in loader:
    predictions = decoder.batch(net, images_batch, device=device)[0]
    with openpifpaf.show.image_canvas(im) as ax:
        
        annotation_painter.annotations(ax, predictions)
        
        
        
        
cv2.imshow('ax',ax)
cv2.waitKey(0)
        
cv2.destroyAllWindows()


