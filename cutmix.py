import torch
import copy

class CutMix :
    def __init__(self, img_height, img_width):
        self.h = img_height
        self.w = img_width 
        self.gen = torch.distributions.beta.Beta(1,1)
        
    def __call__(self, a_image, a_label):
        batch_size = a_image.shape[0]
        rand = torch.randperm(batch_size)
        b_image = a_image[rand]
        b_label = a_label[rand]
        
        y = torch.randint(self.h, (1,))[0]
        x = torch.randint(self.w, (1,))[0]

        r = self.gen.sample()
        h = (self.h * torch.sqrt(1-r)).int()
        w = (self.w * torch.sqrt(1-r)).int()
        c_image = copy.deepcopy(a_image)
        c_image[: , : , y:y+h , x:x+w] = b_image[: , : ,y:y+h , x:x+w]

        c_label = a_label * r + b_label * (1-r)
        return c_image, c_label