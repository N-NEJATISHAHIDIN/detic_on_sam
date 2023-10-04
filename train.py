import csv
import numpy as np
import os
import sys 
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import clip
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import functional as F
from torch_geometric.data import Data



from PIL import Image

# from utils import dict_obj_name_id
from dataset import Ai2Thour_re_dataset
# from model import GNN
from utils import draw_bbx_on_image, generate_box
from model import GCN


# load devices
device = "cuda" if torch.cuda.is_available() else "cpu"

# load CLIP models
model, preprocess = clip.load("ViT-B/32", device)
# Load the pre-trained ResNet model
resnet = models.resnet18(pretrained=True).to(device)
resnet.eval()


model_FRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model_FRCNN.eval()

# resnet transform Define a transform to resize the image and convert it to a tensor
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#BBX transfoorm
transform_bbx = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])




# data_root_path = '/scratch/nnejatis/datasets/dataset'
data_root_path = '../dataset'

list_data_ids_path = os.path.join(data_root_path, 'list_data_ids_traning.csv')

# read list data ids csv file
with open(list_data_ids_path, mode='r') as file:
    reader = csv.reader(file)
    list_data_ids = [row for row in reader]

# Create an instance of your dataset
re_dataset = Ai2Thour_re_dataset(data_root_path, list_data_ids)

# Create a DataLoader for your dataset
my_dataloader = DataLoader(re_dataset, batch_size=1, shuffle=True)


#model 

grounding_model = GCN(2664, 128, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = grounding_model(data.x, data.edge_index.to(device))  # Perform a single forward pass.
    import pdb; pdb.set_trace()
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return out

def test():
    model.eval()
    out = grounding_model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    # test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    # return test_acc



for epoch in range(1, 101):


    # Iterate over the DataLoader to get batches of data
    for batch in my_dataloader:
        gt_segmentation_pixels = torch.where(batch[2][0] == batch[4][0])
        # gt_bbx =  generate_box(gt_segmentation_pixels) 
        # print(gt_bbx)


        # # get clip embeding for the sentence tokens 
        # text_description = clip.tokenize(list(batch[5])).to(device)
        # text_target_obj = clip.tokenize(list(batch[6])).to(device)
        # text_refrenced_obj= clip.tokenize(list(batch[7])).to(device)

        # text_description_features = model.encode_text(text_description)
        # text_target_obj_features = model.encode_text(text_target_obj)
        # text_refrenced_obj_features = model.encode_text(text_refrenced_obj)


        # #reshape image and get resnet features
        # img = TF.to_pil_image(batch[1][0].permute(2, 0, 1))
        # transformed_imgs = transform(img)

        # # Add a batch dimension to the tensor
        # tensor = transformed_imgs.unsqueeze(0).to(device)


        # transformed_imgs_bbx = transform_bbx(img).to(device)
        # image_tensor_bbx = transformed_imgs_bbx.unsqueeze(0).to(device)


        # # Pass the tensor through the model to get the features
        # features = resnet(tensor).to(device)



        # # import pdb;pdb.set_trace()

        # # detect 2D bounding boxes
        # # Make a prediction on the image
        # predictions = model_FRCNN(image_tensor_bbx)
        # language_feature = torch.cat([text_description_features, text_target_obj_features, text_refrenced_obj_features],dim=1) 


        # # Extract the bounding box coordinates from the prediction
        # boxes = predictions[0]['boxes']
        # # import pdb;pdb.set_trace()
        # transformed_imgs_bbxs = []
        # for box in boxes:

        #     #compute IOU to fined the gt


        #     y_min = box[0].detach().cpu().numpy()
        #     x_min = box[1].detach().cpu().numpy()
        #     width = (box[2]- box[0]).detach().cpu().numpy()
        #     height = (box[3] - box[1]).detach().cpu().numpy()

        #     croped_tensor = batch[1][0, max(0,int(np.floor(box[1].detach().cpu().numpy()))):min(895,int(np.floor(box[3].detach().cpu().numpy()))+5), 
        #                                     max(0, int(np.floor(box[0].detach().cpu().numpy()))):min(895, int(np.floor(box[2].detach().cpu().numpy())+5))]
        #     croped_tensor_img = TF.to_pil_image(croped_tensor.permute(2, 0, 1))
        #     #reshape image and get resnet features
        #     # import pdb;pdb.set_trace()
        #     transformed_imgs_bbxs.append(torch.cat((resnet(transform(croped_tensor_img).unsqueeze(0).to(device)),language_feature, box.unsqueeze(0)), dim=1))

        #     # draw_bbx_on_image(croped_tensor, y_min, x_min, width, height)
        #     # draw_bbx_on_image(tensor[0].permute(1, 2, 0), y_min, x_min, width, height)
        # # import pdb;pdb.set_trace()


        # num_nodes = len(boxes) # Adjustable number of nodes.
        # # language_feature = torch.cat([text_description_features, text_target_obj_features, text_refrenced_obj_features],dim=1) 
        # x = torch.stack(transformed_imgs_bbxs, dim=0) #, language_feature)



        # adjacency = torch.ones(num_nodes,num_nodes)#- torch.eye(num_nodes)
        # edge_index = torch.cat([torch.where(adjacency.to(torch.int) != 0)[0].unsqueeze(0),torch.where(adjacency.to(torch.int) != 0)[1].unsqueeze(0)], dim=0)
        
        # data = Data(x=x, edge_index=edge_index, edge_attr = features.repeat(edge_index.shape[1],1) )
        # # import pdb; pdb.set_trace()
        # out = grounding_model(data.x, (data.edge_index).to(device))  # Perform a single forward pass.
        # loss = train()
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')




    


    


