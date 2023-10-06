import logging
logging.basicConfig(filename='zeroshot_detic.log', level=logging.DEBUG)


import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
from tqdm import tqdm


# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
# # clone and install Detic
# !git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
# %cd Detic
# !pip install -r requirements.txt

# Some basic setup:
# Setup detectron2 logger
import sys

sys.path.insert(0, 'Detic/')

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'Detic/third_party/CenterNet2')
# sys.path.insert(0, 'Detic')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

sys.path.insert(0, '/home/negar/secondssd/semantic_abs/baseline/Detic')

# Build the detector and download our pretrained weights
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
print("cfg.MODEL.DEVICE ,", cfg.MODEL.DEVICE)

predictor = DefaultPredictor(cfg)


# Setup the model's vocabulary using build-in datasets

BUILDIN_CLASSIFIER = {
    'lvis': 'Detic/datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'Detic/datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'Detic/datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'Detic/datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)


# load sam_data loader
sys.path.insert(0, '../')
from dataset import Ai2Thour_re_dataset
from utils import draw_images, synonim_using_birt, synonim_using_fast_text, mask_iou #, mask_iou
from torch.utils.data import DataLoader
import csv

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


# Fast text model 
import fasttext
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
# Load the pre-trained model
model_path_fast_text = hf_hub_download(repo_id="facebook/fasttext-be-vectors", filename="model.bin")
model_fast_text = fasttext.load_model(model_path_fast_text)    


from transformers import BertTokenizer, BertModel
# Load pre-trained BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

idx = 0
predicted = 0
fasttext_acc = 0
birt_cls_acc = 0
birt_avg_acc = 0

fasttext_num = 0
birt_cls_num = 0 
birt_avg_num = 0 

for batch in tqdm(my_dataloader):
    depth_img = batch[0][0]
    rgb_img = batch[1][0]
    seg_img = batch[2][0]
    target_obj = seg_img == batch[4][0]

    target_obj_name = batch[-2][0]
    descriptions = batch[-3][0]
    reference_obj_name = batch[-1][0]

    img_org_path = "saved_images/pred_vs_label/"
    if not os.path.exists(img_org_path):
        os.makedirs(img_org_path)
    # draw_images([depth_img, rgb_img, seg_img, target_obj], [target_obj_name, descriptions, reference_obj_name], idx, img_org_path )
    im = rgb_img.numpy()

    # Run model and show results

    outputs = predictor(im)
    import pdb; pdb.set_trace()



    # visualize predicted masks
    # v = Visualizer(im[:, :, ::-1], metadata)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # out.get_image()[:, :, ::-1]

    # img_path = "saved_images/predicted_masks/"
    # texts = [target_obj_name, descriptions, reference_obj_name]
    # if not os.path.exists(img_path):
    #     os.makedirs(img_path)
    # cv2.imwrite(img_path + '{}_{}.png'.format(idx, '_'.join(texts)), out.get_image()[:, :, ::-1])

    # filter pred classes strings (water_jar -> water jar)
    pred_classes_unfilterd = [metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]
    pred_class = []
    for x in outputs["instances"].pred_classes.cpu().tolist():
        parts = " ".join(metadata.thing_classes[x].split("_"))
        parts = " ".join(parts.split("("))
        parts = " ".join(parts.split(")"))
        # Join the parts with spaces
        pred_class.append(parts)



    # find the most similar word using fast text 
    similarity_fasttext_idx = synonim_using_fast_text(model_fast_text, target_obj_name, pred_class)

    # find best synonim using birt cls 
    synonim_object_cls_idx = synonim_using_birt(tokenizer, bert_model, target_obj_name, pred_class, True )

    # find best synonim using birt avrage embeding 
    synonim_object_avg_idx = synonim_using_birt(tokenizer, bert_model, target_obj_name, pred_class )
    # import pdb; pdb.set_trace()

    fasttext_pred_mask = outputs["instances"].pred_masks[similarity_fasttext_idx].unsqueeze(2).cpu()
    birt_cls_pred_mask = outputs["instances"].pred_masks[synonim_object_cls_idx].unsqueeze(2).cpu()
    birt_avg_pred_mask = outputs["instances"].pred_masks[synonim_object_avg_idx].unsqueeze(2).cpu()
    
    fasttext_synonim = pred_class[similarity_fasttext_idx]
    birt_cls_synonim = pred_class[synonim_object_cls_idx]
    birt_avg_synonim = pred_class[synonim_object_avg_idx]
    predicted_synonims = [fasttext_synonim, birt_cls_synonim, birt_avg_synonim, "RGB", "semantic_segmentation", target_obj_name]
    # import pdb; pdb.set_trace()

    if target_obj.unsqueeze(2).sum ==0:
        continue

    idx+=1

    fasttext_acc += mask_iou( fasttext_pred_mask, target_obj.unsqueeze(2))
    birt_cls_acc += mask_iou( birt_cls_pred_mask, target_obj.unsqueeze(2))
    birt_avg_acc += mask_iou( birt_avg_pred_mask, target_obj.unsqueeze(2))

    if mask_iou( fasttext_pred_mask, target_obj.unsqueeze(2)) > 0.25 :
        fasttext_num += 1 
    if mask_iou( birt_cls_pred_mask, target_obj.unsqueeze(2)) > 0.25 :
        birt_cls_num += 1
    if mask_iou( birt_avg_pred_mask, target_obj.unsqueeze(2)) > 0.25 : 
        birt_avg_num += 1 



    # img_path = "saved_images/predicted_masks/"
    # texts = [target_obj_name, descriptions, reference_obj_name]
    # if not os.path.exists(img_path):
    #     os.makedirs(img_path)
    # draw_images([fasttext_pred_mask, birt_cls_pred_mask, birt_avg_pred_mask, rgb_img, seg_img, target_obj.unsqueeze(2)], [target_obj_name, descriptions, reference_obj_name,], idx, img_org_path, predicted_synonims )


    # print("target : ", target_obj_name, " , synonim class birt cls : ", pred_class[similarity_fasttext_idx], ", synonim class birt avg : ", pred_class[synonim_object_cls_idx], " , fast text synonim : ", pred_class[synonim_object_avg_idx]) 
    # print("all pred classes : ", pred_class)
    # print("#####################################################################################")


    # if target_obj_name in [metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]:
    #     predicted +=1
    #     # draw_images([depth_img, rgb_img, seg_img, target_obj], [target_obj_name, descriptions, reference_obj_name], idx, img_org_path )
    #     # cv2.imwrite(img_path + '{}_{}.png'.format(idx, '_'.join(texts)), out.get_image()[:, :, ::-1])

    # # look at the outputs. 
    # # See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    # # print(outputs["instances"].pred_classes) # class index
    # # print([metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]) # class names
    # # print(outputs["instances"].scores)
    # # print(outputs["instances"].pred_boxes)

    if idx %100 ==0 :
        logging.info("####################################### %s #######################################", idx)
        logging.info("mean_iou fasttext = %s", fasttext_acc/idx )
        logging.info("mean_iou birt_cls = %s", birt_cls_acc/idx )
        logging.info("mean_iou birt_avg = %s", birt_avg_acc/idx )

        logging.info("accuracy fasttext = %s", fasttext_num/idx )
        logging.info("accuracy birt_cls = %s", birt_cls_num/idx )
        logging.info("accuracy birt_avg = %s", birt_avg_num/idx )
