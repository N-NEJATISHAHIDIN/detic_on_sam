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
from utils import draw_images#, mask_iou
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
for batch in my_dataloader:
    idx+=1
    depth_img = batch[0][0]
    rgb_img = batch[1][0]
    seg_img = batch[2][0]
    target_obj = seg_img == batch[4][0]

    target_obj_name = batch[-2][0]
    descriptions = batch[-3][0]
    reference_obj_name = batch[-1][0]

    img_org_path = "saved_images/org_image_with_labels/"
    if not os.path.exists(img_org_path):
        os.makedirs(img_org_path)
    # draw_images([depth_img, rgb_img, seg_img, target_obj], [target_obj_name, descriptions, reference_obj_name], idx, img_org_path )
    im = rgb_img.numpy()

    # Run model and show results
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out.get_image()[:, :, ::-1]
    img_path = "saved_images/predicted_masks/"
    texts = [target_obj_name, descriptions, reference_obj_name]
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    # cv2.imwrite(img_path + '{}_{}.png'.format(idx, '_'.join(texts)), out.get_image()[:, :, ::-1])

    pred_classe = [metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]



    # find the most similar word using fast text 
    target_object_embedding = model_fast_text.get_word_vector(target_obj_name)
    predictions_embedding = [model_fast_text.get_word_vector(pred) for pred in pred_classe]
    similarity_fasttext_obj = pred_classe[np.argmax(cosine_similarity([target_object_embedding], predictions_embedding), axis=1)[0]]



    # birt similarity 
    tokens = tokenizer.tokenize(tokenizer.cls_token + " " + target_obj_name + " " + tokenizer.sep_token)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

    # Obtain embeddings
    with torch.no_grad():
        outputs = bert_model(input_ids)

    # Extract the embedding of the [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # # Convert the tensor to a NumPy array
    cls_embedding = cls_embedding.numpy()
    # import pdb; pdb.set_trace()

    input_ids_list = []

    for text in pred_classe:
        tokens = tokenizer.tokenize(tokenizer.cls_token + " " + text + " " + tokenizer.sep_token)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids_list.append(input_ids)


    # Pad the input sequences to the same length (optional)
    max_length = max(len(input_ids) for input_ids in input_ids_list)
    input_ids_list = [input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids)) for input_ids in input_ids_list]

    # Convert the input list to a PyTorch tensor
    input_ids_tensor = torch.tensor(input_ids_list)

    # Obtain embeddings
    with torch.no_grad():
        outputs = bert_model(input_ids_tensor)

    # Extract the embeddings of the [CLS] tokens
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    synonim_object = pred_classe[np.argmax(cosine_similarity(cls_embedding, cls_embeddings), axis=1)[0]]
    print("target : ", target_obj_name, " , synonim class birt : ", synonim_object, " , fast text synonim : ", similarity_fasttext_obj) 
    print("all pred classes : ", pred_classe)
    print("#####################################################################################")
    # import pdb; pdb.set_trace()


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
    # if idx %10 ==0 :
    #     print("accuracy = ", predicted/idx )



