import os
import h5py
import csv
import ai2thor.controller
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def read_ai2thour_data(data_root_path, data_id):
    """
    inputs:
        data_root_path : path to the data root directory --> str
        data_id : name of the data instance file. --> str
    outputs: 
        data : the data object --> h5py object
        
    """

    data_instats_path = os.path.join(data_root_path, data_id)
    data_instats_info = h5py.File(data_instats_path,'r')

    return data_instats_info

# def dict_obj_name_id():
#     """
#     dict_keys: the name of the object 
#     dict_values: the id of the object 
#     """
    
#     controller = ai2thor.controller.Controller()
#     event = controller.step('Initialize', gridSize=0.25)
#     dict_obj_ids = {}
#     for obj in event.metadata['objects']:
#         dict_obj_ids [obj['name']] = obj['objectId']
#     import pdb; pdb.set_trace()



def dgenerate_all_gt_bbx(input_image):

    # this fails when we have more than one object of a category in the image
    # generate gt 2D BBX 
    bounding_boxes = {}
    max_bach_obj_id = torch.max(input_image)
    min_bach_obj_id = torch.min(input_image)
    for obj_id in range(min_bach_obj_id, max_bach_obj_id):
        if torch.where(input_image == obj_id)[0].numel() == 0: 
            continue

        import pdb;pdb.set_trace()
        # torch.where(obj_id == )
        x_min = torch.min(torch.where(input_image == obj_id)[0])
        x_max = torch.max(torch.where(input_image == obj_id)[0])
        y_min = torch.min(torch.where(input_image == obj_id)[1])
        y_max = torch.max(torch.where(input_image == obj_id)[1])

        x, y, height, width =  torch.floor((x_max + x_min)/2) ,  torch.floor((y_max + y_min)/2),  x_max - x_min + 4 , y_max - y_min + 4 

       
        bounding_boxes[obj_id] = [x, y, height, width]
    
    return bounding_boxes


def generate_box(segmentation_pixels):
    x_min = torch.min(segmentation_pixels[0])
    x_max = torch.max(segmentation_pixels[0])
    y_min = torch.min(segmentation_pixels[1])
    y_max = torch.max(segmentation_pixels[1])

    x, y, height, width =  torch.floor((x_max + x_min)/2) ,  torch.floor((y_max + y_min)/2),  x_max - x_min + 4 , y_max - y_min + 4 
    return [x, y, height, width] , [x_min, x_max, y_min, y_max]


def draw_bbx_on_image(image, y_min, x_min, width, height):

    #draw on the image 
    fig, ax = plt.subplots()
    # rect = patches.Rectangle((y_min-2, x_min-2), width, height , linewidth=2, edgecolor='g', facecolor='none')
    # ax.add_patch(rect)
    plt.imshow(image)
    plt.savefig('imgs/rW_{}.png'.format(width))
    plt.close()

def draw_images(images, texts, idx, img_path, predicted_synonims=None):
    # import pdb; pdb.set_trace()
    height, width, channels = images[1].shape


    #draw on the image 
    fig, axs = plt.subplots(1, len(images), figsize=(width/100*3, height/100)) 
    fig.suptitle(' '.join(texts), fontsize=30)


    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].axis('off')
        if predicted_synonims:
            axs[i].set_title(f'{predicted_synonims[i]}', fontsize=20)
    # plt.text(5, 5, ' '.join(texts) , fontsize=12, color='red')
    if img_path == None :
        img_path = '../saved_images/'
        if not os.path.exists(img_path):
            os.makedirs(img_path)

    plt.savefig(img_path + '{}_{}.png'.format(idx, '_'.join(texts)))
    plt.close()

def masks_iou_top_k( pred_masks, gt_mask, top_k = -1 ):
    iou_list = []
    for pred_mask in pred_masks[:top_k]:
        overlap = pred_mask.unsqueeze(2) * gt_mask  # Logical AND
        union = (pred_mask.unsqueeze(2) + gt_mask)>0  # Logical OR
        iou = overlap.sum() / float(union.sum())
        if iou == None:
            iou = 0
        iou_list.append(iou)
    return np.max(np.array(iou_list)), np.argmax(np.array(iou_list))#, np.argmax(np.array(iou_list))


def mask_iou( pred_mask, gt_mask):
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask)>0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    if iou == None:
        iou = 0
    return iou


def calculate_iou(boxs, box2):
    """
    Calculate the intersection over union (IOU) between two bounding boxes.
    
    Args:
    - box1: a tuple (x1, y1, x2, y2) representing the coordinates of the top-left and bottom-right corners of the first box.
    - box2: a tuple (x1, y1, x2, y2) representing the coordinates of the top-left and bottom-right corners of the second box.
    
    Returns:
    - The IOU between the two boxes as a float.
    """
    iou = []
    for box1 in boxs:
        # Calculate the coordinates of the intersection box
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        # Calculate the area of intersection
        inter_width = torch.max(torch.tensor(0), x2 - x1)
        inter_height = torch.max(torch.tensor(0), y2 - y1)
        inter_area = inter_width * inter_height
        
        # Calculate the area of each input box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate the area of union
        union_area = box1_area + box2_area - inter_area
        
        # Calculate the IOU and return it
        iou.append( inter_area / union_area if union_area > 0 else 0)

    return iou



def synonim_using_birt(tokenizer, bert_model, target_obj_name, pred_classe , cls = False):

    # birt similarity 
    input_ids_list = []
        

    for text in pred_classe:
        tokens = tokenizer.tokenize(tokenizer.cls_token + " " + text + " " + tokenizer.sep_token)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids_list.append(input_ids)

    # Pad the input sequences to the same length (optional)
    max_length = max(len(input_ids) for input_ids in input_ids_list)
    input_ids_list_final = [input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids)) for input_ids in input_ids_list]
    attention_masks = torch.tensor([[1]*len(input_ids)+ [tokenizer.pad_token_id] * (max_length - len(input_ids)) for input_ids in input_ids_list])

    # Convert the input list to a PyTorch tensor
    input_ids_list_final = torch.tensor(input_ids_list_final)

    # Obtain embeddings
    with torch.no_grad():
        outputs = bert_model(input_ids_list_final, attention_masks)

    if cls :

        # Extract the embeddings of the [CLS] tokens
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    else :
        # Extract all token embeddings
        all_embeddings = outputs.last_hidden_state.numpy()

        # Calculate the average embedding for each sentence
        average_embeddings = np.mean(all_embeddings, axis=1)


    tokens_target = tokenizer.tokenize(tokenizer.cls_token + " " + target_obj_name + " " + tokenizer.sep_token)
    input_ids_target = tokenizer.convert_tokens_to_ids(tokens_target)
    input_ids_target = torch.tensor(input_ids_target).unsqueeze(0)  # Add batch dimension
    input_ids_list_target = torch.tensor([input_ids_target[0].tolist() + [tokenizer.pad_token_id] * (max_length - len(input_ids_target[0].tolist())) ])
    attention_mask =  torch.tensor([input_ids_target[0].shape[0] *[1] + [tokenizer.pad_token_id] * (max_length - input_ids_target[0].shape[0]) ])

    # Obtain embeddings
    with torch.no_grad():
        outputs_target = bert_model(input_ids_list_target,attention_mask)

    if cls :
        # Extract the embedding of the [CLS] token
        cls_embedding = outputs_target.last_hidden_state[:, 0, :].numpy()
        # print(cosine_similarity(cls_embedding, cls_embeddings))
        idx = np.argmax(cosine_similarity(cls_embedding, cls_embeddings), axis=1)[0]
        idx_sorted = np.argsort(cosine_similarity(cls_embedding, cls_embeddings), axis=1)[0]

    else:
        all_embedding = outputs_target.last_hidden_state.numpy()
        # Calculate the average embedding for each sentence
        average_embedding = np.mean(all_embedding, axis=1)
        # print(cosine_similarity(average_embedding, average_embeddings))
        idx = np.argmax(cosine_similarity(average_embedding, average_embeddings), axis=1)[0]
        idx_sorted = np.argsort(cosine_similarity(average_embedding, average_embeddings), axis=1)[0]

    return idx , idx_sorted



def synonim_using_fast_text(model_fast_text, target_obj_name, pred_class):

    # find the most similar word using fast text 
    target_object_embedding = model_fast_text.get_word_vector(target_obj_name)
    predictions_embedding = [model_fast_text.get_word_vector(pred) for pred in pred_class]
    idx = np.argmax(cosine_similarity([target_object_embedding], predictions_embedding), axis=1)[0]
    idx_sorted = np.argsort(cosine_similarity([target_object_embedding], predictions_embedding), axis=1)[0]
    return idx , idx_sorted


def compute_similarity_gpt3_api():
    import openai
    # Replace 'YOUR_API_KEY' with your actual API key
    api_key = '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'

    # Define two noun phrases
    noun_phrase1 = "a cat on a mat"
    noun_phrase2 = "a dog on a rug"


    # Encode the noun phrases using GPT-3
    response = openai.Completion.create(
        engine="text-davinci-002",  # Choose an appropriate GPT-3 engine
        prompt=f"Compare the semantic similarity between the following noun phrases:\n1. {noun_phrase1}\n2. {noun_phrase2}\n",
        max_tokens=1,  # Set the desired max_tokens value
        api_key=api_key
    )

    # Extract the similarity score from the response
    similarity_score = float(response.choices[0].text.strip())

    # Print the similarity score
    print(f"Semantic similarity score: {similarity_score}")


if __name__ == "__main__":
    import pdb; pdb.set_trace()
    