import pickle
import os
import h5py
import csv


# open the pickled file in read binary mode
with open('../dataset/vool_split.pkl', 'rb') as f:
    # load the data from the file
    list_of_ids = pickle.load(f)

list_data_ids = []
#find number of instances of data in each data file
for data_id in list_of_ids["train"]:
    data_root_path = '../dataset'
    data_instats_path = os.path.join(data_root_path, data_id)
    data_instats_info = h5py.File(data_instats_path,'r')
    instance_id = data_instats_info['data']['descriptions']['spatial_relation_name'].shape[0]
    list_data_ids.extend([[data_id, i] for i in range(instance_id)])
    # import pdb; pdb.set_trace()

# creat new csv file for required list_id 
data_root_path = '../dataset'
list_data_ids_path = os.path.join(data_root_path, 'list_data_ids_traning.csv')

with open(list_data_ids_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(list_data_ids)





