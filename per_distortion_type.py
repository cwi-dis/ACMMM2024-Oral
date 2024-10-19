import os, argparse, time
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from models.multimodal_pcqa_per_distortion import MM_PCQAnet
from utils.MultimodalDataset_test import MMDataset
from utils.loss import L2RankLoss
import scipy
import wandb
import datetime
import pandas as pd

def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # fix the random seed


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


# this function is used to put pyperparameters and the path dir
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--gpu", help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument(
        "--num_epochs", help="Maximum number of training epochs.", default=30, type=int
    )
    parser.add_argument("--batch_size", help="Batch size.", default=8, type=int)
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="learning rate in training"
    )
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="decay rate")
    parser.add_argument("--model", default="", type=str)
    parser.add_argument(
        "--data_dir_texture_img", default="", type=str, help="path to the images"
    )
    parser.add_argument(
        "--data_dir_depth_img", default="", type=str, help="path to the depth images"
    )
    parser.add_argument(
        "--data_dir_normal_img", default="", type=str, help="path to the normal images"
    )
    parser.add_argument(
        "--data_dir_texture_pc",
        default="",
        type=str,
        help="path to the texture patches",
    )
    parser.add_argument(
        "--data_dir_normal_pc",
        default="",
        type=str,
        help="path to the normal of patches",
    )
    parser.add_argument(
        "--data_dir_position_pc",
        default="",
        type=str,
        help="path to the xyz position of patches",
    )
    parser.add_argument(
        "--patch_length_read", default=6, type=int, help="number of the using patches"
    )
    parser.add_argument(
        "--img_length_read", default=6, type=int, help="number of the using images"
    )
    parser.add_argument("--loss", default="l2rank", type=str)
    parser.add_argument("--database", default="SJTU", type=str)
    parser.add_argument(
        "--k_fold_num",
        default=9,
        type=int,
        help="9 for the SJTU-PCQA, 5 for the WPC, 4 for the WPC2.0",
    )
    parser.add_argument(
        "--use_classificaiton",
        default=1,
        help="if use classification, 1 or 0",
        type=int,
    )
    parser.add_argument("--use_local", help="", default=1, type=int)
    parser.add_argument("--img_inplanes", help="", default=2048, type=int)
    parser.add_argument("--pc_inplanes", help="", default=1024, type=int)
    parser.add_argument("--cma_planes", help="", default=1024, type=int)
    parser.add_argument(
        "--method_label",
        help="Description of the method of the model",
        default="",
        type=str,
    )
    parser.add_argument(
        "--modality",
        help="Description of the modality input of the model",
        default="both",
        type=str,
    )
    parser.add_argument(
        "--attention",
        help="Default is to use the attention",
        default=1, type=int,
    )
    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path, _num_class, args):
    # Replace this function with your actual checkpoint loading logic
    # Example: loading a PyTorch model checkpoint
    # Define the model
    model = MM_PCQAnet(num_classes=_num_class, args=args)
    print("Using model: MM-PCQA")
     # load the saved checkpoints to test
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint,strict=False) 
        model = model.to(device)  # moves the model to the device.
        print("Load the model successfully")
    else:
        raise
    return model


if __name__ == "__main__":
    print(
        "*******************************test set************************************************"
    )
    
    #set the args
    args = parse_args()
    gpu = args.gpu
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    database = args.database
    modality = args.modality
    attention = args.attention
    patch_length_read = args.patch_length_read
    img_length_read = args.img_length_read
    data_dir_texture_img = args.data_dir_texture_img
    data_dir_depth = args.data_dir_depth_img
    data_dir_normal = args.data_dir_normal_img
    data_dir_texture_pc = args.data_dir_texture_pc
    data_dir_position_pc = args.data_dir_position_pc
    data_dir_normal_pc = args.data_dir_normal_pc
    if args.database == "SJTU":
            _num_class = 7
            num_folds = 9
    elif args.database == "WPC":
            _num_class = 5
            num_folds = 5
    elif args.database == "BASICS":
            _num_class = 4
            num_folds = 1
    elif args.database == "MJPCCD":
            _num_class = 3
            num_folds = 6
    else:
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    for fold in range(1, num_folds + 1):
        k_fold_id = fold #xm: k is based on which checkpoint to load
        if database == "SJTU":
            train_filename_list = (
                "csvfiles/sjtu_data_info/train_" + str(k_fold_id) + ".csv"
            )
            test_filename_list = (
                "csvfiles/sjtu_data_info/test_" + str(k_fold_id) + ".csv"
            )
        elif database == "WPC":
            train_filename_list = (
                "csvfiles/wpc_data_info/train_" + str(k_fold_id) + ".csv"
            )
            test_filename_list = (
                "csvfiles/wpc_data_info/test_" + str(k_fold_id) + ".csv"
            )
        elif database == "BASICS":
            train_filename_list = (
                "csvfiles/basics_data_info/train_" + str(k_fold_id) + ".csv"
            )
            test_filename_list = (
                "csvfiles/basics_data_info/test_" + str(k_fold_id) + ".csv"
            )
        elif database == "MJPCCD":
            train_filename_list = (
                "csvfiles/mjpccd_data_info/train_" + str(k_fold_id) + ".csv"
            )
            test_filename_list = (
                "csvfiles/mjpccd_data_info/test_" + str(k_fold_id) + ".csv"
            )
        checkpoint_name = f"{database}_Fold_{fold}Dis_1Loc_1Epo_100BS_4Method_E4_with_dep_norModality_bothAttention_1_best_model.pth"
        checkpoint_file = os.path.join("/home/xuemei/metrics/local-global-pcqa/ckpts/Main_result/",database, checkpoint_name)
        # checkpoint_file = os.path.join("ckpts/Main_Result/",database, checkpoint_name)
        
        model = load_checkpoint(checkpoint_file, _num_class, args)
        if gpu==True:
            print("Using GPU: {} ".format(gpu))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if True:
            transformations_train = transforms.Compose(
                [
                    # transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            transformations_test = transforms.Compose(
                [
                    # transforms.CenterCrop(224),  # xm: wht test is centercrop?
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        test_dataset = MMDataset(
            data_dir_texture=data_dir_texture_img,
            data_dir_depth=data_dir_depth,
            data_dir_normal=data_dir_normal,
            data_dir_texture_pc=data_dir_texture_pc,
            data_dir_position_pc=data_dir_position_pc,
            data_dir_normal_pc=data_dir_normal_pc,
            datainfo_path=test_filename_list,
            transform=transformations_test,
            is_train=False,
            )

        test_loader = torch.utils.data.DataLoader(
                    dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8
                )
        n_test = len(test_dataset)
        print("Test dataset size: {}".format(n_test))       

        model.eval()
        y_output = np.zeros(n_test)
        y_test = np.zeros(n_test)
        # distortion classification accuracy
        distortion_label = np.zeros(n_test)
        distortion_output = np.zeros(n_test)
        distortion_test = np.zeros(n_test)
        # Rank the values based on their normalized values
        ranking_all_attention = np.zeros((12,n_test))
        ranking_geo_tex_attention = np.zeros((6,n_test))
        value_all_attention = np.zeros((12,n_test))
        value_geo_tex_attention = np.zeros((6,n_test))
        name_list = []

        with torch.no_grad():
            for i, (
                img_name,
                tex_imgs,
                dep_imgs,
                nor_imgs,
                tex_pcs,
                nor_pcs,
                pos_pcs,
                mos,
                dis,
            ) in enumerate(test_loader):
                if args.database == "SJTU":
                    mos = mos
                elif args.database == "WPC":
                    mos = mos / 10
                elif args.database == "BASICS":
                    mos = mos 
                elif args.database == "MJPCCD":
                    mos = mos 
                else:
                    raise
                tex_imgs = tex_imgs.to(device)
                dep_imgs = dep_imgs.to(device)
                nor_imgs = nor_imgs.to(device)
                tex_pcs = torch.Tensor(tex_pcs.float())
                tex_pcs = tex_pcs.to(device)
                nor_pcs = torch.Tensor(nor_pcs.float())
                nor_pcs = nor_pcs.to(device)
                pos_pcs = torch.Tensor(pos_pcs.float())
                pos_pcs = pos_pcs.to(device)
                y_test[i] = mos.item()
                distortion_test[i] = dis.item()
                (_, _, fusion_output_local_global, output_local, output_global, _, _, _, _, \
                tex2D_geo2D_attention, \
                tex2D_geo2D_global_attention, \
                geo2D_tex2D_attention, \
                geo2D_tex2D_global_attention, \
                tex3D_geo3D_attention, \
                tex3D_geo3D_global_attention, \
                geo3D_tex3D_attention, \
                geo3D_tex3D_global_attention,\
                tex2D_tex3D_attention,\
                tex2D_tex3D_global_attention, \
                tex3D_tex2D_attention, \
                tex3D_tex2D_global_attention, \
                tex2D_geo3D_attention, \
                tex2D_geo3D_global_attention,\
                geo3D_tex2D_attention,\
                geo3D_tex2D_global_attention,\
                geo2D_tex3D_attention, \
                geo2D_tex3D_global_attention, \
                tex3D_geo2D_attention, \
                tex3D_geo2D_global_attention, \
                geo2D_geo3D_attention, \
                geo2D_geo3D_global_attention, \
                geo3D_geo2D_attention, \
                geo3D_geo2D_global_attention)= model(
                    tex_imgs, dep_imgs, nor_imgs, tex_pcs, nor_pcs, pos_pcs
                )  #xm: I need to re-write the model to get the feature map instead of the mos_output and dts_output
                    #should I also report the performance for each fold but its already in the log file.
                    # compute the similarity between the last second feature map and the last feature map
                    # gradient backpropagation to visualize which part of the attention feature map is important per distortion type
                # load model once
                # torch.autograd.backward(loss_total)
                # X1.grad
                # print(pc_texture.grad.shape)  # (B, 1024, 6)->(B,1024)
                # visualize the gradient,based on pc_texture.grad
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                local2all = cos(output_local,fusion_output_local_global)
                global2all = cos(output_global,fusion_output_local_global)  # why the two vectors get the same similarity? both equal to 1
                tex2D_geo2D2all = cos(tex2D_geo2D_attention.mean(dim=1), fusion_output_local_global)
                tex2D_geo2D_global2all = cos(tex2D_geo2D_global_attention, fusion_output_local_global)
                geo2D_tex2D2all = cos(geo2D_tex2D_attention.mean(dim=1), fusion_output_local_global)
                geo2D_tex2D_global2all = cos(geo2D_tex2D_global_attention, fusion_output_local_global)
                tex3D_geo3D2all = cos(tex3D_geo3D_attention.mean(dim=1), fusion_output_local_global)
                tex3D_geo3D_global2all = cos(tex3D_geo3D_global_attention, fusion_output_local_global)
                geo3D_tex3D2all = cos(geo3D_tex3D_attention.mean(dim=1), fusion_output_local_global)
                geo3D_tex3D_global2all = cos(geo3D_tex3D_global_attention, fusion_output_local_global)
                tex2D_tex3D2all = cos(tex2D_tex3D_attention.mean(dim=1), fusion_output_local_global)
                tex2D_tex3D_global2all = cos(tex2D_tex3D_global_attention, fusion_output_local_global)
                tex3D_tex2D2all = cos(tex3D_tex2D_attention.mean(dim=1), fusion_output_local_global)
                tex3D_tex2D_global2all = cos(tex3D_tex2D_global_attention, fusion_output_local_global)
                tex2D_geo3D2all = cos(tex2D_geo3D_attention.mean(dim=1), fusion_output_local_global)
                tex2D_geo3D_global2all = cos(tex2D_geo3D_global_attention, fusion_output_local_global)
                geo3D_tex2D2all = cos(geo3D_tex2D_attention.mean(dim=1), fusion_output_local_global)
                geo3D_tex2D_global2all = cos(geo3D_tex2D_global_attention, fusion_output_local_global)
                geo2D_tex3D2all = cos(geo2D_tex3D_attention.mean(dim=1), fusion_output_local_global)
                geo2D_tex3D_global2all = cos(geo2D_tex3D_global_attention, fusion_output_local_global)
                tex3D_geo2D2all = cos(tex3D_geo2D_attention.mean(dim=1), fusion_output_local_global)
                tex3D_geo2D_global2all = cos(tex3D_geo2D_global_attention, fusion_output_local_global)
                geo2D_geo3D2all = cos(geo2D_geo3D_attention.mean(dim=1), fusion_output_local_global)
                geo2D_geo3D_global2all = cos(geo2D_geo3D_global_attention, fusion_output_local_global)
                geo3D_geo2D2all = cos(geo3D_geo2D_attention.mean(dim=1), fusion_output_local_global)
                geo3D_geo2D_global2all = cos(geo3D_geo2D_global_attention, fusion_output_local_global)
                attention_list = [
                    tex2D_geo2D2all, geo2D_tex2D2all,\
                    tex3D_geo3D2all, geo3D_tex3D2all,\
                    tex2D_tex3D2all, tex3D_tex2D2all,\
                    tex2D_geo3D2all, geo3D_tex2D2all,\
                    geo2D_tex3D2all, tex3D_geo2D2all,\
                    geo2D_geo3D2all, geo3D_geo2D2all,\
                                    ]
                # Convert the list to a NumPy array
                attention_array = np.array(torch.tensor(attention_list, device='cpu'))
                # Normalize the values to the range (0, 1)
                min_val = attention_array.min().item()
                max_val = attention_array.max().item()
                normalized_attentions = (attention_array - min_val) / (max_val - min_val)
                attention_list_geo_tex = [
                    tex2D_geo2D2all + geo2D_tex2D2all,\
                    tex3D_geo3D2all + geo3D_tex3D2all,\
                    tex2D_tex3D2all + tex3D_tex2D2all,\
                    tex2D_geo3D2all + geo3D_tex2D2all,\
                    geo2D_tex3D2all + tex3D_geo2D2all,\
                    geo2D_geo3D2all + geo3D_geo2D2all,\
                                        ]
                # Convert the list to a NumPy array
                attention_array_geo_tex = np.array(torch.tensor(attention_list_geo_tex, device='cpu'))
                min_val = attention_array_geo_tex.min().item()
                max_val = attention_array_geo_tex.max().item()
                normalized_attentions_geo_tex = (attention_array_geo_tex - min_val) / (max_val - min_val)
                # sorted_attention_indices = np.argsort(normalized_attentions)[::-1]
                # sorted_attention_indices_geo_tex = np.argsort(normalized_attentions_geo_tex,descending=True)
                sorted_attention_indices = scipy.stats.rankdata(-normalized_attentions)
                sorted_attention_indices_geo_tex = scipy.stats.rankdata(-normalized_attentions_geo_tex)
                
                # # Sort the normalized values
                ranking_all_attention[:,i] = sorted_attention_indices
                ranking_geo_tex_attention[:,i] = sorted_attention_indices_geo_tex
                name_list.append(img_name)
                distortion_label[i] = dis.item()
                # # save the raw data
                # value_all_attention[:,i] = normalized_attentions
                # value_geo_tex_attention[:,i] = normalized_attentions_geo_tex
            # compute the average ranking of the attention on all the test set
            average_ranking_all_attention = np.mean(ranking_all_attention,axis=1)
            average_ranking_geo_tex_attention = np.mean(ranking_geo_tex_attention,axis=1)
            # Names of the variables
            attention_geo_tex_names = ['tex2D_geo2D2all', 'tex3D_geo3D2all', 'tex2D_tex3D2all','tex2D_geo3D2all', 'geo2D_tex3D2all', 'geo2D_geo3D2all']
            # Concatenate variable names as the first row
            attention_geo_tex_with_header = np.vstack((attention_geo_tex_names, average_ranking_geo_tex_attention))
            # compute the average ranking of the attention by distortion type
            # Combine the data into a structured NumPy array
            num_samples = len(name_list)
            num_features = ranking_geo_tex_attention.shape[0]
            # Initialize numpy arrays
            data_array = np.zeros((num_features,num_samples))
            label_array = np.zeros(num_samples, dtype=int)
            name_array = np.empty(num_samples, dtype=object)
            for i in range(num_samples):
                # Update data_array, label_array, and name_array in each iteration
                data_array[:, i] = ranking_geo_tex_attention[:,i]
                label_array[i] = distortion_label[i] 
                name_array[i] = name_list[i]

            data = np.hstack((data_array.T, name_array.reshape(num_samples,1), label_array.reshape(num_samples,1)))
            print(data)
            
            df = pd.DataFrame(data, columns=['F1','F2','F3','F4','F5','F6', 'name', 'Distortion'])
            directory = "analysis/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            # mean_per_distortion_type = df.groupby('Distortion', as_index=False)['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
            raw_data_save_name = directory + args.database + "_Fold_" + str(k_fold_id) + "_raw_data_ranking_geo_tex.csv"
            df.to_csv(raw_data_save_name, float_format='%.3f')
    
            # Grouping by 'Column1' and computing the average of 'F1', 'F2', 'F3', 'F4', 'F5', 'F6'
            mean_per_distortion_type = df.groupby('Distortion', as_index=False)['F1', 'F2', 'F3', 'F4', 'F5', 'F6'].mean()        
            # Convert the mean values to a NumPy array
            mean_per_distortion_type = np.array(mean_per_distortion_type)
            # Names of the variables
            attention_geo_tex_names = ['tex2D_geo2D2all', 'tex3D_geo3D2all', 'tex2D_tex3D2all','tex2D_geo3D2all', 'geo2D_tex3D2all', 'geo2D_geo3D2all']
            # Concatenate variable names as the first row
            attention_geo_tex_with_header = np.vstack((attention_geo_tex_names, mean_per_distortion_type[:,1:].astype(float), average_ranking_geo_tex_attention.astype(float)))
            # Save to csv file
            df_all = pd.DataFrame(attention_geo_tex_with_header[1:,:], columns=attention_geo_tex_names)
            filename = directory + args.database + "_Fold_" + str(k_fold_id) + "_average_ranking_attention_geo_tex.csv"
            df_all.to_csv(filename, float_format='%.3f')
   
            

           
            



         
       