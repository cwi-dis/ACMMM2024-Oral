import open3d as o3d
import numpy as np
import os
import time
import argparse
import pandas as pd


def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, patch_size):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    npoint = int(N / patch_size) + 1
    if N < npoint:
        idxes = np.hstack(
            (np.tile(np.arange(N), npoint // N), np.random.randint(N, size=npoint % N))
        )
        return point[idxes, :]

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def knn_patch(pcd_name, patch_size=2048):
    pcd = o3d.io.read_point_cloud(pcd_name)
    # nomalize pc and set up kdtree
    points = pc_normalize(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    fps_point = farthest_point_sample(points, patch_size)

    point_size = fps_point.shape[0]

    patch_list = []

    for i in range(point_size):
        [_, idx, dis] = kdtree.search_knn_vector_3d(fps_point[i], patch_size)
        # print(pc_normalize(np.asarray(point)[idx[1:], :]))
        patch_list.append(np.asarray(points)[idx[:], :])

    # visualize(all_point(np.array(patch_list)))
    # visualize(point)
    return np.array(patch_list)




# for SJTU Dataset with each distortion type has 6 distortion levels
# Function to extract distortion degree from the stimulus name
def extract_distortion_degree(stimulus):
    distortion_number = int(stimulus.split('_')[-1])
    return distortion_number % 6

# for WPC Dataset with each distortion type has various distortion levels, 
# based on the distortion levels in SJTU dataset, I divide the distortion levels into 6 levels and do a linear mapping
# Function to extract distortion degree from the stimulus name
def normalize_01(list):
    min_value = min(list)
    max_value = max(list)
    normalized_list = []
    for i in range(len(list)):
        normalized_list.append((list[i] - min_value) / (max_value - min_value))
    return normalized_list 




def main(config):
    if config.dataset == "sjtu":
        start = time.time()
        df = pd.read_csv(config.path)
        stimulis = df["stimulus"]
        test_distortion_degrees = []
        train_distortion_degrees = []
        # Display the updated DataFrame
        print(df)
        for index, name in enumerate(stimulis):
            if "redandblack" in name:
                # get the corredponding stimulus name based on the index
                distortion_degree = extract_distortion_degree(name)
                normalized_distortion_degree = distortion_degree / 5.0  # Normalize to [0, 1]
                test_distortion_degrees.append(normalized_distortion_degree)
        train_distortion_degrees = test_distortion_degrees * 8
        # Create a new DataFrame from the array
        df_new_test = pd.DataFrame({"DD": test_distortion_degrees})
        df_new_train = pd.DataFrame({"DD": train_distortion_degrees})
        for i_test in range(9):
            test_file_name = "test_" + str(i_test + 1) + ".csv"
            train_file_name = "train_" + str(i_test + 1) + ".csv"
            csv_file_test = config.out_path + test_file_name
            csv_file_train = config.out_path + train_file_name
            data_test = pd.read_csv(csv_file_test)
            data_train = pd.read_csv(csv_file_train)
            # Append the new DataFrame to the existing one
            df_combined_test = pd.concat([data_test, df_new_test], axis=1)
            df_combined_train = pd.concat([data_train, df_new_train], axis=1)
            df_combined_test.to_csv(csv_file_test, index=False)
            print("Data appended to the Excel file test.")
            df_combined_train.to_csv(csv_file_train, index=False)
            print("Data appended to the Excel file train.")
        end = time.time()
        print("Consuming seconds /s :" + str(end - start))
    elif config.dataset == "wpc":
        start = time.time()
        df = pd.read_csv(config.path)
        stimulis = df["name"]
        # get all the rows that contains "banana" from the stimulus list
        single_stimuli = df[df["name"].str.contains("banana")]
        # group the stimulus by "DT" and get the DT labels and count the number of each DT
        group_stimuli = single_stimuli.groupby("DT")
        # get the number of each DT
        group_stimuli_count = group_stimuli.count()
        # get the DT labels
        group_stimuli_labels = group_stimuli_count.index
        # get the number of each DT
        group_stimuli_count = group_stimuli_count.values
        # for each DT, get the number of each distortion level
        normalized_distortion_degrees = []
        for i in range(len(group_stimuli_labels)):
            DT_label = group_stimuli_labels[i]
            if DT_label == 0 or DT_label == 1:
                # geometry = [1,2,3]
                # texture = [1,2,3,4]
                distortion_degree = [1,2,3,2,4,6,3,6,9]
            elif DT_label == 2:
                distortion_degree = [1,2,3]
            elif DT_label == 3:
                distortion_degree = [1,2,3,4]
            elif DT_label == 4:
                distortion_degree = [1,2,3,4,2,4,6,8,3,6,9,12]
            # normalize the distortion degree to [0,1]
            normalized_distortion_degree = normalize_01(distortion_degree)
            normalized_distortion_degrees.append(normalized_distortion_degree)
        # flatten the list
        normalized_distortion_degrees = [j for i in normalized_distortion_degrees for j in i]     
        train_distortion_degrees = normalized_distortion_degrees * 16
        test_distortion_degrees = normalized_distortion_degrees * 4
        for i_test in range(5):
            test_file_name = "test_" + str(i_test + 1) + ".csv"
            train_file_name = "train_" + str(i_test + 1) + ".csv"
            csv_file_test = config.out_path + test_file_name
            csv_file_train = config.out_path + train_file_name
            data_test = pd.read_csv(csv_file_test)
            data_train = pd.read_csv(csv_file_train)
            # Create a new DataFrame from the array
            df_new_test = pd.DataFrame({"DD": test_distortion_degrees})
            df_new_train = pd.DataFrame({"DD": train_distortion_degrees})
            # Append the new DataFrame to the existing one
            df_combined_test = pd.concat([data_test, df_new_test], axis=1)
            df_combined_train = pd.concat([data_train, df_new_train], axis=1)
            df_combined_test.to_csv(csv_file_test, index=False)
            print("Data appended to the Excel file test.")
            df_combined_train.to_csv(csv_file_train, index=False)
            print("Data appended to the Excel file train.")
        end = time.time()
        print("Consuming seconds /s :" + str(end - start))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        default="./csvfiles/wpc_data_info/test_1.csv",
    )  # path to the file that contain .ply models
    # for WPC dataset, directly use the path to the csv file: "C:\Xuemei\2024IJCAI\csvfiles\wpc_data_info\test_1.csv"
    parser.add_argument(
        "--out_path",
        type=str,
        default="./csvfiles/wpc_data_info/",
    )  # path to the output patches
    parser.add_argument(
        "--dataset",
        type=str,
        default="wpc",
    )  # path to the output patches
    config = parser.parse_args()
    main(config)
