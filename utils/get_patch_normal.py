import open3d as o3d
import numpy as np
import os
import time
import random
import argparse

def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed) # fix the random seed

def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) #Convert float64 numpy array of shape (n, 3) to Open3D format.
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def farthest_point_sample(point, patch_size):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    npoint = int(N/patch_size) + 1
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:,:3]
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
    point = point[centroids.astype(np.int32)] # convert float64 to int32 for indexing
    return point

def fps(point, scale_ratio_number):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, _ = point.shape
    npoint = scale_ratio_number + 1
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:,:3]
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
    point = point[centroids.astype(np.int32)] # convert float64 to int32 for indexing
    return point

def knn_patch(pcd_name, patch_size = 2048):
    pcd = o3d.io.read_point_cloud(pcd_name)
    # nomalize pc and set up kdtree
    points = pc_normalize(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    fps_point = farthest_point_sample(points,patch_size)
    point_size = fps_point.shape[0]
    patch_list = []

    for i in  range(point_size):
        [_,idx,dis] = kdtree.search_knn_vector_3d(fps_point[i],patch_size)
        patch_list.append(np.concatenate((np.asarray(points)[idx[:], :], np.asarray(pcd.normals)[idx[:], :]), axis=1))

    return np.array(patch_list)

def multi_scale_patch(pcd_name, scale_ratio1, scale_ratio2, scale_ratio3):
    pcd = o3d.io.read_point_cloud(pcd_name)
    N = len(np.asarray(pcd.points))
    scale_ratio1 = int(N * scale_ratio1)
    scale_ratio2 = int(N * scale_ratio2)
    scale_ratio3 = int(N * scale_ratio3)
    points = pc_normalize(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points)
    scale_1 = fps(points,scale_ratio1)
    scale_2 = fps(points,scale_ratio2)
    scale_3 = fps(points,scale_ratio3)
    return scale_1, scale_2, scale_3





def main(config):
    objs = os.walk(config.path)  
    if not os.path.exists(config.out_path+'/'):
        os.mkdir(config.out_path+'/')
    total_start = time.time()
    for path,dir_list,file_list in objs:  
      for obj in file_list:  
        start = time.time()
        pcd_name = os.path.join(path, obj)
        npy_name = os.path.join(config.out_path,obj.split('.')[0] + '.npy')
        print(pcd_name)
        patch = knn_patch(pcd_name)
        np.save(npy_name,patch)

        end = time.time()
        print('Consuming seconds /s :' + str(end-start))
    total_end = time.time()
    # save the total time into a txt file
    save_name = os.path.join(config.out_path, config.dataset, config.modality, 'time.txt')
    with open(save_name,'w') as f:
        f.write(str(total_end-total_start))
        f.close()


if __name__ == '__main__':
    set_rand_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default = '/LSPCQA/dis/') #path to the file that contain .ply models
    # parser.add_argument('--out_path', type=str, default = '/DATA/zzc/3dqa/wpc2.0/wpc2.0_patch_2048/') # path to the output patches
    parser.add_argument('--out_path', type=str, default = '/LSPCQA/lspcqa_patch_2048_normal/') # path to the output patches
    parser.add_argument('--dataset', type=str, default = 'LSPCQA') # path to the output patches
    parser.add_argument('--modality', type=str, default = '3D_normal') # path to the output patches
    config = parser.parse_args()
    main(config)