
python -u train.py \
--wandbkey "" \ # add your wandb key here otherwize the code will not run successfully
--learning_rate 0.00005 \
--model M3_Unity \
--batch_size  4 \
--database SJTU  \
--data_dir_texture_img /gpfs/work3/0/prjs0839/data/MM-PCQA/sjtu_projections_xm/ \
--data_dir_depth_img /gpfs/work3/0/prjs0839/data/MM-PCQA/sjtu_depth_maps/ \
--data_dir_normal_img /gpfs/work3/0/prjs0839/data/MM-PCQA/sjtu_normal_maps/ \
--data_dir_texture_pc /gpfs/work3/0/prjs0839/data/MM-PCQA/sjtu_patch_2048_color/ \
--data_dir_position_pc /gpfs/work3/0/prjs0839/data/MM-PCQA/sjtu_patch_2048_position/ \
--data_dir_normal_pc /gpfs/work3/0/prjs0839/data/MM-PCQA/sjtu_patch_2048_normal/ \
--loss l2rank \
--num_epochs 100 \
--k_fold_num 9 \
--use_classificaiton 1 \
--use_local 1 \
--method_label E4_with_dep_nor

