<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=0,1
./tools/dist_train.sh /home/dereyly/progs/mmdets_NEW/TSD/configs/track/dh_r50_fpn_dcn_ga_waymo_dens_reid_fc.py 2
=======
export CUDA_VISIBLE_DEVICES=1,2,3
./tools/dist_train.sh configs/track/dh_r50_fpn_dcn_ga_waymo_dens_reid_fc.py 3
>>>>>>> 93f361151eedcf14d433eea4f9bed254113fe3fb
# python tools/train.py /home/dereyly/progs/mmdet_res2/configs/atss/atss_r2_50_dcn_waymo_full.py
#python tools/train.py /home/dereyly/progs/mmdets_NEW/TSD/configs/track/dh_r50_fpn_dcn_ga_waymo_dens_reid_fc.py