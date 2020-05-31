export CUDA_VISIBLE_DEVICES=0,1
./tools/dist_train.sh /home/dereyly/progs/mmdets_NEW/TSD/configs/track/dh_r50_fpn_dcn_ga_waymo_dens_reid.py 2
# python tools/train.py /home/dereyly/progs/mmdet_res2/configs/atss/atss_r2_50_dcn_waymo_full.py