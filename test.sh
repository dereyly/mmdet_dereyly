export CUDA_VISIBLE_DEVICES=0  #1,2,3 #0,1,2,3

 # ./tools/dist_test.sh /home/dereyly/progs/my_code/mmdet_tsd/configs/tsd/tsd_rcnn_r101_fpn_dcn_ga_waymo.py /media/dereyly/data_hdd/models/COCO/tsd_r2_101_dcn_ga_waymo_nogt/epoch_7.pth 3 --eval bbox
 # python tools/test.py /home/dereyly/progs/my_code/mmdet_tsd/configs/tsd/tsd_rcnn_r101_fpn_dcn_ga_waymo.py /media/dereyly/data_hdd/models/COCO/tsd_r2_101_dcn_ga_waymo_nogt_val2/epoch_2.pth --eval bbox
 # ./tools/dist_test.sh /home/dereyly/progs/my_code/mmdet_tsd/configs/tsd/tsd_rcnn_r101_fpn_dcn_ga_waymo.py /media/dereyly/data_hdd/models/COCO/tsd_r2_101_dcn_ga_waymo_nogt/epoch_5_v1.pth 3 --eval bbox
# python tools/test.py /home/dereyly/progs/my_code/mmdet_tsd/configs/tsd/tsd_rcnn_r101_fpn_dcn_ga_waymo.py /media/dereyly/data_hdd/models/COCO/tsd_r2_101_dcn_ga_waymo_nogt/epoch_5_v1.pth --eval bbox
# python tools/test.py /home/dereyly/progs/my_code/mmdet_tsd/configs/tsd/tsd_rcnn_r101_fpn_dcn_ga_waymo.py /media/dereyly/data_hdd/models/COCO/tsd_r2_101_dcn_ga_waymo_nogt_val/epoch_10.pth --eval bbox
# python tools/test.py /home/dereyly/progs/my_code/mmdet_tsd/configs/tsd/tsd_rcnn_r101_fpn_dcn_ga_waymo.py /media/dereyly/data_hdd/models/COCO/tsd_r2_101_dcn_ga_waymo_nogt/epoch_5_v1.pth --out tsd_1450_small.pkl
#./tools/dist_test.sh /home/dereyly/progs/my_code/mmdet_tsd/configs/tsd/faster_rcnn_r50_fpn_dcn_ga_waymo.py /media/dereyly/data_hdd/models/waymo/tsd_r2_50_dcn/epoch_1.pth 3 --eval bbox
# python tools/test.py /home/dereyly/progs/my_code/mmdet_tsd/configs/tsd/cascade_r101_fpn_dcn_ga_waymo_nogt.py /media/dereyly/data_hdd/models/COCO/cascade_r2_101_dcn_ga_waymo_nogt_sgd/epoch_6.pth --eval bbox
python tools/test.py /home/dereyly/progs/my_code/TSD/configs/track/dh_r50_fpn_dcn_ga_waymo_dens_reid_fc.py /media/dereyly/data_hdd/models/waymo/denc_reid_fc//epoch_4.pth --eval bbox