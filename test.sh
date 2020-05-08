export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7
# ./tools/dist_test.sh configs/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_face_gc_sample_stretch.py /media/HD2/nsergievskiy/models/cascade_rcnn_dconv_c3-c5_r50_fpn_gc_face2/epoch_18.pth 8 --out res.pkl --eval bbox  
#python tools/test.py configs/nas_fpn/retinanet_crop640_r50_nasfpn_50e.py /media/dereyly/data/models/COCO/mmdet_pret/retinanet_crop640_r50_nasfpn_50e_20191225-b82d3a86.pth --eval bbox
#python tools/test.py /home/dereyly/progs/mmdets_NEW/mmdet_res2/configs/rpn_r50_fpn_1x.py  /media/dereyly/data/models/COCO/mmdet_pret/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth --eval proposal
#/home/dereyly/progs/mmdets_NEW/mmdet_res2/configs/atss/atss_r50_fpn_1x.py /media/dereyly/data/models/COCO/mmdet_pret/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --eval bbox
#/home/dereyly/progs/mmdets_NEW/mmdet_res2/configs/guided_anchoring/ga_rpn_r50_caffe_fpn_1x.py /media/dereyly/data/models/COCO/mmdet_pret/ga_rpn_r50_caffe_fpn_1x_20190513-95e91886.pth --eval proposal
#/home/dereyly/progs/mmdets_NEW/mmdet_res2/configs/atss/atss_r2_50_fpn_waymo.py /media/dereyly/data/models/waymo/atss_r2_50_3.pth --show