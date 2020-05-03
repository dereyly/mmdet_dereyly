export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7
# ./tools/dist_test.sh configs/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_face_gc_sample_stretch.py /media/HD2/nsergievskiy/models/cascade_rcnn_dconv_c3-c5_r50_fpn_gc_face2/epoch_18.pth 8 --out res.pkl --eval bbox  
python tools/test.py configs/nas_fpn/retinanet_crop640_r50_nasfpn_50e.py /media/dereyly/data/models/COCO/mmdet_pret/retinanet_crop640_r50_nasfpn_50e_20191225-b82d3a86.pth --eval bbox
