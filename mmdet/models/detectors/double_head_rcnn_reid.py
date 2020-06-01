import torch

from mmdet.core import bbox2roi, build_assigner, build_sampler
from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class DoubleHeadReID(TwoStageDetector):

    def __init__(self, reg_roi_scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.reg_roi_scale_factor = reg_roi_scale_factor
        self.dens_coef=0.2
        self.embed_coef_id = 0.8
        self.embed_coef_cls = 0.2
        self.num_pairs = 50
        self.criterion_emb=torch.nn.CosineEmbeddingLoss(margin=0.3)
        self.criterion_dens=torch.nn.CrossEntropyLoss()

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(device=img.device)
        # bbox head
        rois = bbox2roi([proposals])
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)
        outs += (cls_score, bbox_pred)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            dens_targets = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                gt_assign = bbox_assigner.assign( gt_bboxes[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                ass_stats=assign_result.gt_inds[assign_result.gt_inds>0]
                dens_targets.append(gt_assign.max_overlaps.clone())
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    # density=gt_assign.max_overlaps,
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_cls_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            bbox_reg_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs],
                rois,
                roi_scale_factor=self.reg_roi_scale_factor)
            if self.with_shared_head:
                #bbox_cls_feats = self.shared_head(bbox_cls_feats)
                bbox_reg_feats = self.shared_head(bbox_reg_feats)
            cls_score, bbox_pred, embed, dens, dens_bin, cls_v2 = self.bbox_head(bbox_cls_feats,
                                                  bbox_reg_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            # tmp2=torch.nonzero(bbox_targets[0])
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            if 1:
                loss_boxx_tmp=self.bbox_head.loss(cls_v2, bbox_pred,
                                                *bbox_targets)
                loss_bbox['loss_cls_v2']=0.05*loss_boxx_tmp['loss_cls']
            if 1:
                # embed=embed.to(torch.device('cuda:0'))
                loss_dens=torch.cuda.FloatTensor([0],device=dens.device) #.to(torch.device('cuda:0'))
                loss_dens2 = torch.cuda.FloatTensor([0],device=dens_bin.device)
                dens=dens.view(-1)
                dens_target=torch.zeros_like(dens)
                dim=sampling_results[0].bboxes.shape[0]
                num_pos=0
                for i in range(num_imgs):
                    npos=sampling_results[i].pos_assigned_gt_inds.shape[0]
                    dens_t = dens_targets[i][sampling_results[i].pos_assigned_gt_inds]
                    # dens_target[i*dim:i*dim+npos]=dens_t
                    # num_pos+=npos
                    loss_dens = torch.abs(dens[i*dim:i*dim+npos] - dens_t)
                    loss_dens += self.dens_coef*loss_dens.sum() / num_imgs / npos
                    loss_dens2 += self.criterion_dens(dens_bin[i*dim:i*dim+npos],(dens_t>0.1).type(torch.long))
                # loss_dens = torch.abs(dens - dens_target)
                # loss_dens = self.dens_coef*loss_dens.sum() / num_imgs / num_pos
                loss_bbox['loss_dens'] = loss_dens
                loss_bbox['loss_dens2'] = loss_dens2
            if 1:
                dim = sampling_results[0].bboxes.shape[0]
                loss_emb_id = torch.cuda.FloatTensor([0], device=embed.device)#.to(torch.device('cuda:0'))
                loss_emb_cls = torch.cuda.FloatTensor([0], device=embed.device)#.to(torch.device('cuda:0'))
                for i in range(num_imgs):
                    pos = sampling_results[i].pos_assigned_gt_inds
                    npos = pos.shape[0]
                    if npos<4:
                        continue

                    ids1 = torch.randint(0,npos,(self.num_pairs,))
                    # pair_pos=[]
                    # pair_neg=[]
                    pairs=[]
                    pairs_lbl=[]
                    for j in range(self.num_pairs):
                        id1 = ids1[j]
                        # weight = torch.ones_like(pos)
                        # weight[id1] = 0
                        # pos_l =pos[weight]
                        ix1=(pos==pos[id1]).nonzero()
                        ix1=ix1[ix1!=id1]
                        if len(ix1)>0:
                            ixx=torch.randint(0, ix1.shape[0], (1,))[0]
                            ix1c=ix1[ixx]
                            pairs.append([id1,ix1c])
                            pairs_lbl.append(1)

                        ix2=(pos!=pos[id1]).nonzero()
                        if len(ix2) > 0:
                            ixx = torch.randint(0, ix2.shape[0], (1,))[0]
                            ix2c = ix2[ixx]
                            pairs.append([id1, ix2c])
                            pairs_lbl.append(-1)
                    pairs_t1 = torch.tensor(pairs,dtype=torch.long,device=embed.device)
                    pairs_lbl_t1 = torch.tensor(pairs_lbl,dtype=torch.long,device=embed.device)
                    if (pairs_lbl_t1==1).sum()>5 and (pairs_lbl_t1==-1).sum()>5:
                        loss_emb_id+= self.criterion_emb(embed[pairs_t1[:,0]],embed[pairs_t1[:,1]],pairs_lbl_t1)

                    pos_lbl = sampling_results[i].pos_gt_labels
                    pairs = []
                    pairs_lbl = []
                    for j in range(self.num_pairs):
                        id1 = ids1[j]
                        ix1 = (pos_lbl == pos_lbl[id1]).nonzero()
                        ix1 = ix1[ix1 != id1]
                        if len(ix1) > 0:
                            ixx = torch.randint(0, ix1.shape[0], (1,))[0]
                            ix1c = ix1[ixx]
                            pairs.append([id1, ix1c])
                            pairs_lbl.append(1)
                        ix2 = (pos_lbl != pos_lbl[id1]).nonzero()
                        if len(ix2) > 0:
                            ixx = torch.randint(0, ix2.shape[0], (1,))[0]
                            ix2c = ix2[ixx]
                            pairs.append([id1, ix2c])
                            pairs_lbl.append(-1)
                        if len(ix1) > 0:
                            ix3c = torch.randint(npos+1, dim, (1,))[0]
                            pairs.append([id1, ix3c])
                            pairs_lbl.append(-1)
                    pairs_t2 = torch.tensor(pairs, dtype=torch.long, device=embed.device)
                    pairs_lbl_t2 = torch.tensor(pairs_lbl, dtype=torch.long, device=embed.device)
                    if (pairs_lbl_t2 == 1).sum() > 5:
                        loss_emb_cls += self.criterion_emb(embed[pairs_t2[:, 0]], embed[pairs_t2[:, 1]], pairs_lbl_t2)

                loss_bbox['loss_emb_id'] = self.embed_coef_id * loss_emb_id
                loss_bbox['loss_emb_cls'] = self.embed_coef_cls * loss_emb_cls
                zz=0
                zz=0
            losses.update(loss_bbox)
        return losses

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels
