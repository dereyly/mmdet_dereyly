import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .pipelines import Compose
from .registry import DATASETS

is_vis = False
is_dbg = False
if is_vis:
    import shutil
    import os
    import cv2
    dir_dbg = '/media/dereyly/data/ImageDB/dbg/'
    shutil.rmtree(dir_dbg, ignore_errors=True)
    os.makedirs(dir_dbg, exist_ok=True)


def get_bboox_mask(mask,optimize_coef=4):

    sz=mask.shape
    sz_new=(sz[1]//optimize_coef,sz[0]//optimize_coef)
    mask=mmcv.imresize(mask, sz_new, return_scale=False, interpolation='bilinear')
    where = np.array(np.where(mask))
    # if where.shape[1]<400/optimize_coef**2:
    #     return None
    x1, y1 = np.amin(where, axis=1)*optimize_coef
    x2, y2 = np.amax(where, axis=1)*optimize_coef

    return [y1,x1,y2,x2]

@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.img_infos = self.load_annotations(self.ann_file)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self.img_infos), dtype=np.uint8)
        for i in range(len(self.img_infos)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        # print(idx)
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None and np.random:
                idx = self._rand_another(idx)
                continue
            if data is None:
                idx = self._rand_another(idx)
                continue
            if self.test_mode:
                data['img'] = [data['img'].data]
                data['img_meta'].data['idx'] = idx
                data['img_meta'] = [data['img_meta']]
                data['gt_bboxes'] = data['gt_bboxes'].data
                data['gt_labels'] = data['gt_labels'].data
                data['gt_masks'] = data['gt_masks'].data
                # except:
                #     print(idx,data['img_meta'].data['filename'])
                #     zz=0
            if is_vis:
                img = data['img'].data.numpy()
                gt_bboxes = data['gt_bboxes'].data.numpy()
                gt_labels = data['gt_labels'].data.numpy()
                if 'gt_masks' in data:
                    gt_masks = data['gt_masks'].data
                name = data['img_meta'].data['filename']
                im = np.transpose((img + 1) * 128, (1, 2, 0))[:, :, ::-1].copy()
                # str_dbg = ''
                for k, bb in enumerate(gt_bboxes):
                    # if gt_labels[k] != 3:
                    #     continue
                    bb = bb.astype(int)
                    cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 1)
                    cv2.putText(im, '%d -- %d' % (k, gt_labels[k]), (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0))

                    if 'gt_masks' in data:
                        mask = gt_masks[k].astype(np.bool)
                        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                        im[mask] = im[mask] * 0.35 + color_mask * 0.65
                # print(img_info['filename'],str_dbg)
                imname = dir_dbg + name.split('/')[-1]
                # print('vis',imname, gt_bboxes,im.shape)
                cv2.imwrite(imname, im)
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        pass

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        annotations = [self.get_ann_info(i) for i in range(len(self.img_infos))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results['recall@{}@{}'.format(num, iou)] = recalls[i,
                                                                            j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
        return eval_results
