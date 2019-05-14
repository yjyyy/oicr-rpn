import torch.nn as nn
import numpy as np
import pdb
from model.utils.cython_bbox import bbox_overlaps
import torch
from torch.autograd import Variable


class box_Augment(nn.Module):
    def __init__(self, ratio, offset):
        super(box_Augment, self).__init__()
        self.ratio = np.array([ratio]).T
        self.offset = np.array([offset]).T

    def forward(self, rois, cls_prob, im_labels, im_data):

        rois_augment = np.empty((0,4), dtype=np.float32)
        rois_init = rois.clone()
        rois = rois[...,1:]
        proposals = self._get_highest_score_proposals(rois, cls_prob, im_labels)
        gt_boxes = proposals['gt_boxes']

        # add original postive samples first
        rois_augment = np.vstack((rois_augment, self._sample_rois(rois, proposals)))

        # base on offset and ratio to augment box
        for i in range(gt_boxes.shape[0]):
            gt_scale = np.array([[gt_boxes[i,3]-gt_boxes[i,1], gt_boxes[i,2]-gt_boxes[i,0]]])
            gt_ctr_ori = np.array([(gt_boxes[i,3]+gt_boxes[i,1])/2, (gt_boxes[i,2]+gt_boxes[i,0])/2])
            aug_scale = self.ratio.dot(gt_scale)
            offset_element = np.vstack((np.hstack((self.offset, np.zeros((self.offset.shape[0], 1)))),
                                                   np.vstack((np.hstack((np.zeros((self.offset.shape[0], 1)), self.offset)), np.hstack((self.offset, self.offset))))))
            gt_ctr = gt_ctr_ori + offset_element
            gt_ctr = np.vstack((gt_ctr_ori, gt_ctr))

            for j in range(gt_ctr.shape[0]):
                for k in range(aug_scale.shape[0]):
                    box = np.array([(gt_ctr[j, 1]-aug_scale[k, 1]/2, gt_ctr[j, 0]-aug_scale[k, 0]/2, gt_ctr[j, 1]+aug_scale[k, 1]/2, gt_ctr[j, 0]+aug_scale[k, 0]/2)])
                    if box[0, 0]>0 and box[0, 1]>0 and box[0, 2]<im_data.shape[3] and box[0, 3]<im_data.shape[2] and box[0,0]<box[0,2] and box[0,1]<box[0,3]:
                        rois_augment = np.vstack((rois_augment, box))

        rois_augment = np.array([np.hstack((np.zeros((rois_augment.shape[0], 1)), rois_augment))], dtype=np.float32)
        rois_augment = Variable(torch.from_numpy(rois_augment).cuda())
        # ret_prob = rois_augment.new().new_zeros(1,rois_augment.size(1),21)

        # return rois_augment, ret_prob
        return rois_augment


    def _get_highest_score_proposals(self, boxes, cls_prob, im_labels):
        """Get proposals with highest score."""

        num_images, num_classes = im_labels.shape
        assert num_images == 1, 'batch size shoud be equal to 1'
        im_labels_tmp = im_labels[0, :]
        gt_boxes = np.zeros((0, 4), dtype=np.float32)
        gt_classes = np.zeros((0, 1), dtype=np.int32)
        gt_scores = np.zeros((0, 1), dtype=np.float32)

        if 21 == cls_prob.shape[2] : # added 1016
            cls_prob = cls_prob[:,:,1:]

        for i in range(num_classes):
            if im_labels_tmp[i] == 1:
                cls_prob_tmp = cls_prob[:,:, i].data
                max_index = np.argmax(cls_prob_tmp)
                gt_boxes = np.vstack((gt_boxes, boxes[:,max_index, :].reshape(1, -1)))
                gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32))) # for pushing ground
                gt_scores = np.vstack((gt_scores,
                    cls_prob_tmp[:, max_index] ))  # * np.ones((1, 1), dtype=np.float32)))
                cls_prob[:, max_index, :] = 0 #in-place operation <- OICR code but I do not agree

        proposals = {'gt_boxes' : gt_boxes,
                    'gt_classes': gt_classes,
                    'gt_scores': gt_scores}

        return proposals


    def _sample_rois(self, all_rois, proposals):

        gt_boxes = proposals['gt_boxes']
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[0], dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        try :
            max_overlaps = overlaps.max(axis=1)
        except :
            pdb.set_trace()

        fg_inds = np.where(max_overlaps >= 0.5)[0]
        # gt_index = np.where(max_overlaps == 1.0)[0]
        # fg_inds = np.array(list(set(fg_inds)-set(gt_index)))
        pos_samples = np.empty((0,4))
        if fg_inds.shape[0] != 0:
            pos_samples = np.vstack((pos_samples, all_rois[0][fg_inds, :]))

        return pos_samples