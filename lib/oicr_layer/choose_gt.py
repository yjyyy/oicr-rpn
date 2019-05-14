import numpy as np
import torch
from torch.autograd import Variable
from model.utils.cython_bbox import bbox_overlaps


def choose_gt(boxes, cls_prob, im_labels):

    boxes = boxes[...,1:]
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 5), dtype=np.float32)

    if 21 == cls_prob.shape[2] :
        cls_prob = cls_prob[:,:,1:]

    for i in range(num_classes):
        if im_labels_tmp[i] == 1:
            gt_boxes_tmp = np.zeros((1, 5), dtype=np.float32)
            cls_prob_tmp = cls_prob[:,:, i].data
            max_index = np.argmax(cls_prob_tmp)
            gt_boxes_tmp[:, 0:4] = boxes[:,max_index, :].reshape(1, -1)
            gt_boxes_tmp[:, 4] = i+1
            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp))

    # choose pos samples by gt
    overlaps = bbox_overlaps(
        np.ascontiguousarray(boxes[0], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    max_overlaps = overlaps.max(axis=1)

    fg_inds = np.where(max_overlaps >= 0.5)[0]
    pos_samples = np.empty((0,4), dtype=np.float32)
    if fg_inds.shape[0] != 0:
        pos_samples = np.vstack((pos_samples, boxes[0][fg_inds, :]))
        pos_samples = np.hstack((np.zeros((pos_samples.shape[0], 1), dtype=np.float32), pos_samples))
    pos_samples = Variable(torch.from_numpy(np.array([pos_samples])).cuda())

    gt_boxes = np.array([gt_boxes])
    gt_boxes = Variable(torch.from_numpy(gt_boxes))
    if torch.cuda.is_available():
        gt_boxes = gt_boxes.cuda()

    return gt_boxes, pos_samples



