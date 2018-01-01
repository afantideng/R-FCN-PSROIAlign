# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # # ---- test ----
    # widths_nan = np.isnan(widths)
    # widths_nan_idx = np.where(widths_nan == True)[0]
    # heights_nan = np.isnan(heights)
    # heights_nan_idx = np.where(heights_nan == True)[0]
    # print '--- widths_nan_idx: --- '
    # print widths_nan_idx
    # print '--- heights_nan_idx: ---'
    # print heights_nan_idx

    # ctr_x_nan = np.isnan(ctr_x)
    # ctr_x_nan_idx = np.where(ctr_x_nan == True)[0]
    # ctr_y_nan = np.isnan(ctr_y)
    # ctr_y_nan_idx = np.where(ctr_y_nan == True)[0]
    # print '--- ctr_x_nan_idx: --- '
    # print ctr_x_nan_idx
    # print '--- ctr_y_nan_idx: ---'
    # print ctr_y_nan_idx  
    # # --------------


    # ----------- test ------------
    # print '----- widths -----'
    # print widths
    # print 'max widths: ' + str(widths.max())
    # print '----- heights -----'
    # print heights
    # print 'max heights: ' + str(heights)
    # print '----- ctr_x -----'
    # print ctr_x
    # print 'max ctr_x: ' + str(ctr_x.max())
    # print '----- ctr_y -----'
    # print ctr_y
    # print 'max ctr_y: ' + str(ctr_y.max())

    widths[widths > 1000.0] = 1000.0
    heights[heights > 1000.0] = 1000.0
    ctr_x[ctr_x > 1000.0] = 1000.0
    ctr_y[ctr_y > 1000.0] = 1000.0
    # -----------------------------


    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]


    # ----------- test ------------
    # dx_nan = np.isnan(dx)
    # dx_nan_idx = np.where(dx_nan == True)[0]
    # dy_nan = np.isnan(dy)
    # dy_nan_idx = np.where(dy_nan == True)[0]
    # dw_nan = np.isnan(dw)
    # dw_nan_idx = np.where(dw_nan == True)[0]
    # dh_nan = np.isnan(dh)
    # dh_nan_idx = np.where(dh_nan == True)[0]
    # print '--- dx_nan_idx: --- '
    # print dx_nan_idx
    # print '--- dy_nan_idx: ---'
    # print dy_nan_idx  
    # print '--- dw_nan_idx: --- '
    # print dw_nan_idx
    # print '--- dh_nan_idx: ---'
    # print dh_nan_idx  

    # print '----- dx -----'
    # print dx 
    # print 'max_dx: ' + str(dx.max())
    # print '----- dy -----'
    # print dy
    # print 'max_dy: ' + str(dy.max())
    # print '----- dw -----'
    # print dw 
    # print 'max_dw: ' + str(dw.max())
    # print '----- dh -----'
    # print dh
    # print 'max_dh: ' + str(dh.max())
    
    dx[dx > 5.0] = 5.0
    dy[dy > 5.0] = 5.0
    dx[dx < -5.0] = -5.0
    dy[dy < -5.0] = -5.0
    dw[dw > 5.0] = 5.0
    dh[dh > 5.0] = 5.0
    dw[dw < -5.0] = -5.0
    dh[dh < -5.0] = -5.0
    # -----------------------------

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    # # ---- test ----
    # pred_ctr_x_nan = np.isnan(pred_ctr_x)
    # pred_ctr_x_nan_idx = np.where(pred_ctr_x_nan == True)[0]
    # pred_ctr_y_nan = np.isnan(pred_ctr_y)
    # pred_ctr_y_nan_idx = np.where(pred_ctr_y_nan == True)[0]
    # print '--- pred_ctr_x_nan_idx: --- '
    # print pred_ctr_x_nan_idx
    # print '--- pred_ctr_y_nan_idx: ---'
    # print pred_ctr_y_nan_idx

    # pred_w_nan = np.isnan(pred_w)
    # pred_w_nan_idx = np.where(pred_w_nan == True)[0]
    # pred_h_nan = np.isnan(pred_h)
    # pred_h_nan_idx = np.where(pred_h_nan == True)[0]
    # print '--- pred_w_nan_idx: --- '
    # print pred_w_nan_idx
    # print '--- pred_h_nan_idx: ---'
    # print pred_h_nan_idx
    # # --------------

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
