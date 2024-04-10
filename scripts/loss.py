def mask_rcnn_loss(loss, cfg):
    combined_loss = (
        cfg.loss_classifier * loss["loss_classifier"]
        + cfg.loss_box_reg * loss["loss_box_reg"]
        + cfg.loss_mask * loss["loss_mask"]
        + cfg.loss_objectness * loss["loss_objectness"]
        + cfg.loss_rpn_box_reg * loss["loss_rpn_box_reg"]
    ) / cfg.total_loss_coeff
    return combined_loss
