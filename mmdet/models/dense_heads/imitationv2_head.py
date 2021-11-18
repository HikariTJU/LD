import torch
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        distance2bbox, multi_apply, reduce_mean, unmap,
                        images_to_levels)
from ..builder import HEADS, build_loss
from .gfocal_head import GFocalHead


def intersect(box_a, box_b):
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)


def jaccard(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(
                  inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(
                  inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


@HEADS.register_module()
class IMv2Head(GFocalHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_im,
                 loss_ld=dict(
                     type='LocalizationDistillationLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_kd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.5,
                     T=10),
                 **kwargs):
        super().__init__(num_classes, in_channels, **kwargs)
        self.loss_im = build_loss(loss_im)
        self.loss_ld = build_loss(loss_ld)
        self.loss_kd = build_loss(loss_kd)

    def forward_train(self,
                      x,
                      out_teacher,
                      teacher_x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, out_teacher, x, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, out_teacher, x,
                                  teacher_x, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def loss_single(self, anchors, cls_score, cls_feat, bbox_pred, labels,
                    label_weights, bbox_targets, stride, soft_targets,
                    soft_label, x, teacher_x, assigned_neg, num_total_samples,
                    num_total_samples_neg):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        soft_targets = soft_targets.permute(0, 2, 3,
                                            1).reshape(-1,
                                                       4 * (self.reg_max + 1))
        soft_label = soft_label.permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)
        cls_feat = cls_feat.permute(0, 2, 3,
                                    1).reshape(-1, self.cls_out_channels)
        # ------------------feature imitation based on GI locations----------------

        teacher_score = soft_label

        student_score = cls_feat.detach()  #[num,80]

        anchor_centers = self.anchor_center(anchors) / stride[0]
        sdistribution = self.integral(bbox_pred)
        tdistribution = self.integral(soft_targets)
        sbox = distance2bbox(anchor_centers, sdistribution)  #[num,4]
        tbox = distance2bbox(anchor_centers, tdistribution)

        z = teacher_score - student_score  #difference between teacher score and student score on the whole locations.
        giscore, index = torch.abs(z).max(dim=1)  #GI scores
        k = z >= 0  #who is bigger
        j = torch.take(
            k, index + self.cls_out_channels *
            (torch.arange(student_score.size(0)).cuda()))
        h = j == 0
        gibox = sbox.new_zeros(sbox.shape)
        gibox[j] = tbox[j] + 0
        gibox[h] = sbox[h] + 0  #GI boxes

        idx_out = torch.ops.torchvision.nms(gibox, giscore, 0.3)[:10]
        # Cluster-NMS
        '''
        _, idx = giscore.sort(0, descending=True)
        boxes_idx = gibox[idx]
        iou= jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_=torch.max(A, dim=0)
            E = (maxA<=0.3).float().unsqueeze(1).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        idx_out = idx[maxA <= 0.3][:10]	# the final GI boxes' locations for feature imitation
        '''
        gi_teacher = teacher_x.permute(0, 2, 3, 1).reshape(-1, 256)[idx_out]
        gi_student = x.permute(0, 2, 3, 1).reshape(-1, 256)[idx_out]

        loss_gibox_im = self.loss_im(gi_student, gi_teacher)
        #loss_gibox_im = bbox_pred.sum() * 0

        # ---------------------------------------------------------------

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        assigned_neg = assigned_neg.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        #gt_inds = (labels != bg_class_ind).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)
        teacher_x = teacher_x.permute(0, 2, 3, 1).reshape(-1, 256)[pos_inds]
        x = x.permute(0, 2, 3, 1).reshape(-1, 256)[pos_inds]
        score_neg = assigned_neg.new_zeros(assigned_neg.shape)
        remain_inds = (assigned_neg > 0).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            # weight_ld = cls_feat.detach().sigmoid()
            # weight_ld = weight_ld.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)

            # anchor_centers = self.anchor_center(anchors) / stride[0]
            # in_gt = torch.zeros(x.shape[-1], x.shape[-2], device='cuda')
            # for target in pos_decode_bbox_targets:

            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pos_soft_targets = soft_targets[pos_inds]
            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)

            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)
            # loss_ld = bbox_pred.sum() * 0
            # dfl loss
            #loss_dfl = self.loss_dfl(
            #pred_corners,
            #target_corners,
            #weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
            #avg_factor=4.0)
            loss_dfl = bbox_pred.sum() * 0
            #loss_im = 0.0 * self.loss_im(x, teacher_x)
            loss_im = bbox_pred.sum() * 0
            # ld loss
            loss_ld = self.loss_ld(
                pred_corners,
                soft_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
            loss_cls_kd = self.loss_kd(
                cls_feat[pos_inds],
                soft_label[pos_inds],
                weight=label_weights[pos_inds],
                avg_factor=pos_inds.shape[0])
        else:
            loss_ld = bbox_pred.sum() * 0
            loss_cls_kd = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_im = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        if len(remain_inds) > 0:

            neg_pred_corners = bbox_pred[remain_inds].reshape(
                -1, self.reg_max + 1)
            neg_soft_corners = soft_targets[remain_inds].reshape(
                -1, self.reg_max + 1)
            weight_targetss = ((cls_score.detach().max(dim=1)[0]) <
                               0).float()
            remain_targets = weight_targetss[remain_inds] + assigned_neg[
                remain_inds]

            neg_bbox_targets = bbox_targets[remain_inds]
            neg_bbox_pred = bbox_pred[remain_inds]
            neg_anchors = anchors[remain_inds]
            neg_anchor_centers = self.anchor_center(neg_anchors) / stride[0]
            neg_bbox_pred_corners = self.integral(neg_bbox_pred)
            neg_decode_bbox_pred = distance2bbox(neg_anchor_centers,
                                                 neg_bbox_pred_corners)
            neg_decode_bbox_targets = neg_bbox_targets / stride[0]

            score_neg[remain_inds] = bbox_overlaps(
                neg_decode_bbox_pred.detach(),
                neg_decode_bbox_targets,
                is_aligned=True)

            # print(neg_pred_corners.size())
            # print(remain_inds[:, None].expand(-1, 4).reshape(-1).size())
            loss_ld_neg = 0.25 * self.loss_ld(
                neg_pred_corners,
                neg_soft_corners,
                weight=remain_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

            # loss_bbox_neg = 0.0 * self.loss_bbox(
            #     neg_decode_bbox_pred,
            #     neg_decode_bbox_targets,
            #     weight=remain_targets,
            #     avg_factor=1.0)
            # loss_cls_kd_neg = 0 * self.loss_kd(
            #     torch.logit(cls_score)[remain_inds],
            #     torch.logit(soft_label)[remain_inds],
            #     weight=label_weights[remain_inds],
            #     avg_factor=remain_inds.shape[0])
            loss_bbox_neg = bbox_pred.sum() * 0

            loss_cls_kd_neg = bbox_pred.sum() * 0
        else:
            loss_ld_neg = bbox_pred.sum() * 0
            loss_bbox_neg = bbox_pred.sum() * 0
            loss_cls_kd_neg = bbox_pred.sum() * 0

        # cls (qfl) loss

        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, loss_ld, loss_ld_neg, loss_cls_kd, loss_cls_kd_neg, loss_bbox_neg, loss_im, loss_gibox_im, weight_targets.sum(
        )

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             cls_feat,
             gt_bboxes,
             gt_labels,
             soft_teacher,
             x,
             teacher_x,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        _, soft_target, soft_label = soft_teacher
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, labels_list_neg,
         label_weights_list_neg, bbox_targets_list_neg, bbox_weights_list_neg,
         num_total_pos_neg, num_total_neg_neg,
         assigned_neg_list) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        num_total_samples_neg = reduce_mean(
            torch.tensor(num_total_pos_neg, dtype=torch.float,
                         device=device)).item()
        num_total_samples_neg = max(num_total_samples_neg, 1.0)

        losses_cls, losses_bbox, losses_dfl, losses_ld, losses_ld_neg, losses_cls_kd, losses_cls_kd_neg,losses_bbox_neg, losses_im, losses_gibox_im,\
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                cls_feat,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                soft_target,
                soft_label,
                x,
                teacher_x,
                assigned_neg_list,
                num_total_samples=num_total_samples,
                num_total_samples_neg=num_total_samples_neg)

        avg_factor = sum(avg_factor) + 1e-6
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = [x / avg_factor for x in losses_bbox]
        losses_dfl = [x / avg_factor for x in losses_dfl]
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_ld=losses_ld,
            loss_ld_neg=losses_ld_neg,
            loss_cls_kd=losses_cls_kd,
            loss_cls_kd_neg=losses_cls_kd_neg,
            loss_im=losses_im,
            loss_gibox_im=losses_gibox_im)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        '''
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, all_assigned_neg, assigned_neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        '''
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, all_labels_neg,
         all_label_weights_neg, all_bbox_targets_neg, all_bbox_weights_neg,
         pos_inds_list_neg, neg_inds_list_neg, all_assigned_neg) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        #num_total_remain_neg = sum([max(inds.numel(), 1) for inds in assigned_neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        assigned_neg_list = images_to_levels(all_assigned_neg,
                                             num_level_anchors)

        # sampled anchors of all images
        num_total_pos_neg = sum(
            [max(inds.numel(), 1) for inds in pos_inds_list_neg])
        num_total_neg_neg = sum(
            [max(inds.numel(), 1) for inds in neg_inds_list_neg])
        # split targets to a list w.r.t. multiple levels
        labels_list_neg = images_to_levels(all_labels_neg, num_level_anchors)
        label_weights_list_neg = images_to_levels(all_label_weights_neg,
                                                  num_level_anchors)
        bbox_targets_list_neg = images_to_levels(all_bbox_targets_neg,
                                                 num_level_anchors)
        bbox_weights_list_neg = images_to_levels(all_bbox_weights_neg,
                                                 num_level_anchors)
        #assigned_neg_list = images_to_levels(all_assigned_neg,
        #num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg, labels_list_neg, label_weights_list_neg,
                bbox_targets_list_neg, bbox_weights_list_neg,
                num_total_pos_neg, num_total_neg_neg, assigned_neg_list)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        #assign_result, assigned_neg, assigned_neg_inds = self.assigner.assign(anchors, num_level_anchors_inside,
        #gt_bboxes, gt_bboxes_ignore,
        #gt_labels)
        assign_result = self.assigner.assign_pos(anchors,
                                                 num_level_anchors_inside,
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        assign_result_neg, assigned_neg = self.assigner.assign_neg(
            anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore,
            gt_labels)

        sampling_result_neg = self.sampler.sample(assign_result_neg, anchors,
                                                  gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        bbox_targets_neg = torch.zeros_like(anchors)
        bbox_weights_neg = torch.zeros_like(anchors)

        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        labels_neg = anchors.new_full((num_valid_anchors, ),
                                      self.num_classes,
                                      dtype=torch.long)

        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        label_weights_neg = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_inds_neg = sampling_result_neg.pos_inds
        neg_inds_neg = sampling_result_neg.neg_inds

        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if len(pos_inds_neg) > 0:
            pos_bbox_targets_neg = sampling_result_neg.pos_gt_bboxes
            bbox_targets_neg[pos_inds_neg, :] = pos_bbox_targets_neg
            bbox_weights_neg[pos_inds_neg, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels_neg[pos_inds_neg] = 0
            else:
                labels_neg[pos_inds_neg] = gt_labels[
                    sampling_result_neg.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights_neg[pos_inds_neg] = 1.0
            else:
                label_weights_neg[pos_inds_neg] = self.train_cfg.pos_weight
        if len(neg_inds_neg) > 0:
            label_weights_neg[neg_inds_neg] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            assigned_neg = unmap(assigned_neg, num_total_anchors, inside_flags)

            labels_neg = unmap(
                labels_neg,
                num_total_anchors,
                inside_flags,
                fill=self.num_classes)
            label_weights_neg = unmap(label_weights_neg, num_total_anchors,
                                      inside_flags)
            bbox_targets_neg = unmap(bbox_targets_neg, num_total_anchors,
                                     inside_flags)
            bbox_weights_neg = unmap(bbox_weights_neg, num_total_anchors,
                                     inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, labels_neg, label_weights_neg,
                bbox_targets_neg, bbox_weights_neg, pos_inds_neg, neg_inds_neg,
                assigned_neg)
