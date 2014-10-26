function rec = attach_proposals_region(voc_rec, candidates, class_to_id, num_classes)
  % ------------------------------------------------------------------------
  %           gt: [2108x1 double]
  %      overlap: [2108x20 single]
  %      dataset: 'voc_2007_trainval'
  %        boxes: [2108x4 single]
  %         feat: [2108x9216 single]
  %        class: [2108x1 uint8]
  %       sp2reg: [nR x nSP]
  %           sp: [465x560]
  %region_overlaps: [nR x nR]

  if isfield(voc_rec, 'objects')
    gt_boxes = cat(1, voc_rec.objects(:).bbox);
    ind = isKey(class_to_id, {voc_rec.objects(:).class});
    gtc = class_to_id.values({voc_rec.objects(ind).class});
    gtc = cat(1, gtc{:});
    gt_classes = zeros(length(voc_rec.objects), 1);
    gt_classes(ind) = gtc;
    num_gt_boxes = size(gt_boxes, 1);
    inst_mask = zeros([size(voc_rec.inst)]);
    for i = 1:length(voc_rec.objects),
      inst_mask(voc_rec.inst == voc_rec.objects(i).instanceId) = i;
    end
  else
    gt_boxes = [];
    gt_classes = [];
    num_gt_boxes = 0;
    inst_mask = zeros([size(voc_rec.inst)]);
  end
  all_boxes = cat(1, gt_boxes, candidates.bboxes);
  num_boxes = size(candidates.bboxes, 1);

  % Add the ground truth instances to the set of regions
  n_sp = max(candidates.superpixels(:));
  sp2reg_gt = accumarray([candidates.superpixels(:), inst_mask(:)+1], 1, [n_sp num_gt_boxes+1])' > 0;
  
  % slower code
  % sp_area = accumarray(candidates.superpixels(:), 1)';
  % sp2reg_gt_2 = bsxfun(@rdivide, accumarray([candidates.superpixels(:), inst_mask(:)+1], 1, [n_sp num_gt_boxes+1])', sp_area) > 0.5;
  % assert(isequal(sp2reg_gt, sp2reg_gt_2));

  sp2reg_gt = sp2reg_gt(2:end,:);
  all_sp2reg = cat(1, sp2reg_gt, candidates.sp2reg);
  [iu, inter, reg_area_1, reg_area_2] = compute_region_overlap(candidates.superpixels, sp2reg_gt, all_sp2reg);

  rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
  rec.overlap = zeros(num_gt_boxes+num_boxes, num_classes, 'single');
  for i = 1:num_gt_boxes
    if(gt_classes(i) > 0)
      rec.overlap(:, gt_classes(i)) = ...
          max(rec.overlap(:, gt_classes(i)), iu(i,:)');
    end
  end
  rec.boxes = single(all_boxes);
  rec.feat = [];
  rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));
  rec.sp = candidates.superpixels;
  rec.sp2reg = all_sp2reg;
  rec.info = 'overlap is region overlap';
end
