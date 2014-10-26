function roidb = roidb_from_nyud2_region(imdb, II)
  if(~exist('II', 'var')), II = 1:length(imdb.image_ids); end

  imdb_name = imdb.name;
  image_ids = imdb.image_ids;
  cls_to_id = imdb.cls_to_id;
  num_classes = imdb.num_classes;
  regionDir = imdb.regionDir;

  parfor ii = 1:length(II)
    i = II(ii);
    tic_toc_print('roidb (%s): %d/%d\n', imdb_name, ii, length(II));
    
    % Load the ground truth annotations
    rec = getGroundTruthBoxes(imdb, i); 

    % Load the boxes
    dt = load(fullfile_ext(regionDir, image_ids{i}, 'mat'), 'bboxes', 'superpixels', 'sp2reg');
    dt.bboxes = dt.bboxes(1:min(imdb.max_boxes, size(dt.bboxes,1)), [2 1 4 3]);
    dt.sp2reg = dt.sp2reg(1:min(imdb.max_boxes, size(dt.bboxes,1)), :);
   
    % Attach the regions
    rois(ii) = attach_proposals_region(rec, dt, cls_to_id, num_classes);
  end
  roidb.rois(II) = rois;
  if(length(roidb.rois) < length(imdb.image_ids))
    roidb.rois(length(imdb.image_ids)+1) = roidb.rois(1);
    roidb.rois(length(imdb.image_ids)+1) = [];
  end
end
