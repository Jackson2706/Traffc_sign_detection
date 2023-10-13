from cv2 import resize
from numpy import array, argsort, maximum, minimum, where


def sliding_window(img, window_sizes, stride):
    img_height, img_width = img.shape[:2]
    windows = []
    for window_size in window_sizes:
        window_width, window_height = window_size = window_size
        for y_min in range(0, img_height-window_height+1, stride):
            for x_min in range(0, img_width-window_width+1, stride):
                x_max = x_min + window_width
                y_max = y_min + window_height
                windows.append([x_min, y_min, x_max, y_max])
    return windows


def pyramid(img, scale=0.8, min_size=(30,30)):
    acc_scale = 1.0
    pyramid_imgs = [(img, acc_scale)]
    i = 0
    while True:
        acc_scale = acc_scale * scale
        h = int(img.shape[0]*acc_scale)
        w = int(img.shape[1]*acc_scale)
        if h < min_size[0] or w < min_size[0]:
            break
        img = resize(img, (w,h))
        pyramid_imgs.append((img, acc_scale*scale**i))
        i =  i + 1
    return pyramid_imgs


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    x1 = maximum(box[0], boxes[:, 0])
    y1 = maximum(box[1], boxes[:, 1])
    x2 = minimum(box[2], boxes[:, 2])
    y2 = minimum(box[3], boxes[:, 3])
    
    intersection =  maximum((x2-x1), 0) * maximum((y2-y1), 0)

    union = box_area + boxes_area[:] - intersection

    iou = intersection * 1.0 / union

    return iou

def nms(bboxes, iou_threshold):
    if not bboxes:
        return []
    scores = array([bbox[5] for bbox in bboxes])
    sorted_indices = argsort(scores)[::-1]

    xmin = array([bbox[0] for bbox in bboxes])
    ymin = array([bbox[1] for bbox in bboxes])
    xmax = array([bbox[2] for bbox in bboxes])
    ymax = array([bbox[3] for bbox in bboxes])

    areas = (xmax - xmin + 1) * (ymax - ymin + 1)

    keep = []

    while sorted_indices.size > 0:
        i = sorted_indices[0]
        keep.append(i)

        iou = compute_iou(
            [xmin[i], ymin[i],xmax[i],ymax[i]],
            array(
                [xmin[sorted_indices[1:]],
                 ymin[sorted_indices[1:]],
                 xmax[sorted_indices[1:]],
                 ymax[sorted_indices[1:]]]
            ).T,
            areas[i],
            areas[sorted_indices[1:]]
        )

        idx_to_keep = where(iou <= iou_threshold)[0]
        sorted_indices = sorted_indices[idx_to_keep + 1]
    return [bboxes[i] for i in keep]