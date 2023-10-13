from matplotlib.pyplot import imshow, axis, show
from cv2 import cvtColor, COLOR_BGR2RGB, rectangle, getTextSize, FONT_HERSHEY_SIMPLEX, putText

def iou_bbox(box1, box2):
    '''
        Input: box1 (x1,y1,x2,y2)
               box2 (x1,y1,x2,y2)
        Output: IoU of box1 and box2
    '''
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # Conpute intersections areas
    x1 = max(b1_x1, b2_x1)
    y1 = max(b1_y1, b2_y1)
    x2 = min(b1_x2, b2_x2)
    y2 = min(b1_y2, b2_y2)

    inter = max((x2-x1), 0) * max((y2-y1), 0)

    # Compute onion areas

    box1Area = abs((b1_x1 - b1_x2)*(b1_y1 - b1_y2))
    box2Area = abs((b2_x1 - b2_x2)*(b2_y1 - b2_y2))

    union = float(box1Area+box2Area-inter)

    # Compute IoU ratio
    iou = inter/union

    return iou

def visualize_bbox(img, bboxes, label_encoder):
    img = cvtColor(img, COLOR_BGR2RGB)
    for box in bboxes:
        x_min, y_min, x_max, y_max, predict_id, conf_score = box
        rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        class_name = label_encoder.inverse_transform([predict_id])[0]
        label = f"{class_name} {conf_score}"
        (w,h), _ = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1)
        rectangle(img, (x_min, y_min-20),(x_min+w, y_min), (0,255,0), -1)
        putText(img, label, (x_min, y_min-5), FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),1) 

    imshow(img)
    axis("off")
    show()

