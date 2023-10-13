from cv2 import imread
from joblib import load
from numpy import argmax
import numpy as np
from utils.localization_process import pyramid, sliding_window, nms
from utils.classification_preprocess import preprocess_img
from utils.metrics import visualize_bbox

image_path = "/home/jackson/Desktop/PetProject/Traffc_sign_detection/traffic_sign_detection/images/road260.png"
window_size = [
    (32,32),
    (64,64),
    (128,128),
]
stride = 12
conf_threshold = 0.745
iou_threshold = 0.005
image = imread(image_path)

clf, scaler, label_encoder = load("weights/clf_model_and_scaler_feature.pkl")

pyramid_imgs = pyramid(image)
bboxes = []
for pyramid_img_info in pyramid_imgs:
    pyramid_img, scale_factor = pyramid_img_info
    window_lst = sliding_window(pyramid_img,
                                window_sizes=window_size,
                                stride=stride
                                )
    for window in window_lst:
        xmin, ymin, xmax, ymax = window
        object_img = pyramid_img[ymin:ymax, xmin:xmax]
        preprocessed_img = preprocess_img(object_img)
        normalized_img = scaler.transform([preprocessed_img])[0]
        decision = clf.predict_proba([normalized_img])[0]
        if np.all(decision < conf_threshold):
            continue
        predict_id = argmax(decision)
        conf_score = decision[predict_id]
        xmin = int(xmin/scale_factor)
        ymin = int(ymin/scale_factor)
        xmax = int(xmax/scale_factor)
        ymax = int(ymax/scale_factor)
        bboxes.append([xmin, ymin, xmax, ymax, predict_id, conf_score])

# conf_list = [box[-1] for box in bboxes]
# print(min(conf_list), min(conf_list))
bboxes = nms(bboxes, iou_threshold=iou_threshold)

visualize_bbox(image, bboxes, label_encoder)