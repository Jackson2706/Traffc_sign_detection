from cv2 import resize, cvtColor, COLOR_BGR2GRAY
from numpy import float64
from skimage.transform import resize
from skimage.feature import hog

"""
    Extract images from RGB images to 1D images with using HoG method
    @param: image_path, for example: "./images/abc.jpg"
    @return: a hoG image

"""
def preprocess_img(img):
    if len(img.shape) > 2:
        img = cvtColor(img, COLOR_BGR2GRAY)
    img = img.astype(float64)
    resized_img = resize(img, output_shape=(32,32), anti_aliasing= True) 
    hog_features = hog(resized_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
                       transform_sqrt=True, block_norm="L2", feature_vector=True)
    return hog_features


def crop_object(img, bbox):
    x_min, y_min, x_max, y_max = bbox
    object_image = img[y_min:y_max, x_min:x_max]
    return object_image