from sklearn.preprocessing import LabelEncoder, StandardScaler
from cv2 import imread
from numpy import array
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from joblib import dump

from dataset.dataset import Dataset
from utils.classification_preprocess import preprocess_img, crop_object

"""
    Preprocessing dataset before training SVM
"""
def preprocess_dataset(image_path_list, annotation_list):
    image_feature_list = []
    label_list = []
    for image_path, annotation in zip(image_path_list, annotation_list):
        image = imread(image_path)
        for [x1,y1,x2,y2, label] in annotation:    
            object_img = crop_object(image, [x1,y1,x2,y2])
            hog_object_image = preprocess_img(object_img)
            image_feature_list.append(hog_object_image)
            label_list.append(label)
    return array(image_feature_list), array(label_list)

if __name__ == '__main__':
    data = Dataset(dataset_dir="traffic_sign_detection")
    image_path_list, annotation_list = data.__call__()
    X, y = preprocess_dataset(image_path_list=image_path_list, annotation_list=annotation_list)
    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 1234, test_size=0.3)

    '''
        Training phase
    '''
    # Normalize the feature
    scaler = StandardScaler()
    scaler.fit_transform(X_train)

    # Defining the model
    clf = SVC(
        kernel='rbf',
        random_state = 123,
        probability = True,
        C=0.5
    )
    clf.fit(X_train, y_train)

    '''
        Validation phase
    '''
    # Predict val data
    y_pred = clf.predict(X_test)
    # Evaluate the model
    acc_score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    print("Accuracy: {} \nPrecision: {} \nRecall: {} \n".format(acc_score, precision, recall))

    dump((clf, scaler, label_encoder), "weights/clf_model_and_scaler_feature.pkl")




    
