from libraries import *

bins=8
def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img
def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img
# image segmentation

# for extraction of green and brown color


def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()



test=r"C:\Users\moshi\webpage\sample_images\download_8.jpg"

fixed_size             = tuple((500, 500))
bins                   = 8

image1 = cv2.imread(test)
#input_img = np.expand_dims(image1, axis=0)
image = cv2.resize(image1 ,fixed_size)
#print(image1.shape)




RGB_BGR       = rgb_bgr(image)

BGR_HSV       = bgr_hsv(RGB_BGR)
print(BGR_HSV.shape)


#test_global_feature     = rgb_bgr(image)
RGB_BGR       = rgb_bgr(image)
#cv2.imshow(RGB_BGR)
BGR_HSV       = bgr_hsv(RGB_BGR)
#cv2.imshow(BGR_HSV)
IMG_SEGMENT   = img_segmentation(RGB_BGR,BGR_HSV)
#cv2.imshow(IMG_SEGMENT)

fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
fv_haralick   = fd_haralick(IMG_SEGMENT)
fv_histogram  = fd_histogram(IMG_SEGMENT)
test_global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

#model_path=r"C:\Users\moshi\webpage\models\random_model.sav"
train=r"C:\Users\moshi\webpage\models\train_data_new.h5"
label=r"C:\Users\moshi\webpage\models\train_labels.h5"

# import the feature vector and trained labels
h5f_data  = h5py.File(train, 'r')
h5f_label = h5py.File(label, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

#train test spliting the data
num_trees = 50
test_size = 0.20
seed      = 9

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),np.array(global_labels),test_size=test_size,random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
#clf  = RandomForestClassifier(n_estimators=150, random_state=seed)

#model=clf.fit(trainDataGlobal, trainLabelsGlobal)
#y_predict=clf.predict(testDataGlobal)
#cm = confusion_matrix(testLabelsGlobal,y_predict)
#import seaborn as sns
#sns.heatmap(cm ,annot=True)
#from sklearn.metrics import accuracy_score
#print(accuracy_score(testLabelsGlobal, y_predict)*100)
model_path=r"C:\Users\moshi\webpage\models\randomfores.joblib"
#joblib.dump(clf,model_path)



p=joblib.load(model_path)

res=p.predict(test_global_feature.reshape(1,-1))
res1=p.predict_proba(test_global_feature.reshape(1,-1))
print(res1)

#le.inverse_transform(res)[0]


