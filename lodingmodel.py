
from libraries import *
bins=8
fixed_size             = tuple((500, 500))
labels_name=['Altrnaria_blight', 'Cercospora_LF', 'anthracnose',
       'bacteral_leaf_spot', 'bacterial_wilt', 'cucumber_mosic_virus',
       'downy_mildew', 'good_cucumber', 'powdery_mildew']


le=LabelEncoder()
le.fit_transform(labels_name)


model_path=r"C:\Users\moshi\webpage\models\randomfores.joblib"




def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img
def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img



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



test=r'C:\Users\moshi\webpage\sample_images\download (1).jpg'

def model_result(file_path):

    fixed_size             = tuple((500, 500))
    bins                   = 8

    image1 = cv2.imread(file_path)
    image = cv2.resize(image1 ,(500,500))


    #test_global_feature     = rgb_bgr(image)
    RGB_BGR       = rgb_bgr(image)
    BGR_HSV       = bgr_hsv(RGB_BGR)
    IMG_SEGMENT   = img_segmentation(RGB_BGR,BGR_HSV)


    fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
    fv_haralick   = fd_haralick(IMG_SEGMENT)
    fv_histogram  = fd_histogram(IMG_SEGMENT)
    test_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])





    #loding model 
    predict_test=joblib.load(model_path)
    #testing using the loded model

    res=predict_test.predict(test_feature.reshape(1,-1))

    print(res)
    res= (le.inverse_transform(res)[0])
    return res


