"""
To use this script:
	> python run_predict.py predict 
	This load the object net-specialists.pickle and write all output csv files with _base.csv

	 -OR-

	> python run_analysis.py fit name.pickle name
	This load the object name.pickle and write all output csv files with _name.csv
"""

from helpers import *

class DataAugmentation_FullTest:

    def __init__(self, X, PIXELS = 60, rotation = 90, translation = (0,0), do_flip = False):

        self.X = copy.copy(X)
        #self.X_copy = copy.copy(self.X)
        self.PIXELS = PIXELS

        self.augmentation_params = {
            'zoom_range': (1.0, 1.1),
            'rotation_range': (0, 360),
            'shear_range': (0, 5),
            'translation_range': (-4, 4),
        }

        self.IMAGE_WIDTH = self.PIXELS
        self.IMAGE_HEIGHT = self.PIXELS
        self.rotation = rotation
        self.translation = translation
        self.do_flip = do_flip

    def fast_warp(self, img, tf, output_shape=(PIXELS,PIXELS), mode='nearest'):
        """
        This wrapper function is about five times faster than skimage.transform.warp, for our use case.
        """
        m = tf.params
        img_wf = np.empty((output_shape[0], output_shape[1]), dtype='float32')
        img_wf = skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode)
        return img_wf

    def random_perturbation_transform(self,zoom_range, rotation_range, shear_range, translation_range, do_flip=False):
        # random shift [-4, 4] - shift no longer needs to be integer!
        shift_x = np.random.uniform(*translation_range)
        shift_y = np.random.uniform(*translation_range)
        translation = (shift_x, shift_y)

        # random rotation [0, 360]
        rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

        # random shear [0, 5]
        shear = np.random.uniform(*shear_range)

        # random zoom [0.9, 1.1]
        # zoom = np.random.uniform(*zoom_range)
        log_zoom_range = [np.log(z) for z in zoom_range]
        zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.
        
        translation = (0,0)
        rotation = 0.0
        shear = 0.0
        zoom = 1.0
        
        rotation = self.rotation
        translation = self.translation    

        # flip
        if self.do_flip:
            shear += 180
            rotation += 180

        print "   translation = ", translation
        print "   rotation = ", rotation
        print "   shear = ",shear
        print "   zoom = ",zoom
        print ""

        return self.build_augmentation_transform(zoom, rotation, shear, translation)


    def build_augmentation_transform(self, zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
        center_shift = np.array((self.IMAGE_HEIGHT, self.IMAGE_WIDTH)) / 2. - 0.5
        tform_center = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_augment = transform.AffineTransform(scale=(1/zoom, 1/zoom), 
                                                  rotation=np.deg2rad(rotation), 
                                                  shear=np.deg2rad(shear), 
                                                  translation=translation)
        tform = tform_center + tform_augment + tform_uncenter 
        return tform


    def transform(self, plot = False):

        tform_augment = self.random_perturbation_transform(**self.augmentation_params)
        tform_identity = skimage.transform.AffineTransform()
        tform_ds = skimage.transform.AffineTransform()
        
        for i in range(self.X.shape[0]):
            new = self.fast_warp(self.X[i][0], tform_ds + tform_augment + tform_identity, 
                                 output_shape=(self.PIXELS,self.PIXELS), mode='nearest').astype('float32')
            self.X[i,:] = new

        return self.X

    def delete(self):
        del self.X



def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    predictions = np.clip(y_pred, eps, 1 - eps)
    predictions /= predictions.sum(axis=1)[:, np.newaxis]
    actual = np.zeros(np.shape(y_pred))
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss



def predict(fname_specialists='net-specialists.pickle', append_name = "base"):
    with open(fname_specialists, 'rb') as f:
        specialists = pickle.load(f)

	print "Making Prediction for:",fname_specialists
	print "Appending to name:",append_name

    X,y = load2d(test=True)
    augmentations_stacked = [X]
        
    flips = [False,True]
    rotations = [0,90,180,270,45,135,225,315]
    names = ['0', '90', '180', '270', '45','135','225','315'
             '0 - flip', '90 - flip', '180 - flip', '270 - flip','45 - flip','135 - flip','225 - flip','315 - flip']
    predictions_stacked = []
    ii = 0

    for flip in flips:
        for rot in rotations:
            print "Creating test set for",rot," + ",flip
            test_aug = DataAugmentation_FullTest(X, PIXELS = 60, rotation = rot, translation = (0,0), do_flip = flip)
            new_test = test_aug.transform()    

            
            for model in specialists.values():
                print "   Predicting for test model:",names[ii]
                y_pred = model.predict_proba(new_test)
                predictions_stacked.append(y_pred)
                print "   Mulit Log loss = ",multiclass_log_loss(y,y_pred)
                print ""
                ii = ii + 1
                test_aug.delete()
            

    print "Numer of models predicted = ",len(predictions_stacked)
    print ""

    avg_rot_base = (np.array(predictions_stacked[0]) + np.array(predictions_stacked[1]) + np.array(predictions_stacked[2]) + np.array(predictions_stacked[3])) / 4

    avg_rot_flip = (np.array(predictions_stacked[0]) + np.array(predictions_stacked[1]) + np.array(predictions_stacked[2]) + np.array(predictions_stacked[3]) + \
                    np.array(predictions_stacked[8]) + np.array(predictions_stacked[9]) + np.array(predictions_stacked[10]) + np.array(predictions_stacked[11])) / 8

    avg_full_rot_flip = (np.array(predictions_stacked[0]) + np.array(predictions_stacked[1]) + np.array(predictions_stacked[2]) + np.array(predictions_stacked[3]) + \
             	         np.array(predictions_stacked[4]) + np.array(predictions_stacked[5]) + np.array(predictions_stacked[6]) + np.array(predictions_stacked[7]) + \
             	         np.array(predictions_stacked[8]) + np.array(predictions_stacked[9]) + np.array(predictions_stacked[10]) + np.array(predictions_stacked[11]) + \
             	         np.array(predictions_stacked[12]) + np.array(predictions_stacked[13]) + np.array(predictions_stacked[14]) + np.array(predictions_stacked[15])) / 16


    print "FINAL COMBINED Mulit Log loss for base = ",multiclass_log_loss(y,predictions_stacked[0])
    print "FINAL COMBINED Mulit Log loss for base rotations = ",multiclass_log_loss(y,avg_rot_base)
    print "FINAL COMBINED Mulit Log loss for base rotations + Flip = ",multiclass_log_loss(y,avg_rot_flip)
    print "FINAL COMBINED Mulit Log loss for all rotations + Flip = ",multiclass_log_loss(y,avg_full_rot_flip)
    print ""

    df = DataFrame(predictions_stacked[0])
    print df.head()
    print df.shape
    df.to_csv("DBN_prediction_out_" + append_name + ".csv")

    df_rot = DataFrame(avg_rot_base)
    print df_rot.head()
    print df_rot.shape
    df_rot.to_csv("DBN_prediction_out_blended_test_sets_base_rot_" + append_name + ".csv")

    df_rot_flip = DataFrame(avg_rot_flip)
    print df_rot_flip.head()
    print df_rot_flip.shape
    df_rot_flip.to_csv("DBN_prediction_out_blended_test_sets_base_rot_flip_" + append_name + ".csv")

    df_full_rot_flip = DataFrame(avg_full_rot_flip)
    print df_full_rot_flip.head()
    print df_full_rot_flip.shape
    df_full_rot_flip.to_csv("DBN_prediction_out_blended_test_sets_full_rot_flip_" + append_name + ".csv")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        func = globals()[sys.argv[1]]
        func(*sys.argv[2:])
