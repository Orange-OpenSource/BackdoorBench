# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger

# Modifications: Add the masked blended and multitarget attacks support

class blendedImageAttack(object):

    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--perturbImagePath', type=str,
                            help='path of the image which used in perturbation')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, target_image, blended_rate, target_mask=None):
        self.target_image = target_image
        self.blended_rate = blended_rate
        self.target_mask = target_mask


    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if self.target_mask is None:
            return (1-self.blended_rate) * img + (self.blended_rate) * self.target_image
        else:
            return img * (1-self.target_mask) + (1-self.blended_rate) * (img*self.target_mask) + (self.blended_rate) * (self.target_image*self.target_mask)


class MultiTargetBlendedAttack(object):
    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--attack_trigger_img_path', type=str,
                            help='path to the trigger directory')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, pat_dict, mask_dict, blended_rate):
        self.pat_dict = pat_dict
        self.mask_dict = mask_dict
        self.blended_rate = blended_rate

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img, target)

    def add_trigger(self, img, target_label):
        target_image, target_mask = self.pat_dict[target_label], self.mask_dict[target_label]
        if target_mask is None:
            return (1-self.blended_rate) * img + (self.blended_rate) * target_image
        else:
            return img * (1-target_mask) + (1-self.blended_rate) * (img*target_mask) + (self.blended_rate) * (target_image*target_mask)

