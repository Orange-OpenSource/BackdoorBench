# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.



# Modifications: Add the MaskBlended attack and multitarget attack support


import sys, logging, os
sys.path.append('../../')
import imageio
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from utils.bd_img_transform.lc import labelConsistentAttack
from utils.bd_img_transform.blended import blendedImageAttack, MultiTargetBlendedAttack
from utils.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger
from utils.bd_img_transform.sig import sigTriggerAttack
from utils.bd_img_transform.SSBA import SSBA_attack_replace_version
from utils.bd_img_transform.ftrojann import ftrojann_version
from utils.bd_label_transform.backdoor_label_transform import *
from torchvision.transforms import Resize
from utils.bd_img_transform.ctrl import ctrl


class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list
    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img

class convertNumpyArrayToFloat32(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np_img_float32.astype(np.float32)
npToFloat32 = convertNumpyArrayToFloat32()

class clipAndConvertNumpyArrayToUint8(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np.clip(np_img_float32, 0, 255).astype(np.uint8)
npClipAndToUint8 = clipAndConvertNumpyArrayToUint8()

def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''

    if args.attack in ['badnet',]:


        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(args.patch_mask_path)),
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif args.attack in ['blended', 'maskblended']:

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            transforms.ToTensor()
        ])

        target_mask = None
        if (args.attack=='maskblended') and (len(args.attack_trigger_mask_path) > 0):
            target_mask = trans(
                        imageio.imread(args.attack_trigger_mask_path)  # '../data/hello_kitty_mask.jpeg'
                    ).cpu().numpy().transpose(1, 2, 0) * 255
            target_mask = ((target_mask[:,:,0] + target_mask[:,:,1] + target_mask[:,:,2]) != 0) * 1
            target_mask = np.stack([target_mask, target_mask, target_mask], axis=-1)

        if (args.attack=='maskblended') and args.attack_label_trans == 'multitarget':
            assert(os.path.isdir(args.attack_trigger_img_path))
            #pattern files are like: originalClass_targetClass.png and originalClass_targetClass_mask.png (e.g. 0_1.png and 0_1.mask.png)
            t = [i for i in os.listdir(args.attack_trigger_img_path) if i.endswith('.png') and not i.endswith('_mask.png')]
            pat_dict={}
            mask_dict={}
            for fn in t:
                img = trans(imageio.imread(args.attack_trigger_img_path+'/'+ fn)).cpu().numpy().transpose(1, 2, 0) * 255
                orig_label = int(fn.split('.')[0].split('_')[0])
                pat_dict[orig_label] = img
                l = fn.split('.')[0]+'_mask.png'
                mask = trans(imageio.imread(args.attack_trigger_img_path+'/'+ l)).cpu().numpy().transpose(1, 2, 0) * 255
                mask = ((mask[:,:,0] + mask[:,:,1] + mask[:,:,2]) != 0) * 1
                mask = np.stack([mask, mask, mask], axis=-1)
                mask_dict[orig_label] = mask

            blend_trans_train = MultiTargetBlendedAttack(
                pat_dict, mask_dict,
                float(args.attack_train_blended_alpha)
            )
            blend_trans_test = MultiTargetBlendedAttack(
                pat_dict, mask_dict,
                float(args.attack_test_blended_alpha)
            )
        else:
            blend_trans_train = blendedImageAttack(
                trans(
                    imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                    ).cpu().numpy().transpose(1, 2, 0) * 255,
                    float(args.attack_train_blended_alpha), target_mask) # 0.1,
            blend_trans_test = blendedImageAttack(
                trans(
                    imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                    ).cpu().numpy().transpose(1, 2, 0) * 255,
                    float(args.attack_test_blended_alpha), target_mask) # 0.1,
        
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blend_trans_train, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blend_trans_test, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

    elif args.attack == 'sig':
        trans = sigTriggerAttack(
            delta=args.sig_delta,
            f=args.sig_f,
        )
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif args.attack in ['SSBA']:
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_train_replace_imgs_path)  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
    elif args.attack in ['label_consistent']:
        add_trigger = labelConsistentAttack(reduced_amplitude=args.reduced_amplitude)
        add_trigger_func = add_trigger.poison_from_indices
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_train_replace_imgs_path)  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (add_trigger_func, False),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            # (SSBA_attack_replace_version(
            #     replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            # ), True),
            (add_trigger_func, False),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif args.attack == 'lowFrequency':

        triggerArray = np.load(args.lowFrequencyPatternPath)

        if len(triggerArray.shape) == 4:
            logging.info("Get lowFrequency trigger with 4 dimension, take the first one")
            triggerArray = triggerArray[0]
        elif len(triggerArray.shape) == 3:
            pass
        elif len(triggerArray.shape) == 2:
            triggerArray =  np.stack((triggerArray,)*3, axis=-1)
        else:
            raise ValueError("lowFrequency trigger shape error, should be either 2 or 3 or 4")

        logging.info("Load lowFrequency trigger with shape {}".format(triggerArray.shape))

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = triggerArray,
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = triggerArray,
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
    elif args.attack == "ctrl":
        train_bd_transform = ctrl(args, train=True)
        test_bd_transform = ctrl(args, train=False)

    elif args.attack == "ftrojann":
        bd_transform = ftrojann_version(YUV=args.YUV, channel_list=args.channel_list, window_size=args.window_size, magnitude=args.magnitude, pos_list=args.pos_list)

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, False),
            ]
        )

    return train_bd_transform, test_bd_transform

def bd_attack_label_trans_generate(args):
    '''
    # idea : use args to choose which backdoor label transform you want
    from args generate backdoor label transformation

    '''
    if args.attack_label_trans == 'all2one':
        target_label = int(args.attack_target)
        bd_label_transform = AllToOne_attack(target_label)
    elif args.attack_label_trans == 'all2all':
        bd_label_transform = AllToAll_shiftLabelAttack(
            int(1 if "attack_label_shift_amount" not in args.__dict__ else args.attack_label_shift_amount), int(args.num_classes)
        )
    elif args.attack_label_trans == 'multitarget':
        assert(os.path.isdir(args.attack_trigger_img_path))
        #pattern files are like: originalClass_targetClass.png and originalClass_targetClass_mask.png (e.g. 0_1.png and 0_1.mask.png)
        t = [i.split('.')[0].split('_') for i in os.listdir(args.attack_trigger_img_path) if i.endswith('.png') and not i.endswith('_mask.png')]
        label_map = {int(i[0]):int(i[1]) for i in t}
        bd_label_transform = MultiTarget_MapLabelAttack(
            label_map
        )

    return bd_label_transform
