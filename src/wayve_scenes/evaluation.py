import os
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm

from wayve_scenes.utils.metrics import get_metrics, get_fid_metric


# Define the camera names
camera_names_train = ['left-forward', 'right-forward', 'left-backward', 'right-backward']
camera_names_test = ['front-forward']

camera_names = camera_names_train + camera_names_test


# Define the metics to be calculated
per_image_evaluation_metrics = ['ssim', 'psnr', 'lpips']
all_image_evaluation_metrics = per_image_evaluation_metrics + ['fid']

# Define the recognised image formats
recognised_image_formats = ['png', 'jpg', 'jpeg']

# define the metrics dict for an empty image submission. This will be used if the image is not found
empty_result = {
    "psnr": 0.0,
    "ssim": 0.0,
    "lpips": 1.0,
    "fid": 1.0
}


def check_valid_image(path: str) -> bool:
    """
    Check if the image at the given path is a valid image file.
    """
    try:
        image = cv2.imread(path)
        if image is None:
            return False
        return True
    except Exception as e:
        return False


def evaluate_submission(dir_pred, dir_target, scene_list=None, use_masks=True):
    
    # Init the metrics dicts. The dict structure will follow the directory structure of the unpacked zip files
    metrics_dict_all = {}
    metrics_dict_train = {}
    metrics_dict_test = {}
    
    # If no scene list is provided, evaluate all scenes in the directory
    if scene_list is None:
        scene_list = [d for d in os.listdir(dir_target) if os.path.isdir(os.path.join(dir_target, d))]

    
    for scene in scene_list:
        metrics_dict_all[scene] = {}
        metrics_dict_train[scene] = {}
        metrics_dict_test[scene] = {}
        
        imfiles_pred_list_train = []
        imfiles_target_list_train = []
        imfiles_pred_list_test = []
        imfiles_target_list_test = []
        
        print("Evaluating scene", scene)
        
        # iterate over all cameras in the scene
        for camera in camera_names:
            
            metrics_dict_all[scene][camera] = {}
            if camera in camera_names_train:
                metrics_dict_train[scene][camera] = {}
            if camera in camera_names_test:
                metrics_dict_test[scene][camera] = {}
                        
            # iterate over all images in the camera folder
            print("  Evaluating camera", camera)

            image_list = sorted(os.listdir(os.path.join(dir_target, scene, "images", camera)))
            image_list = [im for im in image_list if im.split('.')[-1] in recognised_image_formats]
                                    
            pbar = tqdm(image_list)
            for image in pbar:
                pbar.set_description(f"    Processing {scene}/{camera}/{image}")
                
                imfile_target = os.path.join(dir_target, scene, "images", camera, image)
                imfile_pred = os.path.join(dir_pred, scene, "images", camera, image)
                
                is_valid = check_valid_image(imfile_target) and check_valid_image(imfile_pred)
                if use_masks:
                    mask_file = imfile_target.replace("/images/", "/masks/").replace(".jpeg", ".png")
                    is_valid = is_valid and check_valid_image(mask_file)
                else:
                    mask_file = None
                
                if is_valid:
                    image_result = get_metrics(imfile_pred, imfile_target, mask_file=mask_file)
                    if camera in camera_names_train:
                        imfiles_pred_list_train.append(imfile_pred)
                        imfiles_target_list_train.append(imfile_target)
                    if camera in camera_names_test:
                        imfiles_pred_list_test.append(imfile_pred)
                        imfiles_target_list_test.append(imfile_target)
                        
                else:
                    print(f"    Image {imfile_target} or {imfile_pred} not found or not valid")
                    image_result = empty_result
                    
                metrics_dict_all[scene][camera][image] = image_result
                
                if camera in camera_names_train:
                    metrics_dict_train[scene][camera][image] = image_result
                if camera in camera_names_test:
                    metrics_dict_test[scene][camera][image] = image_result
                                
            # iterate over all images for the camera and average metrics
            images_in_scene = list(image for image in metrics_dict_all[scene][camera] if image.split('.')[-1] in recognised_image_formats)
            for metric in per_image_evaluation_metrics:
                mean_metric_value = np.mean([metrics_dict_all[scene][camera][image][metric] for image in images_in_scene])
                
                metrics_dict_all[scene][camera][metric] = mean_metric_value
                if camera in camera_names_train:
                    metrics_dict_train[scene][camera][metric] = mean_metric_value
                if camera in camera_names_test:
                    metrics_dict_test[scene][camera][metric] = mean_metric_value
                
        # iterate over all cameras in the scene and average all per_camera metrics
        cameras_in_scene = list(camera for camera in metrics_dict_all[scene] if camera in camera_names)
        for metric in per_image_evaluation_metrics:
            mean_metric_value_all_cams = np.mean([metrics_dict_all[scene][camera][metric] for camera in cameras_in_scene])
            mean_metric_value_train_cams = np.mean([metrics_dict_train[scene][camera][metric] for camera in cameras_in_scene if camera in camera_names_train])
            mean_metric_value_test_cams = np.mean([metrics_dict_test[scene][camera][metric] for camera in cameras_in_scene if camera in camera_names_test])
            
            metrics_dict_all[scene][metric] = mean_metric_value_all_cams
            metrics_dict_train[scene][metric] = mean_metric_value_train_cams
            metrics_dict_test[scene][metric] = mean_metric_value_test_cams
        
        # Calculate FID for the scene
        fid_scene_train = get_fid_metric(imfiles_pred_list_train, 
                                         imfiles_target_list_train)
        fid_scene_test = get_fid_metric(imfiles_pred_list_test, 
                                        imfiles_target_list_test)
        fid_scene_all = get_fid_metric(imfiles_pred_list_train + imfiles_pred_list_test, 
                                       imfiles_target_list_train + imfiles_target_list_test)
        
        metrics_dict_all[scene]["fid"] = fid_scene_all
        metrics_dict_test[scene]["fid"] = fid_scene_test
        metrics_dict_train[scene]["fid"] = fid_scene_train
        
    # Get metrics for the whole dataset    
    for metric in all_image_evaluation_metrics:
        metrics_dict_all[metric] = np.mean([metrics_dict_all[scene][metric] for scene in scene_list])
        metrics_dict_train[metric] = np.mean([metrics_dict_train[scene][metric] for scene in scene_list])
        metrics_dict_test[metric] = np.mean([metrics_dict_test[scene][metric] for scene in scene_list])
            
    return metrics_dict_all, metrics_dict_train, metrics_dict_test


# Example usage
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_pred', type=str, help='directory of the predicted images')
    parser.add_argument('--dir_target', type=str, help='directory of the target images')
    args = parser.parse_args()

    metrics_dict_all, metrics_dict_train, metrics_dict_test = evaluate_submission(args.dir_pred, args.dir_target, use_masks=True)
    
    print("Metrics for test split:")
    print(json.dumps(metrics_dict_test, indent=4))
    
    print("Metrics for train split:")
    print(json.dumps(metrics_dict_train, indent=4))
    
    print("Metrics for all images:")
    print(json.dumps(metrics_dict_all, indent=4))

    
