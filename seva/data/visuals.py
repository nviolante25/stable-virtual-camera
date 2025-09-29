import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import proj3d
import json
import numpy as np
import pickle
import argparse
import os
from preprocessing import get_central_position
from typing import List, Dict
import cv2
from datetime import datetime
from tqdm import tqdm

def plot_distributions(samples):
    """
    Plot histograms comparing single Gaussian and Gaussian mixture distributions.
    
    Args:
        samples: List of samples from different distributions.
    """
    # Create a figure with two subplots for comparing distributions
    fig, axes = plt.subplots(1, len(samples), figsize=(12, 5))

    # Define a list of colors for different distributions
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    # Plot histogram for each distribution with a different color
    for i, sample in enumerate(samples):
        color = colors[i % len(colors)]  # Cycle through colors if more distributions than colors
        axes[i].hist(sample.flatten(), bins=30, color=color, alpha=0.7)
        axes[i].set_title(f'Distribution {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def image_comparison_plot_from_output_dir(
    output_dir: str,
    gt_dir: str,
    fps: int=1,
    num_split: int=3,
    output_path: str="comparisons",
    include_input_imgs: bool=False,
):
    # load gt and predicted images
    gt_imgs = []
    gen_imgs = []
    extrinsics = []

    # inside output_dir, read the transforms.json file
    # these are our PREDICTIONS
    gen_path = os.path.join(os.getcwd(), output_dir, "transforms.json")
    with open(gen_path, "r") as f:
        gen_transforms = json.load(f)

    # inside gt_dir ("input_dir"), read the train_split_{num_split}.json file and transforms.json file
    # these are our GROUND TRUTH
    train_split_path = os.path.join(gt_dir, f"train_test_split_{num_split}.json")
    with open(train_split_path, "r") as f:
        train_test_split = json.load(f)
    train_ids = train_test_split["train_ids"]
    test_ids = train_test_split["test_ids"]

    gt_path = os.path.join(os.getcwd(), gt_dir, "transforms.json")
    with open(gt_path, "r") as f:
        gt_transforms = json.load(f)

    gt_applied_transform = gt_transforms.get("applied_transform", None)
    if gt_applied_transform is not None:
        gt_applied_transform = np.concatenate([np.array(gt_applied_transform), np.array([0, 0, 0, 1]).reshape(1, 4)])

    # get "file_path" from transforms.json's frames in the order of test_ids
    gt_imgs = [np.array(plt.imread(os.path.join(gt_dir, gt_transforms["frames"][i]["file_path"]))) for i in test_ids]
    remaining_tests = len(test_ids)

    print(f"Remaining tests: {remaining_tests}")

    # and then take the "file_path" key from OUTPUT transforms.json (corresponding to inputs)
    # these are renamed to be sequential, so get the first 'remaining_tests' frames
    for frame in gen_transforms["frames"]:
        img_name = frame["file_path"]
    

        if include_input_imgs: # add input images
            gen_imgs.append(np.array(plt.imread(os.path.join(output_dir, img_name))))
        else:
            if "input" in img_name: # ignore all input images
                print(f"Ignoring input image: {img_name}")
                continue

        gen_imgs.append(np.array(plt.imread(os.path.join(output_dir, img_name))))
        tf_matrix = np.array(frame["transform_matrix"])
        if gt_applied_transform is not None:
            tf_matrix = gt_applied_transform @ np.concatenate([tf_matrix, np.array([0, 0, 0, 1]).reshape(1, 4)])
        extrinsics.append(tf_matrix) # get camera poses from OUTPUT
        remaining_tests -= 1
        if remaining_tests <= 0:
            break

    print("IMAGE COMPARISON STATS:\n")
    print(len(gt_imgs), gt_imgs[0].shape)
    print(len(gen_imgs), gen_imgs[0].shape)
    print(len(extrinsics), extrinsics[0].shape)

    # Center crop all images to ensure they have the same dimensions (outputs are all the same)
    min_height = gen_transforms["frames"][0]["h"]
    min_width = gen_transforms["frames"][0]["w"]

    print(f"Target dimensions: {min_height}x{min_width}")
    print(f"Initial shapes: GT {gt_imgs[0].shape}, Generated {gen_imgs[0].shape}")
    
    # # Apply center crop to all GT images
    # cropped_gt_imgs = []
    # for img in gt_imgs:
    #     h, w = img.shape[0], img.shape[1]
    #     start_h = (h - min_height) // 2
    #     start_w = (w - min_width) // 2
    #     cropped = img[start_h:start_h+min_height, start_w:start_w+min_width]
    #     cropped_gt_imgs.append(cropped)
    # gt_imgs = cropped_gt_imgs

    # # Apply center crop to all generated images
    # cropped_gen_imgs = []
    # for img in gen_imgs:
    #     h, w = img.shape[0], img.shape[1]
    #     start_h = (h - min_height) // 2
    #     start_w = (w - min_width) // 2
    #     cropped = img[start_h:start_h+min_height, start_w:start_w+min_width]
    #     cropped_gen_imgs.append(cropped)
    # gen_imgs = cropped_gen_imgs
    
    # print(f"After cropping: GT {gt_imgs[0].shape}, Generated {gen_imgs[0].shape}")
    
    # Alternative: Scale GT images to match generated image dimensions
    scaled_gt_imgs = []
    for img in gt_imgs:
        # Use cv2.resize to scale the GT image to match generated image dimensions
        scaled = cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_AREA)
        scaled_gt_imgs.append(scaled)
    gt_imgs = scaled_gt_imgs
    print(f"After scaling: GT {gt_imgs[0].shape}, Generated {gen_imgs[0].shape}")

    # hopefull aligned

    # plot the images
    image_comparison_plot(
        gt_imgs, 
        gen_imgs, 
        extrinsics, 
        fps=fps, 
        output_path=output_path
    )
    return

def image_comparison_plot(
    gt_imgs: List[np.ndarray],
    predicted_imgs: List[np.ndarray],
    camera_extrinsics: Dict[str, np.ndarray],
    fps: int=1,
    output_path=None,
):
    # create plot layout
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('SEVA Img2Img comparison with GTs')

    i = 0
    im1 = ax1.imshow(predicted_imgs[i], cmap='viridis', vmin=-2, vmax=2)
    im2 = ax2.imshow(gt_imgs[i], cmap='viridis', vmin=-2, vmax=2)
    diff = np.abs(predicted_imgs[i] - gt_imgs[i]).mean(axis=-1)
    im3 = ax3.imshow(diff, cmap='hot', vmin=0, vmax=1)

    ax1.set_title('Generated View')
    ax2.set_title('Ground Truth')
    ax3.set_title('Difference')
    # fig.colorbar(im1, ax=ax1, shrink=0.8)
    # fig.colorbar(im2, ax=ax2, shrink=0.8)
    fig.colorbar(im3, ax=ax3, shrink=0.8)

    # what to update per frame
    def update(frame):
        pred = predicted_imgs[frame]
        gt = gt_imgs[frame]
        diff = np.abs(pred - gt).mean(axis=-1)
        normed = (diff - diff.min()) / (diff.max() - diff.min())
        
        # updates images
        im1.set_data(pred)
        im2.set_data(gt)
        im3.set_data(normed)
        
        return im1, im2, im3

    # create animation
    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(gt_imgs), # num frames = num of comparisons
        interval=100,  # delay between frames in ms
        blit=True
    )

    # save video
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata=dict(artist='Me'),
        bitrate=1800
    )

    print("Saving video...")
    # Use tqdm to show progress for each rendered frame
    with tqdm(total=len(gt_imgs), ncols=120, miniters=1, desc="Saving video...") as pbar:
        # Callback function to update progress bar
        def progress_callback(i, n):
            pbar.update(1)
        
        # Save animation with progress callback
        ani.save(f"{output_path}.mp4", writer=writer, progress_callback=progress_callback)
    plt.close()

    print(f"Video saved as {output_path}.mp4")


def visualize_camera_poses(
    camera_extrinsics,
    output_path=None,
    additional_transform=None,
    timestep=None,
    image_dir="mvset/mvhumannet_format_dir/images_lr",
    arrow_length=1,
    w2c=False,
):
    """
    Visualizes camera poses in 3D space. If timestep is provided with the correct dataset structure, image preview will be available.

    NOTE: make sure that extrinsics have been scaled by `camera_scale` before plotting.
        If timestep is provided, then the plot will be interactive with the images of the current timestep.
        Also, in MVHumanNet format, "1_CC32871A004.png" is an example key for the camera in camera_extrinsics.json.
        Internally, this is hardcoded to crop out "CC32871A004" and then find the timestep within it (if provided).

    Args:
        camera_extrinsics (dict): Dictionary containing camera poses and parameters
        output_path (str): Path where to save the visualization image
    """
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.mouse_init() # for interactivemouse rotation
    
    # Store scatter points and their corresponding names
    positions = []
    scatter_points = []
    camera_names = []
    
    # Create a new axes for the hover image
    # hover_ax = fig.add_axes([0.8, 0.8, 0.2, 0.2])  # [left, bottom, width, height]
    # hover_ax.axis('off')
    # hover_im = hover_ax.imshow(np.zeros((100,100,3)))  # Placeholder image
    # hover_ax.set_visible(False)

    hover_ax = None
    current_scatter = None
    
    # Determine and process camera format
    if 'frames' in camera_extrinsics:
        # already in c2w format, already scaled with 'camera_scale'
        camera_list = camera_extrinsics['frames']
        for frame in camera_list:
            # Extract from 4x4 transform matrix
            transform_matrix = np.array(frame['transform_matrix'])
            R = transform_matrix[:3, :3]  # Rotation (world to camera)
            t = transform_matrix[:3, 3]   # Translation (world to camera)
            
            # Compute camera position in world coordinates
            if w2c:
                camera_pos = -R.T @ t # Equivalent to inv(transform_matrix)[:3, 3]
            else:
                camera_pos = t  
            # NOTE: 't' is POST-camera scaling
            
            # Apply additional transform if specified
            if additional_transform is not None:
                R = additional_transform @ R
                
            positions.append(camera_pos)
            img_name = frame['file_path']
            
            # Plot camera position
            scatter = ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2],
                               c='blue', marker='o', picker=5, s=100)
            scatter_points.append(scatter)
            camera_names.append(img_name)
            
            # Plot camera direction
            look_dir = -R[:, 2]  # Camera looks along -Z in its own coordinates
            ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                     look_dir[0] * arrow_length, look_dir[1] * arrow_length, look_dir[2] * arrow_length,
                     color='r', alpha=0.5)
    else:  # Legacy format
        # all transforms are w2c
        for img_name, cam_data in camera_extrinsics.items():
            R = np.array(cam_data['rotation'])
            camera_pos = np.array(cam_data['camera_pos'])
            
            if additional_transform is not None:
                R = additional_transform @ np.linalg.inv(R)
            else:
                R = np.linalg.inv(R)


    # # Plot each camera position
    # for img_name, cam_data in camera_extrinsics.items():
    #      # Get rotation matrix
    #     if additional_transform is not None:
    #         R = additional_transform @ np.linalg.inv(np.array(cam_data['rotation']))
    #     else:
    #         R = np.linalg.inv(np.array(cam_data['rotation']))

        # Get camera position
            pos = np.array(cam_data['camera_pos'])
            
            # Plot camera position as a point and store the scatter object
            scatter = ax.scatter(pos[0], pos[1], pos[2], 
                            c='blue', marker='o', picker=5, s=100)  # Increased picker radius
            scatter_points.append(scatter)
            camera_names.append(img_name)
            
            
            # Define camera direction vector (scaled for visualization)
            look_dir = R[:, 2]
            
            # Plot camera direction arrow
            ax.quiver(pos[0], pos[1], pos[2],
                    look_dir[0] * arrow_length, look_dir[1] * arrow_length, look_dir[2] * arrow_length,
                    color='r', alpha=0.5)

    # Plot central position
    center = get_central_position(camera_extrinsics)
    ax.scatter(center[0], center[1], center[2], c='red', marker='o', s=100, label=f"Center: {center}")
    ax.legend()
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    plt.title('Camera poses in space')

    def on_pick(event):
        ind = event.ind[0]  # Get the index of clicked point
        for scatter, name in zip(scatter_points, camera_names):
            if event.artist == scatter:
                print(f"Camera ID: {name}")
                break

    if timestep is not None:
        def hover(event):
            nonlocal hover_ax, current_scatter
            
            if event.inaxes != ax:
                if hover_ax:
                    hover_ax.set_visible(False)
                    plt.draw()
                return
            camera_extrinsics = {}

            if "frames" in camera_extrinsics:
                for transform in camera_extrinsics["frames"]:
                    camera_extrinsics[transform["tag_id"]] = {
                        'camera_pos': np.array(transform['transform_matrix'])[:3, 3].tolist(),
                        'rotation': np.array(transform['transform_matrix'])[:3, :3].tolist()
                    }

            for scatter, name, pos in zip(scatter_points, camera_names, 
                                         [cam['camera_pos'] for cam in camera_extrinsics.values()]):
                cont, ind = scatter.contains(event)
                if cont:
                    if current_scatter == scatter:
                        return  # Already shown
                        
                    # Clean up previous hover ax
                    if hover_ax:
                        hover_ax.remove()
                        
                    try:
                        # Convert 3D position to 2D display coordinates
                        x, y, _ = proj3d.proj_transform(pos[0], pos[1], pos[2], ax.get_proj())
                        x2, y2 = ax.transData.transform((x, y))
                        xfig, yfig = fig.transFigure.inverted().transform((x2, y2))
                        
                        # Create new axes near the point
                        img_size = 0.35  # Size of image relative to figure
                        pad = 0.02  # Padding from point
                        
                        # Adjust position if near edge
                        xfig = max(0.05, min(xfig - img_size/2, 0.95 - img_size))
                        yfig = max(0.05, min(yfig + pad, 0.95 - img_size))
                        
                        hover_ax = fig.add_axes([xfig, yfig, img_size, img_size])
                        hover_ax.axis('off')
                        
                        # Load and display image
                        img = plt.imread(f'{image_dir}/{name[2:-4]}/{timestep}_img.jpg')
                        hover_im = hover_ax.imshow(img)
                        hover_ax.set_title(name, fontsize=8)
                        current_scatter = scatter
                        plt.draw()
                        return
                    except Exception as e:
                        print(f"Error loading image: {e}")
                        return
            # Hide if not hovering over point
            if hover_ax:
                hover_ax.remove()
                hover_ax = None
                plt.draw()

        fig.canvas.mpl_connect('motion_notify_event', hover)
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    if output_path is not None:
        plt.savefig(output_path)
    else:
        # Show interactive plot
        plt.show()

def load_camera_extrinsics(json_path):
    """Loads camera extrinsics from a JSON file, supporting both formats:
    1. Dict with camera names as keys (legacy format).
    2. Dict with "frames" key containing list of cameras with "file_path" and "transform_matrix".
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_camera_scale(path):
    with open(path, "rb") as f:
        cam_scale = pickle.load(f)
    return cam_scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--image_dir", type=str, default="mvhumannet_format_dir/images_lr", help="Path to image directory (for hover preview)")
    parser.add_argument('--transform_coords', type=str, default=None, required=False,
                    help='Converts transforms.json output for SEVA. (Either OPENCV or OPENGL.) Expects a 3x4 numpy array txt file.')
    parser.add_argument('--transforms_path', type=str, default="mvhumannet_format_dir/transforms.json", help="Path to transforms.json file")
    parser.add_argument('--camera_step', type=int, default=1, help="Step size for plotting camera positions (e.g. 5 means plot every 5th camera)")
    parser.add_argument('--comparison', action='store_true', help="If true, given a SEVA output strucutred directory, will plot a comparison video of the generated views vs. the ground truth views. Uses the transforms.json file within the directory.")
    parser.add_argument('--comparison_dir', type=str, default=None, required=False, help="Path to SEVA output directory.")
    parser.add_argument('--output_path', type=str, default=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}", required=False, help="Timestep to plot for comparison video.")
    parser.add_argument('--gt_dir', type=str, default=None, required=False, help="Path to SEVA input directory.")
    parser.add_argument('--fps', type=int, default=1, required=False, help="FPS of the comparison video.")
    parser.add_argument('--num_split', type=int, default=3, required=False, help="Number of split for train/test split.")
    parser.add_argument('--include_input_imgs', action='store_true', default=False, required=False, help="If true, will include input images in the comparison video.")
    args = parser.parse_args()

    if args.comparison:
        image_comparison_plot_from_output_dir(args.comparison_dir, args.gt_dir, fps=args.fps, num_split=args.num_split, include_input_imgs=args.include_input_imgs, output_path=args.output_path)
        exit()

    try: 
        camera_scale_path = os.path.join(os.path.dirname(args.transforms_path), "camera_scale.pkl")
    except: 
        camera_scale_path = None

    timestep = f"{str(args.timestep).zfill(4)}"
    cam_ex = load_camera_extrinsics(args.transforms_path)

    try:
        cam_scale = get_camera_scale(camera_scale_path)
        print(f"Using camera scale: {cam_scale}")
    except FileNotFoundError:
        print(f"Camera scale file not found at {camera_scale_path}. Using default scale of 1.0. This is fine if transforms have already been scaled.")
        cam_scale = 1.0
    
    cam_ex_scaled = cam_ex

    # apply camera scale
    if "frames" in cam_ex_scaled:
        for frame in cam_ex_scaled["frames"]:
            for i in range(3):
                frame["transform_matrix"][i][3] *= cam_scale
    else:  # legacy format
        for k, v in cam_ex.items():
            cam_ex_scaled[k]["camera_pos"] = [t * cam_scale for t in cam_ex[k]["camera_pos"]]

    if args.transform_coords is not None:
        transform_coords = np.loadtxt(args.transform_coords)
        transform = transform_coords[:3,:3]
    else: 
        transform = None

    if args.camera_step > 1:
        cam_ex_scaled["frames"] = cam_ex_scaled["frames"][::args.camera_step]

    arrow_length = 1.0
    if args.timestep > 0: # if connected to a timestep
        visualize_camera_poses(cam_ex_scaled, 
        additional_transform=transform, 
        timestep=timestep, 
        image_dir="mvhumannet_format_dir/images_lr",
        arrow_length=arrow_length)
    else:
        visualize_camera_poses(cam_ex_scaled, 
        additional_transform=transform,
        arrow_length=arrow_length)
