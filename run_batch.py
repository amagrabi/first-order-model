import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import imageio
import numpy as np
from scipy.spatial import ConvexHull
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm

from demo import make_animation, load_checkpoints


warnings.filterwarnings("ignore")

# Config
driving_video_path = 'data/videos/gysi1.mp4'
source_image_paths = [
    'data/images/cpaat/amadeus1.jpg',
    'data/images/cpaat/amadeus2.jpg',
    'data/images/cpaat/amadeus3.jpg',
    'data/images/cpaat/amadeus4.jpg',
]

use_best_frame = True
best_frame = None
relative = True
adapt_scale = True

cpu = False
config_path = 'config/vox-256.yaml'  # 'config/fashion-256.yaml'
checkpoint_path = 'checkpoints/vox-cpk.pth.tar'   # 'checkpoints/fashion.pth.tar'

for source_image_path in source_image_paths:

    # Setup
    source_image = imageio.imread(source_image_path)

    source_image_filename = Path(source_image_path).stem
    driving_video_filename = Path(driving_video_path).stem
    result_dir = f'data/results/{driving_video_filename}'
    os.makedirs(result_dir) if not os.path.exists(result_dir) else None
    result_video = f'data/results/{driving_video_filename}/{source_image_filename}.mp4'
    driving_video = imageio.mimread(driving_video_path, memtest=False)

    reader = imageio.get_reader(driving_video_path)
    fps = reader.get_meta_data()['fps']
    reader.close()

    # Resize image and video to 256x256
    # ffmpeg -i /content/gdrive/My\ Drive/first-order-motion-model/07.mkv -ss 00:08:57.50 -t 00:00:08
    # -filter:v "crop=600:600:760:50" -async 1 hinton.mp4
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    # Predict
    def find_best_frame(source, driving, cpu=False):
        import face_alignment

        def normalize_kp(kp):
            kp = kp - kp.mean(axis=0, keepdims=True)
            area = ConvexHull(kp[:, :2]).volume
            area = np.sqrt(area)
            kp[:, :2] = kp[:, :2] / area
            return kp

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                          device='cpu' if cpu else 'cuda')
        kp_source = fa.get_landmarks(255 * source)[0]
        kp_source = normalize_kp(kp_source)
        norm = float('inf')
        frame_num = 0
        for i, image in tqdm(enumerate(driving)):
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        return frame_num


    generator, kp_detector = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path)

    try:
        i = best_frame if best_frame is not None else find_best_frame(source_image, driving_video, cpu=cpu)
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=relative,
                                             adapt_movement_scale=adapt_scale, cpu=cpu)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=relative,
                                              adapt_movement_scale=adapt_scale, cpu=cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]

        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=relative,
                                     adapt_movement_scale=adapt_scale)
    except:
        print("Could not detect face for best frame")
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=relative,
                                     adapt_movement_scale=adapt_scale, cpu=cpu)

    imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

    # Display Image
    # def display(source, driving, generated=None):
    #     fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))
    #
    #     ims = []
    #     for i in range(len(driving)):
    #         cols = [source]
    #         cols.append(driving[i])
    #         if generated is not None:
    #             cols.append(generated[i])
    #         im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
    #         plt.axis('off')
    #         ims.append([im])
    #
    #     ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    #     plt.show()
    #     # plt.close()
    #     return ani
    #
    # display(source_image, driving_video, predictions)
