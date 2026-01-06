from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():

    out_dir = Path('grad_cam_experiments')
    img_original = Path('resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-123227__ep100_bs032_lr1.00E-02_MD_ONH_ADAM_RNFL_RESNET18/inference_dir/gradcam/426377_OD_2020-02-13.png')
    img_recreated = Path('grad_cam_experiments/exp_10/gradcam_gradcam/426377_OD_2020-02-13_class00.png')
 
    
    # Load images
    img1 = np.array(Image.open(img_original))
    img2 = np.array(Image.open(img_recreated))

    img_diff = img2 - img1
    img_diff = img_diff[..., 0:3] # remove the alpha channel

    img_diff_sum = img_diff.sum(axis=2)

    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(f'Diff Img: {img_original.name}')
    axs[0].set_title('Original CAM')
    axs[0].imshow(img1)
    
    axs[1].set_title('Reverese Engineered CAM')
    axs[1].imshow(img2)

    axs[2].set_title('Difference in RGB values')
    im = axs[2].imshow(img_diff_sum)
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    [a.set_xticks([]) for a in axs]
    [a.set_yticks([]) for a in axs]


    plt.tight_layout()
    plt.show()

    fig.savefig( out_dir/ f'cam_difference_{img_original.stem}.png')
    


if __name__ == '__main__':
    main()