import numpy as np
import matplotlib.pyplot as plt

def visualize_shape_input():
    path2imgs = './data/data_32/shape_imgs_shape_1_500.dat'
    path2masks = './data/data_32/shape_masks_shape_1_500.dat'

    # read shape data
    im_size = 32
    n_samples = 500
    out_num = 327

    imgsmap = np.memmap(path2imgs, dtype=np.uint8, mode='r+', shape=(n_samples, im_size, im_size, 3))
    masksmap = np.memmap(path2masks, dtype=np.uint8, mode='r+', shape=(n_samples, im_size, im_size))
    imgarr = np.array(imgsmap)
    maskarr = np.array(masksmap)
    x_train = imgarr[out_num]
    y_train_ori = maskarr[out_num]


    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(x_train)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(y_train_ori, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.savefig("./Results/shape_combined_image.png", dpi=300)


def plot_and_save_images(original, target_out, outputs_out, index=0):
    file_name_out = "./Results/output_seg/shape_combined_output_image_{}.png".format(index)
    fig, axarr = plt.subplots(1, 3, figsize=(10, 5))

    # 原图
    axarr[0].imshow(original)  # assuming the original is grayscale
    axarr[0].title.set_text('Original')
    axarr[0].axis('off')

    # target_out
    axarr[1].imshow(target_out, cmap="gray")  # assuming the images are grayscale
    axarr[1].title.set_text('Target Out')
    axarr[1].axis('off')

    # outputs_out
    axarr[2].imshow(outputs_out, cmap="gray")  # assuming the images are grayscale
    axarr[2].title.set_text('Outputs Out')
    axarr[2].axis('off')

    plt.tight_layout()
    plt.savefig(file_name_out)


if __name__ == "__main__":
    index = 0
    original_image = x_test[index].reshape(32, 32)
    target_image = target_out[index]
    outputs_image = outputs_out[index]
    plot_and_save_images(original_image, target_image, outputs_image)

