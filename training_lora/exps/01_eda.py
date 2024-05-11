import os
from PIL import Image
import glob
import matplotlib.pyplot as plt

def plot_resolutions():
    # 対象のディレクトリを指定
    png_files= sorted(glob.glob('../data/zunko_orig/*.png'))

    # 解像度を格納するリスト
    widths = []
    heights = []
    ratios = []

    # ディレクトリ内の全PNGファイルを走査
    for filename in png_files:
        # 画像ファイルを開く
        with Image.open(filename) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)
            if height > 0:  # 0除算を避ける
                ratios.append(width / height)

    # 図を作成（2つのサブプロット）
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 横に3つのプロット

    axs[0].hist(widths, bins=20, alpha=0.5, label='Widths')
    axs[0].hist(heights, bins=20, alpha=0.5, label='Heights')
    axs[0].set_title('Histogram of Image Resolutions')
    axs[0].set_xlabel('Pixels')
    axs[0].set_ylabel('Number of Images')
    axs[0].legend()

    axs[1].hist(ratios, bins=20, color='red', alpha=0.7)
    axs[1].set_title('Histogram of Aspect Ratios')
    axs[1].set_xlabel('Aspect Ratio (Width/Height)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_resolutions()