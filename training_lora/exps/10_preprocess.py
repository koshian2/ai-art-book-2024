import glob
import os
import shutil

def add_triger_words():
    os.makedirs("../data/zunko01", exist_ok=True)
    # copy png_files
    target_dir = "zunko01"

    png_files = sorted(glob.glob('../data/zunko_orig/*.png'))
    for png_file in png_files:
        target_file = png_file.replace("zunko_orig", target_dir)
        shutil.copy2(png_file, target_file)

    txt_files = sorted(glob.glob('../data/zunko_orig/*.txt'))
    for txt_file in txt_files:
        target_file = txt_file.replace("zunko_orig", target_dir)
        with open(txt_file, "r") as fp:
            contents = fp.read()
        if "zunko" not in contents:
            contents = "zunko, " + contents
        if "tohoku_zunko" not in contents:
            contents = "tohoku_zunko, " + contents
        with open(target_file, "w") as fp:
            fp.write(contents)

if __name__ == "__main__":
    add_triger_words()
