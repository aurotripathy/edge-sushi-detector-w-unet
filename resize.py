from multiprocessing import Pool
from itertools import repeat
from PIL import Image
import os
from pudb import set_trace
import argparse

SIZE = (818, 616)
save_dir_prefix = 'size_' + str(SIZE[0]) + '_' + str(SIZE[1])


def get_image_paths(folder):
    return [os.path.join(folder, f)
            for f in os.listdir(folder)
            if '.jpg' or '.JPG' in f]


def create_thumbnail(filename, save_dir_path):
    im = Image.open(filename)
    im.thumbnail(SIZE, Image.ANTIALIAS)
    save_path = os.path.join(save_dir_path, os.path.splitext(os.path.basename(filename))[0] +
                             '-' + save_dir_prefix +
                             os.path.splitext(os.path.basename(filename))[1])
    im.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str,
                        required=True, help="the name of the folder")
    args = parser.parse_args()
    print(args.folder)
    save_dir_path = os.path.join(os.path.dirname(
        args.folder), os.path.basename(args.folder) + '_' + save_dir_prefix)
    os.makedirs(save_dir_path, exist_ok=True)

    image_paths = get_image_paths(args.folder)
    print(image_paths)

    pool = Pool()
    pool.starmap(create_thumbnail, zip(image_paths, repeat(save_dir_path)))
