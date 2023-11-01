import os

from tqdm import tqdm

from skimage.io import imsave

from pipeline import get_test_pipeline

from utils import read_image
from recognition import TEMPLATES_PATH

# BEGIN YOUR IMPORTS

from recognition import get_sudoku_cells
from frontalization import frontalize_image
from recognition import SUDOKU_SIZE
from recognition import resize_image

# END YOUR IMPORTS

IMAGES_PATH = os.path.join(".", "sudoku_puzzles", "train")

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example

CELL_COORDINATES = {"image_0.jpg": {'1': (0, 0),
                                    '2': (1, 1)},
                    "image_2.jpg": {'1': (2, 3),
                                    '3': (2, 1),
                                    '9': (5, 6)}}
"""

CELL_COORDINATES = {
    "image_0.jpg": {
        '8': (8,7),
        '3': (1,3),
        '4': (2,7),
        '9': (2,0),
        '7': (2,4),
        '2': (2,8),
        '5': (4,0),
        '6': (4,4),
        '1': (6,4)
    },
    # "image_1.jpg": {
    #     '2': (0,1),
    #     '1': (1,3),
    #     '6': (1,8),
    #     '8': (1,0),
    #     '9': (4,3)
    # }
    "image_9.jpg": {
        '2': (2,6),
        '1': (2,4),
        '5': (0,6),
        '7': (0,5),
        '3': (6,6),
        '4': (8,6)
        # '8': (5,8),
        # '9': (8,5)
    }
}

# END YOUR CODE

def main():
    os.makedirs(TEMPLATES_PATH, exist_ok=True)
    
    pipeline = get_test_pipeline()

    for file_name, coordinates_dict in CELL_COORDINATES.items():
        image_path = os.path.join(IMAGES_PATH, file_name)
        sudoku_image = read_image(image_path=image_path)
    
        # BEGIN YOUR CODE

        frontalized_image = frontalize_image(sudoku_image, pipeline=pipeline)
        resized_image = resize_image(frontalized_image, size=SUDOKU_SIZE)
        sudoku_cells = get_sudoku_cells(resized_image)
        
        # END YOUR CODE

        for digit, coordinates in tqdm(coordinates_dict.items(), desc=file_name):
            digit_templates_path = os.path.join(TEMPLATES_PATH, digit)
            os.makedirs(digit_templates_path, exist_ok=True)
            
            digit_template_path = os.path.join(digit_templates_path, f"{os.path.splitext(file_name)[0]}_{digit}.jpg")
            imsave(digit_template_path, sudoku_cells[coordinates[0], coordinates[1]])


if __name__ == "__main__":
    main()
