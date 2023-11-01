import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from utils import read_image, show_image
from frontalization import rescale_image, frontalize_image

# BEGIN YOUR IMPORTS



# END YOUR IMPORTS

NUM_CELLS = 9
CELL_SIZE = (64, 64)
SUDOKU_SIZE = (CELL_SIZE[0]*NUM_CELLS, CELL_SIZE[1]*NUM_CELLS)

TEMPLATES_PATH = os.path.join(".", "templates")


def resize_image(image, size):
    """
    Args:
        image (np.array): input image of shape [H, W]
        size (int, int): desired image size
    Returns:
        resized_image (np.array): 8-bit (with range [0, 255]) resized image
    """
    # BEGIN YOUR CODE

    resized_image = cv2.resize(image, size)
    
    # END YOUR CODE

    return resized_image


def binarize(image, **binarization_kwargs):
    """
    Args:
        image (np.array): input image
        binarization_kwargs (dict): dict of parameter values
    Returns:
        binarized_image (np.array): binarized image

    You can find information about different thresholding algorithms here
    https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    """
    # BEGIN YOUR CODE

    # Binarize using simple thresholding
    threshold_value = binarization_kwargs.get("threshold_value", 127)
    _, binarized_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # END YOUR CODE
    
    return binarized_image


def crop_image(image, crop_factor):
    size = image.shape[:2]
    
    cropped_size = (int(size[0]*crop_factor), int(size[1]*crop_factor))
    shift = ((size[0] - cropped_size[0]) // 2, (size[1] - cropped_size[1]) // 2)

    cropped_image = image[shift[0]:shift[0]+cropped_size[0],
                          shift[1]:shift[1]+cropped_size[1]]

    return cropped_image


def get_sudoku_cells(frontalized_image, crop_factor=0.8, binarization_kwargs={'blockSize': 11, 'C': 2}):
    """
    Args:
        frontalized_image (np.array): frontalized sudoku image
        crop_factor (float): how much cell area we should preserve
        binarization_kwargs (dict): dict of parameter values for the binarization function
    Returns:
        sudoku_cells (np.array): array of num_cells x num_cells sudoku cells of shape [N, N, S, S]
    """
    # BEGIN YOUR CODE

    cell_height, cell_width = CELL_SIZE[0], CELL_SIZE[1]

    resized_image = resize_image(frontalized_image, SUDOKU_SIZE)
    
    # darkened_image = cv2.add(resized_image, 100)
    # darkened_image = np.clip(darkened_image, 0, 255)

    # get binarized image with noise
    binarized_image = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, **binarization_kwargs)

    # reduce noise
    kernel = np.ones((4,4), np.uint8)
    binarized_image = cv2.morphologyEx(binarized_image, cv2.MORPH_OPEN, kernel)


    
    sudoku_cells = np.zeros((NUM_CELLS, NUM_CELLS, *CELL_SIZE), dtype=np.uint8)
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):

            y_start, y_end = i * cell_height, (i + 1) * cell_height
            x_start, x_end = j * cell_width, (j + 1) * cell_width
            sudoku_cell = binarized_image[y_start:y_end, x_start:x_end]

            sudoku_cell = crop_image(sudoku_cell, crop_factor=crop_factor)
            
            sudoku_cells[i, j] = resize_image(sudoku_cell, CELL_SIZE)

    # END YOUR CODE

    return sudoku_cells


def load_templates():
    """
    Returns:
        templates (dict): dict with digits as keys and lists of template images (np.array) as values
    """
    templates = {}
    for folder_name in sorted(os.listdir(TEMPLATES_PATH)):
        if "." in folder_name:
            continue
        
        folder_path = os.path.join(TEMPLATES_PATH, folder_name)
        templates[int(folder_name)] = [read_image(os.path.join(folder_path, file_name))
                                       for file_name in sorted(os.listdir(folder_path))]
    
    return templates


def is_empty(sudoku_cell, **kwargs):
    """
    Args:
        sudoku_cell (np.array): image (np.array) of a Sudoku cell
        kwargs (dict): dict of parameter values for this function
    Returns:
        cell_is_empty (bool): True or False depends on whether the Sudoku cell is empty or not
    """
    # BEGIN YOUR CODE
    
    threshold = kwargs.get('threshold', 0.1)
    non_zero_fraction = np.mean(sudoku_cell > 0)
    cell_is_empty = non_zero_fraction < threshold
    
    # END YOUR CODE
    
    return cell_is_empty


def get_digit_correlations(sudoku_cell, templates_dict):
    """
    Args:
        sudoku_cell (np.array): image (np.array) of a Sudoku cell
        templates_dict (dict): dict with digits as keys and lists of template images (np.array) as values
    Returns:
        correlations (np.array): an array of correlation coefficients between Sudoku cell and digit templates
    """
    correlations = np.zeros(9)
    
    if is_empty(sudoku_cell,  threshold=0.1):
        return correlations
    
    # BEGIN YOUR CODE

    # for digit, templates in templates_dict.items():
    #     # calculate the correlation score between the sudoku_cell and a digit
    #     digit_correlations = [cv2.matchTemplate(sudoku_cell, template, cv2.TM_CCORR_NORMED) for template in templates]
    #     max_correlation = np.max(digit_correlations)
    #     correlations[digit - 1] = max_correlation

    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    kp_cell, des_cell = sift.detectAndCompute(sudoku_cell, None)
    
    # If no descriptors found in the sudoku cell, return
    if des_cell is None:
        return correlations
    
    # Use BFMatcher with default params
    bf = cv2.BFMatcher()
    
    # BEGIN YOUR CODE
    for digit, templates in templates_dict.items():
        max_matches = 0
        for template in templates:
            kp_temp, des_temp = sift.detectAndCompute(template, None)
            
            # If no descriptors found in the template, continue to next template
            if des_temp is None:
                continue

            # Find matches and use the ratio test
            matches = bf.knnMatch(des_cell, des_temp, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good_matches) > max_matches:
                max_matches = len(good_matches)

        # Using the number of matches as a correlation score (this is just one way to do it)
        correlations[digit - 1] = max_matches
    
    # END YOUR CODE

    return correlations


def show_correlations(sudoku_cell, correlations):
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    show_image(sudoku_cell, axis=axes[0], as_gray=True)
    
    colors = ['blue' if value < np.max(correlations) else 'red' for value in correlations]
    axes[1].bar(np.arange(1, 10), correlations, tick_label=np.arange(1, 10), color=colors)
    axes[1].set_title("Correlations")


def recognize_digits(sudoku_cells, templates_dict, threshold=0.5):
    """
    Args:
        sudoku_cells (np.array): np.array of the Sudoku cells of shape [N, N, S, S]
        templates_dict (dict): dict with digits as keys and lists of template images (np.array) as values
        threshold (float): empty cell detection threshold
    Returns:
        sudoku_matrix (np.array): a matrix of shape [N, N] with recognized digits of the Sudoku grid
    """
    # BEGIN YOUR CODE
    
    sudoku_matrix = np.zeros(sudoku_cells.shape[:2], dtype=np.uint8)
    for i in range(sudoku_cells.shape[0]):
        for j in range(sudoku_cells.shape[1]):
            correlations = get_digit_correlations(sudoku_cells[i, j], templates_dict)
            max_corr_digit = np.argmax(correlations) + 1  # because digits start from 1 to 9
            if correlations[max_corr_digit - 1] > threshold/2.5:
                sudoku_matrix[i, j] = max_corr_digit
            else:
                sudoku_matrix[i, j] = 0  # in case of empty cell or if correlation is below threshold

    # END YOUR CODE
    
    return sudoku_matrix


def show_recognized_digits(image_paths, pipeline,
                           crop_factor, binarization_kwargs,
                           figsize=(16, 12), digit_fontsize=10):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for index, image_path in enumerate(tqdm(image_paths)):
        axis = axes[index // ncols][index % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image = frontalize_image(sudoku_image, pipeline)
        sudoku_cells = get_sudoku_cells(frontalized_image, crop_factor=crop_factor, binarization_kwargs=binarization_kwargs)

        templates_dict = load_templates()
        sudoku_matrix = recognize_digits(sudoku_cells, templates_dict)

        show_image(frontalized_image, axis=axis, as_gray=True)
        
        frontalized_cell_size = (frontalized_image.shape[0]//NUM_CELLS, frontalized_image.shape[1]//NUM_CELLS)
        for i in range(NUM_CELLS):
            for j in range(NUM_CELLS):
                axis.text((j + 1)*frontalized_cell_size[0] - int(0.3*frontalized_cell_size[0]),
                          i*frontalized_cell_size[1] + int(0.3*frontalized_cell_size[1]),
                          str(sudoku_matrix[i, j]), fontsize=digit_fontsize, c='r')


def show_solved_sudoku(frontalized_image, sudoku_matrix, sudoku_matrix_solved, digit_fontsize=20):
    show_image(frontalized_image, as_gray=True)

    frontalized_cell_size = (frontalized_image.shape[0]//NUM_CELLS, frontalized_image.shape[1]//NUM_CELLS)
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):
            if sudoku_matrix[i, j] == 0:
                plt.text(j*frontalized_cell_size[0] + int(0.3*frontalized_cell_size[0]),
                         (i + 1)*frontalized_cell_size[1] - int(0.3*frontalized_cell_size[1]),
                         str(sudoku_matrix_solved[i, j]), fontsize=digit_fontsize, c='g')