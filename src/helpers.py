def get_sample_id(line):
    # Given a string
    # Returns the sample name from the string
    # Returns False if no sample id is found
    try:
        line = line.split("/")[-1].split("_")[1]
    except IndexError:
        print("Warning: no sample name found in " + line + ".")
        line = False
    return line


def get_image_id(line):
    # Given a string
    # Returns the image name from the string
    # Returns False if no image id is found
    try:
        line = line.split("/")[-1].split("_")[2]
    except IndexError:
        print("Warning: no image name found in " + line + ".")
        line = False
    return line


def crop_to_circle(image, circle):
    height, width, _ = image.shape                          # Get the height and width of the original image
    x1 = int(max(circle[0] - circle[2], 0))          # Calculate the bounding box coordinates
    y1 = int(max(circle[1] - circle[2], 0))
    x2 = int(min(circle[0] + circle[2], width))
    y2 = int(min(circle[1] + circle[2], height))
    return image[y1:y2, x1:x2, :]                           # Return a cropped image


def get_sample_index(sample_list, sample_id):
    # Given a sample name
    # Returns the index of the sample if it exists
    # Returns -1 if the sample does not exist
    index = -1
    for i, sample in enumerate(sample_list):
        if sample_id == sample.get_id():
            index = i
            break
    return index
