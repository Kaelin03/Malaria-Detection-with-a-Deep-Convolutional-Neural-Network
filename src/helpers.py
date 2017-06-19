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
