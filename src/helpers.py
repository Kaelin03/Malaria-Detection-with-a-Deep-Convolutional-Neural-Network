def crop_to_circle(image, circle):
    height, width, _ = image.shape                          # Get the height and width of the original image
    x1 = int(max(circle[0] - circle[2], 0))          # Calculate the bounding box coordinates
    y1 = int(max(circle[1] - circle[2], 0))
    x2 = int(min(circle[0] + circle[2], width))
    y2 = int(min(circle[1] + circle[2], height))
    return image[y1:y2, x1:x2, :]                           # Return a cropped image
