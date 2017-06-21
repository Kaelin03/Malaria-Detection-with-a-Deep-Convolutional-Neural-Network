class Cell(object):

    def __init__(self, position, radius):
        self._position = position
        self._radius = radius
        self._status = None

    def get_position(self):
        # Returns the position of the cell
        return self._position

    def get_radius(self):
        # Returns the radius of the cell
        return self._radius

    def get_status(self):
        # Returns the status of the cell
        return self._status
