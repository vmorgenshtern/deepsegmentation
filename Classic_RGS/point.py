class Point:
    """
    This class is used to store the properties of each point.
    """
    def __init__(self, coordinates=0, normal=0, curvature=0, neighbours=0, segment="NaN"):
        self.coordinates = coordinates
        self.normal = normal
        self.curvature = curvature
        self.neighbours = neighbours

        'Note: each point has a segment that it belongs to after segmentation. ' \
        'Difference to cluster: There are less clusters than segments, because clusters ' \
        'less than a minimum number of points are not counted'
        self.segment = segment

    def setCoordinates(self, coordinates):
        self.coordinates = coordinates

    def setNormal(self, normal):
        self.normal = normal

    def setCurvature(self, curvature):
        self.curvature = curvature

    def setNeighbours(self, neighbours):
        self.neighbours = neighbours

    def setSegment(self, segment):
        self.segment = segment

    def getCoordinates(self):
        return self.coordinates

    def getNormal(self):
        return self.normal

    def getCurvature(self):
        return self.curvature

    def getNeighbours(self):
        return self.neighbours

    def getSegment(self):
        return self.segment
