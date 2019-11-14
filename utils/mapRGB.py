def mapRGB(iterator):
    """
    This function allows to iterate through a list of strings that represent standard rgb values.

    Inputs:     iterator(int): Integer to choose from list of RGB values
    Outputs:    String with RGB values
    """
    colourList = [
        'rgb(255, 0, 0)',  # red
        'rgb(0, 255, 0)',  # lime
        'rgb(0, 0, 255)',  # blue
        'rgb(0, 100, 0)',  # dark green
        'rgb(0, 255, 255)',  # cyan
        'rgb(255, 0, 255)',  # magenta
        'rgb(128, 128, 128)',  # gray
        'rgb(128, 0, 0)',  # maroon
        'rgb(128, 128, 0)',  # olive
        'rgb(160, 32, 240)',  #no name - purplish
        'rgb(128, 0, 128)',  # purple
        'rgb(0, 128, 128)',  # teal
        'rgb(255, 127, 80)',  # coral
        'rgb(65, 105, 225)',  # royal blue
        'rgb(255, 140, 0)',  # dark orange
        'rgb(220, 20, 60)',  # crimson
        'rgb(0, 250, 154)',  # medium spring green
        'rgb(139, 0, 139)',  # dark magenta
        'rgb(255, 105, 180)',  # pink
        'rgb(255, 255, 0)',  # yellow
    ]

    'if iterator bigger than list length begin from new'
    return colourList[iterator % len(colourList)]