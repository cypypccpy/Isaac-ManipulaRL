import numpy
import math

a = numpy.array([[math.sqrt(3) / 2, 0.5, 0, 0, 0, 0],
                [-0.5, math.sqrt(3) / 2, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, math.sqrt(3) / 2, -0.5, 0],
                [0, 0, 0, 0.5, math.sqrt(3) / 2, 0],
                [0, 0, 0, 0, 0, 1]], dtype='float32')

at = numpy.transpose(a)

b = numpy.array([[1, 0, 0, -1, 0, 0],
                [0, 12, 6, 0, -12, 6],
                [0, 6, 4, 0, -6, 2],
                [-1, 0, 0, 1, 0, 0],
                [0, -12, -6, 0, 12, 6],
                [0, 6, 2, 0, -6, 4]], dtype='float32')

print(at * b * a)