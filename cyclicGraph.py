__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import matplotlib.pyplot as plt
import random


class CyclicGraph(object):
    def __init__(self, v):
        self.v = v
        self.edges = [(v_i, v_i + 1) if v_i - 1 != v else (v_i, 1) for v_i in range(1, v)]
        self.vertices = (v_i for v_i in range(1, v + 1))

    def displayGraph(self):
        lastv2 = self.edges[-1][1]
        lastv2y = random.random()
        for v1, v2 in self.edges[:-1]:
            rv2y = random.random()

            tv1, tv2 = [v1, random.random()], [v2, rv2y]
            lv1, lv2 = [v1, rv2y], [lastv2, lastv2y]

            plt.plot(lv1, lv2)
            plt.plot(tv1, tv2)

            print lv1, lv2
            print tv1, tv2

            lastv2 = v2
            lastv2y = rv2y

        plt.show()


cg = CyclicGraph(5)

cg.displayGraph()
