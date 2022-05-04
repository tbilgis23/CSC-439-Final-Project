'''
File: BootstrapResampling.py
Author: Tugay Bilgis
Purpose: A simple program to do bootstrap resampling and calculate the new p-value
'''

import random


class BootstrapResampling():
    """ This class  implements the non-parametric bootstrap resampling procedure discussed in class.
    """

    def getAverageBaselineScore(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with key
            'baselineScore' (float), calculate the average baselineScore
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'baselineScore':0.0, 'experimentalScore':1.0}, ... ]

            :param dataIn: List of dictionaries with key 'baselineScore'
            :return: Average 'baselineScore' across all elements in list.
        """
        total = 0
        for dic in dataIn:
            total += dic['baselineScore']
        total /= len(dataIn)
        return total

    def getAverageExperimentalScore(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with key
            'experimentalScore' (float), calculate the average baselineScore
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'experimentalScore':0.0, 'experimentalScore':1.0}, ... ]

            :param dataIn: List of dictionaries with key 'experimentalScore'
            :return: Average 'experimentalScore' across all elements in list.
        """
        total = 0
        for dic in dataIn:
            total += dic['experimentalScore']
        total /= len(dataIn)
        return total

    def createDifferenceScores(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with keys 'baselineScore'
            and 'experimentalScore', calculate their difference scores
            (experimentalScore - baselineScore).
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'experimentalScore':0.0, 'experimentalScore':1.0}, ... ]
            Example output: [1.0, ...]

            :param dataIn: List of dictionaries with float keys 'baselineScore', 'experimentalScore'
            :return: List of floats representing difference scores (experimental - baseline)
        """
        return [ dic['experimentalScore'] - dic['baselineScore'] for dic in dataIn ]

    def generateOneResample(self, differenceScores:list):
        """Randomly resamples the difference scores, to make a bootstrapped resample
            Example input: [0, 1, 0, 0, 1, 0, 1, 1, 0]
            Example output: [1, 0, 1, 0, 0, 1, 0, 1, 1]

            :param differenceScores: A list of difference scores (floats).
            :return: A list of randomly resampled difference scores (floats),
                of the same length as the input, populated using random
                sampling with replacement.
        """
        random_diff = []
        for _ in differenceScores:
            i = random.randint(0, len(differenceScores)-1)
            random_diff.append(differenceScores[i])
        return random_diff

    def calculatePValue(self, dataIn:list, numResamples=10000):
        """Calculate the p-value of a dataset using the bootstrap resampling procedure.
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'baselineScore':0.0, 'experimentalScore':1.0}, ... ]
            Example output: 0.01

            :param dataIn: List of dictionaries with float keys 'baselineScore', 'experimentalScore' populated
            :param numResamples: The number of bootstrap resamples to use (typically 10,000 or higher)
            :return: A value representing the p-value using the bootstrap resampling procedure (float)
        """
        total = 0
        diff_scores = self.createDifferenceScores(dataIn)
        for _ in range(numResamples):
            rand_diff = self.generateOneResample(diff_scores)
            if sum(rand_diff) <= 0:
                total += 1
        return total/numResamples


