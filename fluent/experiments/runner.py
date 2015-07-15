# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import collections
import cPickle as pkl
import itertools
import numpy
import os
import random

from collections import defaultdict
from fluent.utils.csv_helper import readCSV
from fluent.utils.plotting import PlotNLP

from fluent.utils.text_preprocess import TextPreprocess



class Runner(object):
  """
  Class to run the baseline NLP experiments with the specified data, models,
  text processing, and evaluation metrics.
  """

  def __init__(self,
               dataPath,
               resultsDir,
               experimentName,
               load,
               modelName,
               modelModuleName,
               numClasses,
               plots,
               orderedSplit,
               trainSize=None,
               verbosity=1):
    """
    @param dataPath         (str)     Path to raw data file for the experiment.
    @param resultsDir       (str)     Directory where for the results metrics.
    @param experimentName   (str)     Experiment name, used for saving results.
    @param load             (bool)    True if a serialized model is to be
                                      loaded.
    @param modelName        (str)     Name of nupic.fluent Model subclass.
    @param modeModuleName   (str)     Model module -- location of the subclass.
    @param numClasses       (int)     Number of classes (labels) per sample.
    @param plots            (int)     Specifies plotting of evaluation metrics.
    @param orderedSplit     (bool)    Indicates method for splitting train/test
                                      samples; False is random, True is ordered.
    @param trainSize        (str)     Number of samples to use in training.
    @param verbosity        (int)     Greater value prints out more progress.

    """
    self.dataPath = dataPath
    self.resultsDir = resultsDir
    self.experimentName = experimentName
    self.load = load
    self.modelName = modelName
    self.modelModuleName = modelModuleName
    self.numClasses = numClasses
    self.plots = plots
    self.orderedSplit = orderedSplit
    self.trainSize = trainSize
    self.verbosity = verbosity

    self.modelPath = os.path.join(
      self.resultsDir, self.experimentName, self.modelName)
    if not os.path.exists(self.modelPath):
      os.makedirs(self.modelPath)

    if self.plots:
      self.plotter = PlotNLP()

    # self.dataDict is a defaultdict of OrderedDicts. Keys of the former are CSV
    # names, and keys-values of the latter are samples-classifications.
    self.dataDict = None
    self.labelRefs = None
    self.patterns = defaultdict(list)
    self.partitions = defaultdict(list)
    self.results = defaultdict(list)


  def _calculateTrialAccuracies(self, evaluationDict):
    """
    @param evaluationDict              ()

    @return trialAccuracies     (defaultdict)   Items are defaultdicts, one for
        each size of the training set. Inner defaultdicts keys are
        classification categories, with numpy array values that contain one
        accuracy value for each trial.
    """
    # To handle multiple trials of the same size:
    # trialSize -> (category -> list of accuracies)
    trialAccuracies = defaultdict(lambda: defaultdict(lambda:
        numpy.ndarray(0)))
    for trial, evals in evaluationDict.iteritems():
      for trainSize, acc in enumerate(evals):
        for datafile in acc.keys():
          accList = trialAccuracies[trainSize][datafile]
          trialAccuracies[trainSize][datafile] = numpy.append(accList, acc[datafile][0])

    return trialAccuracies


  def _calculateClassificationAccuracies(self, trialAccuracies):
    """
    @param trialAccuracies            (defaultdict)   Please see the description
        in self._calculateTrialAccuracies().

    @return classificationAccuracies  (defaultdict)   Keys are classification
        categories, with multiple numpy arrays as values -- one for each size of
        training sets, with one accuracy value for each run of that training set
        size.
    """
    # category -> list of list of accuracies
    classificationAccuracies = defaultdict(list)
    for _, accuracies in trialAccuracies.iteritems():
      # Iterate through each training size, in order [0, 1, 2, ...] as
      # in trialAccuracies.keys()
      for label, acc in accuracies.iteritems():
        classificationAccuracies[label].append(acc)

    return classificationAccuracies


  @staticmethod
  def getListSet(iterable):
    """Return a list of the set of items in the input iterable."""
    return list(set(itertools.chain.from_iterable(iterable)))


  def _mapLabelRefs(self, dataDict):
    """
    Replace the label strings in dataDict with corresponding ints; operates on
    dataDict values in place.
    """
    for samples, labels in dataDict.iteritems():
      dataDict[samples] = numpy.array(
          [self.labelRefs.index(label) for label in labels])


  def _preprocess(self, dataDict, preprocess):
    """
    Tokenize the samples, with or without preprocessing; operates on dataDict
    values in place.
    """
    texter = TextPreprocess()
    if preprocess:
      samples = [(texter.tokenize(sample,
                                  ignoreCommon=100,
                                  removeStrings=["[identifier deleted]"],
                                  correctSpell=True),
                  labels) for sample, labels in dataDict.iteritems()]
    else:
      samples = [(texter.tokenize(sample), labels)
                 for sample, labels in dataDict.iteritems()]

    if self.verbosity > 1:
      for i, s in enumerate(samples): print i, s

    return samples


  def setupData(self, preprocess=False):
    """
    Generate list for label references, map the labels to these indices, and
    tokenize the text samples.

    @param preprocess     (bool)              To preprocess the text or not.
    """
    labels = []
    for filename, data in self.dataDict.iteritems():
      labels.append(self.getListSet(data.values()))
    self.labelRefs = self.getListSet(labels)

    # Map label strings to indices, preprocess the text
    for filename, data in self.dataDict.iteritems():
      self._mapLabelRefs(data)
      self.dataDict[filename] = self._preprocess(data, preprocess)

    if self.plots:
      # Rename the keys from filenames to category names.
      for filename in self.dataDict.keys():
        self.dataDict[filename.split(".")[-2]] = self.dataDict.pop(filename)


  def initModel(self):
    """Load or instantiate the classification model."""
    if self.load:
      with open(os.path.join(self.modelPath, "model.pkl"), "rb") as f:
        self.model = pkl.load(f)
      print "Model loaded from \'{0}\'.".format(self.modelPath)
    else:
      try:
        module = __import__(self.modelModuleName, {}, {}, self.modelName)
        modelClass = getattr(module, self.modelName)
        self.model = modelClass(verbosity=self.verbosity)
      except ImportError:
        raise RuntimeError("Could not find model class \'{0}\' to import.".
                           format(self.modelName))


  def encodeSamples(self):
    """
    Encode the text samples into bitmap patterns, and log to txt file. The
    encoded patterns are stored in a dict along with their corresponding class
    labels.

    TODO: alternatively, don't have patterns dict, use self.dataDict[filename]
    """
    for filename, data in self.dataDict.iteritems():

      self.patterns[filename] = [{"pattern": self.model.encodePattern(sample[0]),
                                  "labels": sample[1]}
                                 for sample in data]

    self.model.logEncodings(self.patterns, self.modelPath)


  def runExperiment(self, trainSize):
    """
    Train and test the model for each trial specified by trainSize; the model
    is reset each run.

    The training indices will be chosen at random for each trial, unless the
    member variable orderedSplit is set to True.

    TODO: broken
    """
    for i, size in enumerate(trainSize):
      self.partitions.append(self.partitionIndices(size))

      if self.verbosity > 0:
        print ("\tRunner selects to train on sample(s) {0}, and test on "
               "samples(s) {1}.".format(self.partitions[i][0],
                                        self.partitions[i][1]))

      self.model.resetModel()
      print "\tTraining for run {0} of {1}.".format(i+1, len(trainSize))
      self.training(partitions[0])
      print "\tTesting for this run."
      self.testing(partitions[1])


  def runTrial(self, trainSize):
    """
    Train and test the model for one trial specified by trainSize; the model
    is reset each run.

    @return partitionsDict  (defaultdict)
    @return results         (defaultdict)       Keys are datafile names, values
        are two-tuples, where the elements are lists of the predicted and actual
        classifications; these items are numpy arrays.

    TODO: move the returned dicts to member variables; see commented out lines.
    """
    self.model.resetModel()
    partitionsDict = defaultdict(list)
    for filename, data in self.patterns.iteritems():
      # Determine samples indices for which to train on.
      length = len(data)
      split = trainSize if trainSize < length else length
      partitions = self.partitionIndices(split, length)
      # self.partitions[filename].append(partitions)
      partitionsDict[filename] = (partitions)

      print ("\tRunning trial for the {0} data, with training size {1}.".
             format(filename, split))
      if self.verbosity > 0:
        print ("\tRunner selects to train on sample(s) {0}, and test "
               "on sample(s) {1}.".
               format(partitions[0], partitions[1]))

      self.training(data, partitions[0])

    results = defaultdict(list)
    for filename, data in self.patterns.iteritems():
      # if partitionsDict[filename][-1:][0][1]:
      if partitionsDict[filename][1]:
        results[filename] = self.testing(
            filename, data, partitionsDict[filename][1])
        # results[filename].append(self.testing(
        #     filename, data, partitionsDict[filename][-1:][0][1]))
    return partitionsDict, results


  def training(self, data, indices):
    """
    Train the model one-by-one on each pattern specified in this trial's
    partition of indices.
    """
    for i in indices:
      self.model.trainModel(data[i]["pattern"], data[i]["labels"])


  def testing(self, filename, data, indices):
    results = ([], [])
    for i in indices:
      predicted = self.model.testModel(data[i]["pattern"])
      results[0].append(predicted)
      results[1].append(data[i]["labels"])
    return results
    # self.results[filename].append(results)


  def calculateResults(self, partitions, results):
    """
    Calculate evaluation metrics from the result classifications.

    @param partitions    (defaultdict)     Keys are filenames, values are
        two-tuples: ([indices trained on], [indices tested on]).

    @param results       (defaultdict)     Keys are filenames, values are
        two-tuples: ([predicted classes], [actual classes]).

    @return resultCalcs  (defaultdict)     Keys are filenames, values are
        two-tuples: (accuracy, confusion matrix numpy array).

    TODO: pass intended CM results to plotter.plotConfusionMatrix()
    """
    resultCalcs = defaultdict(list)
    for filename, res in results.iteritems():
      # res is a two-tuple of lists for predicted and actual classifications
      resultCalcs[filename] = self.model.evaluateResults(
          results[filename], self.labelRefs, partitions[filename][1])

      # self._printClassificationsReport(filename, resultCalcs[filename])

    return resultCalcs


  def calculateFinalResults(self, evaluationDict):
    """

    @param evaluationDict     (defaultdict)     Keys are trial numbers, and each
        value is a list of defaultdicts, one for each trainSize; inner dicts'
        keys are datafiles, and values are evaluation metrics
        (accuracy, confusion matrix) for that run.
    """
    trialAccuracies = self._calculateTrialAccuracies(evaluationDict)

    classificationAccuracies = self._calculateClassificationAccuracies(
          trialAccuracies)

    if self.plots: self._plot(trialAccuracies, classificationAccuracies)


  def save(self):
    """Save the serialized model."""
    print "Saving model to \'{0}\' directory.".format(self.modelPath)
    with open(os.path.join(self.modelPath, "model.pkl"), "wb") as f:
      pkl.dump(self.model, f)


  def getMaxSize(self):
    """Return the max length of the individual data dicts in self.dataDict."""
    maxLength = 0
    for _, data in self.dataDict.iteritems():
      length = len(data)
      if length > maxLength:
        maxLength = length
    return maxLength


  def partitionIndices(self, split, length):
    """
    Returns train and test indices.

    TODO: use StandardSplit in data_split.py
    """
    # length = len(self.samples)
    if self.orderedSplit:
      trainIdx = range(split)
      testIdx = range(split, length)
    else:
      # Randomly sampled, not repeated
      trainIdx = random.sample(xrange(length), split)
      testIdx = [i for i in xrange(length) if i not in trainIdx]

    return (trainIdx, testIdx)


  def validateExperiment(self, expectationFilePath):
    """Returns accuracy of predicted labels against expected labels."""
    dataDict = readCSV(expectationFilePath, 2, self.numClasses)

    accuracies = numpy.zeros((len(self.results)))
    for i, trial in enumerate(self.results):
      for j, predictionList in enumerate(trial[0]):
        predictions = [self.labelRefs[p] for p in predictionList if p]
        expected = dataDict.items()[j+self.trainSize[i]][1]
        accuracies[i] += (float(len(set(predictions) & set(expected)))
                          / len(expected))
      accuracies[i] = accuracies[i] / len(trial[0])

    return accuracies


  @staticmethod
  def _printClassificationsReport(name, results):
    """
    Prints result accuracies.

    @param name         (str)           Name representing this experiment/trial.
    @param results      (list)          Each item is a two-tuple: accuracy and
        confusion matrix; only the accuracy is used here. The items are expected
        to be in order where theire index is indicative of the training set
        size.
    """
    template = "{0:<20}|{1:<10}"
    print "Evaluation results for \'{0}\':".format(name)
    print template.format("Size of training set", "Accuracy")
    for i, r in enumerate(results):
      print template.format(i, r[0])


  def _plot(self, trialAccuracies, classificationAccuracies):
    """Plot evaluation metrics."""
    self.plotter.plotCumulativeAccuracies(
        classificationAccuracies, trialAccuracies.keys())

    # Plot only a few trialAccuracy plots per subplot
    subDict = defaultdict(list)
    for k, v in trialAccuracies.iteritems():
      subDict[k] = v
      if not k % 20:
        self.plotter.plotCategoryAccuracies(subDict, subDict.keys())
        subDict.clear()
    # Plot the remaining few
    if subDict:
      self.plotter.plotCategoryAccuracies(subDict, subDict.keys())

    if self.plots > 1:
      # Plot extra evaluation figures -- confusion matrix.
      self.plotter.plotConfusionMatrix(
          self.setupConfusionMatrices(resultCalcs))
