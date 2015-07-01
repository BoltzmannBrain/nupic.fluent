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

import cPickle as pkl
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
               dataFile,
               resultsDir,
               experimentName,
               load,
               modelName,
               modelModuleName,
               multiclass,
               trainSize,
               verbosity):
    """
    @param dataFile         (str)     Raw data file for the experiment.
    @param resultsDir       (str)     Directory where for the results metrics.
    @param experimentName   (str)     Experiment name, used for saving results.
    @param load             (bool)    True if a serialized model is to be
                                      loaded.
    @param modelName        (str)     Name of nupic.fluent Model subclass.
    @param modeModuleName   (str)     Model module -- location of the subclass.
    @param trainSize        (str)     Number of samples to use in training.
    @param verbosity        (int)     Greater value prints out more progress.

    """
    self.dataFile = dataFile
    self.resultsDir = resultsDir
    self.experimentName = experimentName
    self.load = load
    self.modelName = modelName
    self.modelModuleName = modelModuleName
    self.multiclass = multiclass
    self.trainSize = trainSize
    self.verbosity = verbosity

    self.modelPath = os.path.join(
      self.resultsDir, self.experimentName, self.modelName)
    if not os.path.exists(self.modelPath):
      os.makedirs(self.modelPath)


  def setupData(self, preprocess=False, sampleIdx=2, labelIdx=[3]):
    """
    Get the data from CSV and preprocess if specified.
    One index in labelIdx implies the model will train on a single
    classification per sample.
    """
    if self.model.multiclass and len(labelIdx) < 2:
      raise ValueError("Multiclass model requires more than one CSV column of "
                       "classifications.")

    samples, labels = readCSV(self.dataFile, sampleIdx, labelIdx)
    texter = TextPreprocess()
    
    if (not isinstance(self.trainSize, list) or
      self.trainSize[0] < 0 or
      self.trainSize[0] > len(samples)):
      raise ValueError("Invalid size(s) for training set.")
    
    self.labelRefs = list(set(labels))
    self.labels = numpy.array(
      [self.labelRefs.index(l) for l in labels], dtype="int8")
    if preprocess:
      self.samples = [texter.tokenize(sample,
                                      ignoreCommon=100,
                                      removeStrings=["[identifier deleted]"],
                                      correctSpell=True)
                      for sample in samples]
    else:
      self.samples = [texter.tokenize(sample) for sample in samples]

    if self.verbosity > 1:
      for i, s in enumerate(self.samples):
        print i, s, self.labelRefs[self.labels[i]]


  def initModel(self):
    """Load or instantiate the classification model."""
    if self.load:
      with open(
        os.path.join(modelPath, "model.pkl"), "rb") as f:
        self.model = pkl.load(f)
      print "Model loaded from \'{0}\'.".format(modelPath)
    else:
      try:
        module = __import__(self.modelModuleName, {}, {}, self.modelName)
        modelClass = getattr(module, self.modelName)
        self.model = modelClass(verbosity=self.verbosity)
      except ImportError:
        raise RuntimeError("Could not find model class \'{0}\' to import.".
                           format(self.modelName))


  def encodeSamples(self):
    """Encode the text samples into bitmap patterns, and log to txt file."""
    self.patterns = [self.model.encodePattern(s) for s in self.samples]
    self.model.logEncodings(self.patterns, self.modelPath)


  def runExperiment(self):
    """Train and test the model for each trial specified by self.trainSize."""
    self.testIndices = []
    self.results = []
    for i, size in enumerate(self.trainSize):
      partitions = self.partitionIndices(len(self.samples), size)
      if self.verbosity > 0:
        print ("\tRunner randomly selects to train on sample(s) {0}, and test "
               "on sample(s) {1}.".format(partitions[0], partitions[1]))

      self.model.resetModel()
      print "\tTraining for run {0} of {1}.".format(i+1, len(self.trainSize))
      self.training(partitions[0])
      print "\tTesting for this run."
      self.testing(partitions[1])

      # Save the test indices for printing the results evaluation.
      self.testIndices.append(partitions[1])


  def training(self, idx):
    for i in idx:
      self.model.trainModel(self.patterns[i], self.labels[i])


  def testing(self, idx):
    results = ([], [])
    for i in idx:
      predicted = self.model.testModel(self.patterns[i])
      results[0].append(predicted)
      results[1].append(self.labels[i])

    self.results.append(results)
    

  def calculateResults(self):
    """Calculate evaluation metrics from the result classifications."""

    resultCalcs = [self.model.evaluateResults(
      self.results[i], self.labelRefs, self.testIndices[i])
      for i in xrange(len(self.trainSize))]

    self.model.printFinalReport(self.trainSize, [r[0] for r in resultCalcs])

    # In case there are multiple trials of the same size
    # trialSize -> (category -> list of accuracies)
    trialAccuracies = defaultdict(lambda : defaultdict(lambda: numpy.ndarray(0)))
    for i, size in enumerate(self.trainSize):
      accuracies = self.model.calculateClassificationResults(self.results[i])
      for label, acc in accuracies:
        acc_list = trialAccuracies[size][label]
        trialAccuracies[size][label] = numpy.append(acc_list, acc)

    # Need the accuracies to be ordered for the graph
    trials = sorted(set(self.trainSize))
    # category -> list of list of accuracies
    classificationAccuracies = defaultdict(list)
    for trial in trials:
      accuracies = trialAccuracies[trial]
      for label, acc in accuracies.iteritems():
        classificationAccuracies[label].append(acc)

    plotter = PlotNLP()
    plotter.plotCategoryAccuracies(trialAccuracies, self.trainSize)
    plotter.plotCummulativeAccuracies(classificationAccuracies, self.trainSize)


  def save(self):
    """Save the serialized model."""
    print "Saving model to \'{0}\' directory.".format(self.modelPath)
    with open(os.path.join(self.modelPath, "model.pkl"), "wb") as f:
      pkl.dump(self.model, f)


  @staticmethod
  def partitionIndices(length, split):  ## TODO: use StandardSplit in data_split.py
    """Return two lists of indices; randomly sampled, not repeated."""
    trainIdx = random.sample(xrange(length), split)
    testIdx = [i for i in xrange(length) if i not in trainIdx]
    return (trainIdx, testIdx)
