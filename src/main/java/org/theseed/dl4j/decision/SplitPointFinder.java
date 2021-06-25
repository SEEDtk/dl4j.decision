/**
 *
 */
package org.theseed.dl4j.decision;

import java.util.List;

import org.nd4j.linalg.dataset.DataSet;
import org.theseed.dl4j.train.RandomForestTrainProcessor;

/**
 * This object is used to determine the best split point for a feature in the decision tree.  The default
 * method simply takes the mean.  More sophisticated methods will look at multiple split points.
 *
 * @author Bruce Parrello
 *
 */
public abstract class SplitPointFinder {

    /**
     * Enumerator for split point strategies.
     */
    public static enum Type {
        MEAN {
            @Override
            public SplitPointFinder create(RandomForestTrainProcessor processor) {
                return new SplitPointFinder.Mean();
            }
        }, SEQUENTIAL {
            @Override
            public SplitPointFinder create(RandomForestTrainProcessor processor) {
                return new SequentialSplitPointFinder();
            }
        };

        public abstract SplitPointFinder create(RandomForestTrainProcessor processor);
    }

    /**
     * @return a candidate splitter for the specified input feature in a dataset
     *
     * @param i			index of the specified feature
     * @param nClasses	number of classifications
     * @param rows		list of dataset rows for the training set
     * @param entropy	current entropy level
     */
    public abstract Splitter computeSplit(int i, int nClasses, List<DataSet> rows, double entropy);

    /**
     * Only test the mean for the split point.
     */
    public static class Mean extends SplitPointFinder {

        @Override
        public Splitter computeSplit(int i, int nClasses, List<DataSet> rows, double entropy) {
            double mean = DecisionTree.featureMean(rows, i);
            return Splitter.computeSplitter(i, mean, nClasses, rows, entropy);
        }

    }


}
