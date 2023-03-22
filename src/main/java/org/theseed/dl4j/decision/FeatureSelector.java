/**
 *
 */
package org.theseed.dl4j.decision;

import java.util.Random;

/**
 * This object determines the list of features and the split point finder that should be used at a decision tree
 * node.  The constructor of the subclass should call "setup" with the appropriate values.  It can also override
 * the two retrieval methods for more complicated situations.  This class is constructed once for each node by
 * the FeatureSelectorFactory class.
 *
 * @author Bruce Parrello
 *
 */
public abstract class FeatureSelector {

    // FIELDS
    /** array of feature indexes from which to choose */
    private int[] idxes;
    /** split point finder to use */
    private SplitPointFinder finder;

    /**
     * Construct a feature selector.
     *
     * @param idxes		array of features from which to select
     * @param finder	split point finder to use on each feature
     */
    protected void setup(int[] idxes, SplitPointFinder finder) {
        this.idxes = idxes;
        this.finder = finder;
    }

    /**
     * @return the split point finder to use for features at the specified depth
     *
     * @param depth		depth (0 = root) of the target node
     */
    public SplitPointFinder getFinder() {
        return this.finder;
    }

    /**
     * @return the features to select from at the specified depth
     *
     * @param depth		depth (0 = root) of the target node
     */
    public int[] getFeaturesToUse() {
        return this.idxes;
    }

    /**
     * This feature selector returns a single feature that is processed using the sequential split point finder.
     */
    public static class Single extends FeatureSelector {

        /**
         * Construct a single-feature selector.
         *
         * @param idx	index of the feature to select
         */
        public Single(int idx) {
            this.setup(new int[] { idx }, new SequentialSplitPointFinder());
        }

    }

    /**
     * This feature selector returns an array of features to select from using the mean split point finder.
     */
    public static class Multiple extends FeatureSelector {

        /**
         * Construct a multiple-feature selector.
         *
         * @param idxes			array of input features to select from
         * @param nSelect		nubmer of features to select from the available list
         * @param randomizer	random-number generator to use
         */
        public Multiple(int[] idxes, int nSelect, Random randomizer) {
            // Get all the possible feature indices in a modifiable array.
            int[] range = idxes.clone();
            // Insure our selection size is in range.
            if (nSelect > idxes.length) nSelect = idxes.length - 1;
            // Form the output array.
            int[] choices = new int[nSelect];
            // Loop through the range array, picking random items.
            for (int i = 0; i < nSelect; i++) {
                int rand = randomizer.nextInt(range.length - i);
                choices[i] = range[rand + i];
                range[rand + i] = range[i];
            }
            this.setup(choices, new SplitPointFinder.Mean());
        }

    }
}
