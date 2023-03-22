/**
 *
 */
package org.theseed.dl4j.decision;

import java.util.Iterator;
import java.util.Random;

import org.theseed.dl4j.train.RandomForestTrainProcessor;

/**
 * This is a tree feature selector factory that returns a random selection of multiple features at each tree node.
 *
 * @author Bruce Parrello
 *
 */
public class NormalTreeFeatureSelectorFactory implements Iterator<TreeFeatureSelectorFactory> {

    /**
     * This is the factory class we return.  It outputs a multiple-feature selector for each node.
     */
    public class Builder extends TreeFeatureSelectorFactory {

        /**
         * Construct the builder.
         *
         * @param randSeed		randomizer seed to use
         */
        public Builder(long randSeed) {
            super(randSeed);
        }

        @Override
        public FeatureSelector getSelector(int depth) {
            return new FeatureSelector.Multiple(NormalTreeFeatureSelectorFactory.this.idxes,
                    NormalTreeFeatureSelectorFactory.this.numFeatures, this.getRandomizer());
        }
    }

    // FIELDS
    /** number of trees created so far */
    private int counter;
    /** random-number generator */
    private Random rand;
    /** number of features to select at each node */
    private int numFeatures;
    /** number of trees to output */
    private int nTrees;
    /** array of nontrivial input column indices */
    private int[] idxes;

    /**
     * Construct this selector factory.
     *
     * @param randSeed		randomizer seed
     * @param idxCols		array of nontrival input column indices
     * @param numSelect		number of features to select for each tree
     * @param numTrees		number of trees to build
     */
    public NormalTreeFeatureSelectorFactory(long randSeed, int[] idxCols, int numSelect, int numTrees) {
        this.counter = 0;
        this.rand = new Random(randSeed);
        this.numFeatures = numSelect;
        this.nTrees = numTrees;
        this.idxes = idxCols;
    }

    @Override
    public boolean hasNext() {
        return this.counter < this.nTrees;
    }

    @Override
    public TreeFeatureSelectorFactory next() {
        this.counter++;
        return this.new Builder(rand.nextLong());
    }

    /**
     * @return an iterator that will produce the feature selectors needed for the processor random forest
     *
     * @param nTrees		number of trees to produce
     * @param processor		random forest processor to process the trees
     */
    public static Iterator<TreeFeatureSelectorFactory> iterator(int nTrees, RandomForestTrainProcessor processor) {
        RandomForest.Parms parms = processor.getParms();
        int nFeatures = parms.getNumFeatures();
        // Get the useful-feature array.
        int[] idxes = processor.getUsefulFeatureArray();
        // Insure the number of features per tree is reasonable.
        if (nFeatures >= idxes.length)
            nFeatures = idxes.length / 2;
        // Note that we add a prime number to the seed so that it is not the same as the seed used for example selection.
        return new NormalTreeFeatureSelectorFactory(processor.getSeed() + 3719, idxes, nFeatures,
                parms.getNumTrees());
    }

}
