/**
 *
 */
package org.theseed.dl4j.decision;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.theseed.dl4j.train.RandomForestTrainProcessor;

/**
 * This factory assigns a specific feature to each tree for the root node, and uses random selections
 * for the other nodes.
 *
 * @author Bruce Parrello
 *
 */
public class RootedTreeFeatureSelectorFactory implements Iterator<TreeFeatureSelectorFactory> {

    public class Builder extends TreeFeatureSelectorFactory {

        private int rootIdx;

        /**
         * Construct a feature-selector builder.
         *
         * @param randSeed	randomizer seed
         * @param rootIdx	index of the root feature column
         */
        public Builder(long randSeed, int rootIdx) {
            super(randSeed);
            this.rootIdx = rootIdx;
        }

        @Override
        public FeatureSelector getSelector(int depth) {
            FeatureSelector retVal;
            if (depth == 0)
                retVal = new FeatureSelector.Single(rootIdx);
            else
                retVal = new FeatureSelector.Multiple(RootedTreeFeatureSelectorFactory.this.nCols,
                        RootedTreeFeatureSelectorFactory.this.nSelect, this.getRandomizer());
            return retVal;
        }

    }

    // FIELDS
    /** array of features to use as roots */
    private int[] roots;
    /** number of trees built */
    private int counter;
    /** random seed generator */
    private Random randomizer;
    /** number of input columns */
    private int nCols;
    /** number of trees to produce */
    private int nTrees;
    /** number of features to use at each level */
    private int nSelect;

    /**
     * Create this feature selector factory.
     *
     * @param seed			randomizer seed
     * @param featureNames	list of feature column names
     * @param impactCols	list of the names of the feature columns to use as roots
     * @param numFeatures	number of features to select for each choice node
     * @param numTrees		number of trees to output
     */
    public RootedTreeFeatureSelectorFactory(long seed, List<String> featureNames, List<String> impactCols, int numFeatures,
            int numTrees) {
        this.randomizer = new Random(seed);
        this.nCols = featureNames.size();
        this.nTrees = numTrees;
        this.nSelect = numFeatures;
        this.roots = impactCols.stream().mapToInt(x -> featureNames.indexOf(x)).filter(i -> i >= 0).toArray();
        this.counter = 0;
    }

    /**
     * @return an iterator that produces a different rooted-tree builder for each output tree
     *
     * @param nTrees		number of trees to output
     * @param processor		controlling random-forest trainer
     *
     * @throws IOException
     */
    public static Iterator<TreeFeatureSelectorFactory> iterator(int nTrees, RandomForestTrainProcessor processor) throws IOException {
        RandomForest.Parms parms = processor.getParms();
        return new RootedTreeFeatureSelectorFactory(processor.getSeed() + 8719, processor.getColNames(),
                processor.getImpactCols(), parms.getNumFeatures(), nTrees);
    }

    @Override
    public boolean hasNext() {
        return this.counter < this.nTrees;
    }

    @Override
    public TreeFeatureSelectorFactory next() {
        counter++;
        int rootIdx = this.roots[counter % this.roots.length];
        return this.new Builder(this.randomizer.nextLong(), rootIdx);
    }

}
