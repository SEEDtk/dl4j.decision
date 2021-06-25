/**
 *
 */
package org.theseed.dl4j.decision;

import java.io.IOException;
import java.util.Iterator;
import java.util.Random;

import org.theseed.dl4j.train.RandomForestTrainProcessor;
import org.theseed.utils.IDescribable;

/**
 * Feature selector factories create the feature selectors for each level of a single decision tree.  Each factory
 * belongs to a single tree and returns a selector for each node.
 *
 * @author Bruce Parrello
 *
 */
public abstract class TreeFeatureSelectorFactory {

    public static enum Type implements IDescribable {
        NORMAL {
            @Override
            public Iterator<TreeFeatureSelectorFactory> create(int nTrees, RandomForestTrainProcessor processor) {
                return NormalTreeFeatureSelectorFactory.iterator(nTrees, processor);
            }

            @Override
            public String getDescription() {
                return "Select features randomly.";
            }
        }, ROOTED {
            @Override
            public Iterator<TreeFeatureSelectorFactory> create(int nTrees, RandomForestTrainProcessor processor) throws IOException {
                return RootedTreeFeatureSelectorFactory.iterator(nTrees, processor);
            }

            @Override
            public String getDescription() {
                return "Root each tree in a specified feature.";
            }
        };

        public abstract Iterator<TreeFeatureSelectorFactory> create(int nTrees, RandomForestTrainProcessor processor)
                throws IOException;
    }


    // FIELDS
    /** random-number generator */
    private Random randomizer;

    /**
     * Construct a feature selector factory.
     *
     * @param randSeed	randomizer seed
     */
    public TreeFeatureSelectorFactory(long randSeed) {
        this.randomizer = new Random(randSeed);
    }

    /**
     * Compute the feature selector to be used for the current node.
     *
     * @param depth		depth of the node
     *
     * @return the feature selector to use
     */
    public abstract FeatureSelector getSelector(int depth);

    /**
     * @return the randomizer
     */
    public Random getRandomizer() {
        return this.randomizer;
    }


}
