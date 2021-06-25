/**
 *
 */
package org.theseed.dl4j.decision;

import org.nd4j.linalg.dataset.DataSet;

/**
 * This interface defines the functions required by a randomizer for choosing the training data in a random forest.
 * There is a preparation function and then a function that gets the individual data for each tree.  That last
 * function is executed in parallel, and must be coded accordingly.
 *
 * @author Bruce Parrello
 *
 */
public interface IRandomizer {

    /**
     * Initialize for building a random forest.
     *
     * @param nClasses		number of classifications
     * @param nSize			optimal size of output training sets
     * @param trainingSet	input training set
     */
    public void initializeData(int nClasses, int nSize, DataSet trainingSet);

    /**
     * @return the training set for a particular tree
     *
     * @param seed		seed for random-number generator
     */
    public DataSet getData(long seed);

}
