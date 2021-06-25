/**
 *
 */
package org.theseed.dl4j.train;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This interface describes an object that can be used to compute the prediction error for a
 * dataset.
 *
 * @author Bruce Parrello
 *
 */
public interface IPredictError {

    /**
     * Accumulate the error for a given set of predictions
     *
     * @param labels	expected results
     * @param output	predicted results
     */
    void accumulate(INDArray expect, INDArray output);

    /**
     * @return the prediction error
     */
    double getError();

    /**
     * Denote all of the data has been processed.
     */
    void finish();

    /**
     * Get titles for the auxiliary accuracy statistics.
     */
    String[] getTitles();

    /**
     * Get the values for the auxiliary accuracy statistics.
     */
    double[] getStats();

}
