/**
 *
 */
package org.theseed.reports;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.dl4j.train.IPredictError;

/**
 * This is the interface for classes producing a validation report.  It is used by the testPredictions method of the TrainingProcessor
 * to determine the type of output.
 *
 * @author Bruce Parrello
 *
 */
public interface IValidationReport extends AutoCloseable {

    /**
     * Initialize the report.
     *
     * @param metaCols	list of metadata columns
     * @param labels	list of available labels
     */
    void startReport(List<String> metaCols, List<String> labels);

    /**
     * Write the output for a batch.
     *
     * @param metaData	metadata strings for each row (already tab-delimited)
     * @param expected	expected values for the predictions, one column per label, in the same order as the metadata
     * @param output	output from the predictions, one column per label, in the same order as the metadata
     */
    void reportOutput(List<String> metaData, INDArray expected, INDArray output);

    /**
     * Finish the report.
     *
     * @param predictor		error prediction object
     */
    void finishReport(IPredictError errors);

    /**
     * Close the underlying file.
     */
    public void close();

    /**
     * Set up to track which input rows were included in the model's original training set.
     *
     * @param modelDir	model directory
     * @param idCol		ID column name
     * @param metaList	list of metadata column names
     * @param trainList	if specified, a collection containing the training data
     *
     * @throws IOException
     */
    void setupIdCol(File modelDir, String idCol, List<String> metaList, Collection<String> trainList) throws IOException;

}
