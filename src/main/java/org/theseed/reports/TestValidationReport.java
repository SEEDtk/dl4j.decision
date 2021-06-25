/**
 *
 */
package org.theseed.reports;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.io.LineReader;

/**
 * This is a base class for computing testing error during a validation report.  If an ID column is available,
 * the error is computed only on the testing set; otherwise, it is computed on the whole set.
 *
 * @author Bruce Parrello
 */
public abstract class TestValidationReport implements IValidationReport {

    // FIELDS
    /** saved ID column index */
    private int idColIdx;
    /** set of training set row IDs */
    private Set<String> trained;

    /**
     * Construct the validation report.  The default case it to say no training items
     * are in the input.
     */
    public TestValidationReport() {
        this.trained = Collections.emptySet();
        this.idColIdx = -1;
    }

    /**
     * @return TRUE if a record is NOT in the training set (and should be recorded), else FALSE
     *
     * @param metaData	tab-delimited string of metadata column values
     */
    private boolean isTestingRecord(String metaData) {
        boolean retVal = true;
        if (this.idColIdx >= 0) {
            String[] metaItems = StringUtils.split(metaData,'\t');
            retVal = ! this.trained.contains(metaItems[this.idColIdx]);
        }
        return retVal;
    }

    /**
     * Process the output for a batch.
     *
     * @param metaData	metadata strings for each row (already tab-delimited)
     * @param expected	expected values for the predictions, one column per label, in the same order as the metadata
     * @param output	output from the predictions, one column per label, in the same order as the metadata
     */
    public void reportOutput(List<String> metaData, INDArray expected, INDArray output) {
        // Loop through the data, asking the subclass to process each testing row.
        for (int r = 0; r < metaData.size(); r++) {
            if (this.isTestingRecord(metaData.get(r))) {
                this.processRecord(r, expected, output);
            }
        }
    }

    /**
     * Record the error for a single row of output.
     *
     * @param r				row of interest
     * @param expected		matrix of expected output
     * @param output		matrix of actual output
     */
    protected abstract void processRecord(int r, INDArray expected, INDArray output);

    /**
     * @return the computed mean absolute error (always between 0 and 1)
     */
    public abstract double getError();

    /**
     * Close the underlying file.  This is a no-op, since there is no file for this report.
     */
    public void close() { }

    /**
     * Set up to track which input rows were included in the model's original training set.
     *
     * @param modelDir	model directory
     * @param idCol		ID column name
     * @param metaList	list of metadata column names
     * @param trainList if specified, a list containing the training set
     *
     * @throws IOException
     */
   public void setupIdCol(File modelDir, String idCol, List<String> metaList, Collection<String> trainList) throws IOException {
        this.idColIdx = metaList.indexOf(idCol);
        if (this.idColIdx < 0)
            throw new IllegalArgumentException("ID column \"" + idCol + "\" not found in metadata list.");
        if (trainList != null)
            this.trained = new HashSet<String>(trainList);
        else {
            File trainedFile = new File(modelDir, "trained.tbl");
            this.trained = LineReader.readSet(trainedFile);
        }

    }

}
