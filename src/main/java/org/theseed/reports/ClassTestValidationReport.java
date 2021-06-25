/**
 *
 */
package org.theseed.reports;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.dl4j.train.ClassPredictError;
import org.theseed.dl4j.train.IPredictError;

/**
 * This report computes the testing error for a model during validation.  This is the fractional accuracy (0 = perfect, 1 = all
 * predictions wrong). There is nothing fancy here, just the raw number.
 *
 * @author Bruce Parrello
 *
 */
public class ClassTestValidationReport extends TestValidationReport {

    // FIELDS
    /** total number of rows */
    private int count;
    /** total number of error rows */
    private int errCount;
    /** computed testing error */
    private double error;

    @Override
    public void startReport(List<String> metaCols, List<String> labels) {
        this.count = 0;
        this.errCount = 0;
    }

    @Override
    public void finishReport(IPredictError errors) {
        if (this.count == 0)
            this.error = 0.0;
        else
            this.error = ((double) this.errCount) / this.count;
    }

    @Override
    protected void processRecord(int r, INDArray expected, INDArray output) {
        // This will be set to the expected index.
        int e = ClassPredictError.computeBest(expected, r);
        // This will be set to the index of the highest output value.
        int o = ClassPredictError.computeBest(output, r);
        // Count the result.
        if (e != o)
            this.errCount++;
        this.count++;
    }

    @Override
    public double getError() {
        return this.error;
    }

}
