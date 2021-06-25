/**
 *
 */
package org.theseed.reports;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.dl4j.train.IPredictError;

/**
 * This report computes the testing error for a regression model during validation.  The average absolute error
 * is computed for each label, and it is then scaled to the label's value range.  The mean of the scaled errors
 * is then computed.
 *
 * @author Bruce Parrello
 *
 */
public class RegressionTestValidationReport extends TestValidationReport {

    // FIELDS
    /** maximum expected value for each label */
    private double[] max;
    /** minimum expected value for each label */
    private double[] min;
    /** sum of absolute errors */
    private double[] sum;
    /** number of records processed */
    private int count;
    /** computed error */
    private double error;

    @Override
    public void startReport(List<String> metaCols, List<String> labels) {
        // Initialize the field values for each label.
        int w = labels.size();
        this.max = new double[w];
        Arrays.fill(this.max,  Double.NEGATIVE_INFINITY);
        this.min = new double[w];
        Arrays.fill(this.min, Double.POSITIVE_INFINITY);
        this.sum = new double[w];
        Arrays.fill(this.sum, 0.0);
        this.count = 0;
    }

    @Override
    public void finishReport(IPredictError errors) {
        // Here we compute the final error.
        if (this.count == 0)
            this.error = 0.0;
        else {
            // Here there is data.  We need to accumulate the scaled mean absolute error for each label.  The only
            // tricky part is when all of the expected values are the same, something that will almost never occur.
            // The accumulated error goes in here.
            double errorSum = 0.0;
            for (int i = 0; i < this.sum.length; i++) {
                double rawMeanError = this.sum[i] / this.count;
                if (this.max[i] != this.min[i])
                    rawMeanError /= (this.max[i] - this.min[i]);
                else if (this.max[i] != 0)
                    rawMeanError /= Math.abs(this.max[i]);
                errorSum += rawMeanError;
            }
            this.error = errorSum / this.sum.length;
        }
    }

    @Override
    protected void processRecord(int r, INDArray expected, INDArray output) {
        // Loop through the label columns.
        for (int i = 0; i < this.sum.length; i++) {
            // Get the expected value.  Update min and max.
            double e = expected.getDouble(r, i);
            if (e > this.max[i]) this.max[i] = e;
            if (e < this.min[i]) this.min[i] = e;
            // Get the output value and compute the absolute error.
            double o = output.getDouble(r, i);
            this.sum[i] += Math.abs(e - o);
        }
        this.count++;
    }

    @Override
    public double getError() {
        return this.error;
    }

}
