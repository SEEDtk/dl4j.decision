/**
 *
 */
package org.theseed.dl4j.train;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This computes the prediction error for a classification model.  The error is the number of cases where the
 * best class is incorrectly predicted divided by the total number of cases.
 *
 * @author Bruce Parrello
 *
 */
public class ClassPredictError implements IPredictError {

    // FIELDS
    /** total number of predictions */
    private int rows;
    /** number of true positives */
    private int truePositive;
    /** number of true negatives */
    private int trueNegative;
    /** number of false negatives */
    private int falseNegative;
    /** number of wrong positives */
    private int wrongPositive;
    /** accuracy */
    private double accuracy;
    /** precision */
    private double sensitivity;
    /** sensitivity */
    private double specificity;
    /** fuzziness */
    private double fuzziness;

    public ClassPredictError(List<String> labels) {
        this.rows = 0;
        this.truePositive = 0;
        this.trueNegative = 0;
        this.falseNegative = 0;
        this.wrongPositive = 0;
    }

    @Override
    public void accumulate(INDArray expect, INDArray output) {
        // Loop through the data rows.
        for (int r = 0; r < expect.rows(); r++) {
            int expected = computeBest(expect, r);
            int predicted = computeBest(output, r);
            if (expected != predicted) {
                if (expected == 0)
                    this.falseNegative++;
                else if (predicted > 0)
                    this.wrongPositive++;
            } else if (expected == 0)
                this.trueNegative++;
            else
                this.truePositive++;
            this.rows++;
        }
    }

    /**
     * Compute the index of the best label in the specified row.
     *
     * @param data	array containing data rows, one column per label
     * @param r		row to examine
     *
     * @return the index of the highest value in the row
     */
    public static int computeBest(INDArray data, int r) {
        double best = data.getDouble(r, 0);
        int retVal = 0;
        for (int i = 1; i < data.columns(); i++) {
            double val = data.getDouble(r, i);
            if (val > best) {
                best = val;
                retVal = i;
            }
        }
        return retVal;
    }

    @Override
    public double getError() {
        return (1.0 - accuracy);
    }

    @Override
    public void finish() {
        this.accuracy = 0.0;
        this.sensitivity = 0.0;
        int good = this.trueNegative + this.truePositive;
        if (good > 0) {
            this.accuracy = good / (double) this.rows;
            // Sensitivity is the  number of correct positives over the total number of expected positives.  It
            // indicates how good we are at correctly guessing positives.
            this.sensitivity = good / (double) (this.truePositive + this.falseNegative + this.wrongPositive);
            // Specificity is the  number of negatives over the total number of expected negatives.  It indicates
            // how good we are at correctly guessing negatives.
            this.specificity = this.trueNegative / (double) (this.falseNegative + this.trueNegative);
            // Fuzziness is the number of wrong positives over the total number of positives that were guessed
            // positive. It indicates how often we guess the wrong condition.
            this.fuzziness = this.wrongPositive / (double) (this.wrongPositive + this.truePositive);
        }

    }

    @Override
    public String[] getTitles() {
        return new String[] { "Accuracy", "Sensitivity", "Specificity", "Fuzziness" };
    }

    @Override
    public double[] getStats() {
        return new double[] { this.accuracy, this.sensitivity, this.specificity, this.fuzziness };
    }

}
