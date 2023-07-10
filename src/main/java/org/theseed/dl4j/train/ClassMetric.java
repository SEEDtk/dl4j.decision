/**
 *
 */
package org.theseed.dl4j.train;

import org.nd4j.evaluation.classification.ConfusionMatrix;
import org.nd4j.evaluation.classification.Evaluation;

/**
 * This is a simple enum that computes a metric for a classifier evaluation.  Its purpose is to allow selection of the
 * metric to optimize during searches.
 *
 * @author Bruce Parrello
 *
 */
public enum ClassMetric {
    /** fraction of total results that are correct */
    ACCURACY {
        @Override
        public double compute(ConfusionMatrix<Integer> matrix) {
            return ratio(this.truePositive(matrix) + this.trueNegative(matrix),
                    this.falsePositive(matrix) + this.falseNegative(matrix));
        }
    },
    /** fraction of positive results that are correct */
    PRECISION {
        @Override
        public double compute(ConfusionMatrix<Integer> matrix) {
            return ratio(this.truePositive(matrix), this.falsePositive(matrix));
        }
    },
    /** fraction of negative results that are correct */
    NPV {
        @Override
        public double compute(ConfusionMatrix<Integer> matrix) {
            return ratio(this.trueNegative(matrix), this.falseNegative(matrix));
        }
    },
    /** fraction of actual negatives that are correct */
    SPECIFICITY {
        @Override
        public double compute(ConfusionMatrix<Integer> matrix) {
            return ratio(this.trueNegative(matrix), this.falsePositive(matrix));
        }
    },
    /** fraction of actual positives that are correct */
    SENSITIVITY {
        @Override
        public double compute(ConfusionMatrix<Integer> matrix) {
            return ratio(this.truePositive(matrix), this.falseNegative(matrix));
        }
    },
    /** positive likelihood ratio */
    PLR {
        @Override
        public double compute(ConfusionMatrix<Integer> matrix) {
            return negativeLikelihood(matrix);
        }
    },
    /** negative likelihood ratio */
    NLR {
        @Override
        public double compute(ConfusionMatrix<Integer> matrix) {
            return positiveLikelihood(matrix);
        }
    };

    /**
     * @return the value of this metric
     *
     * @param eval		evaluation object
     */
    public double getValue(Evaluation eval) {
        ConfusionMatrix<Integer> matrix = eval.getConfusion();
        return this.compute(matrix);
    }

    /**
     * Compute the negative likelihood ratio.
     *
     * @param matrix	confusion matrix for this result
     *
     * @return a measure of how likely a negative prediction is correct
     */
    protected double negativeLikelihood(ConfusionMatrix<Integer> matrix) {
        int tp = truePositive(matrix);
        int tn = trueNegative(matrix);
        int fp = falsePositive(matrix);
        int fn = falseNegative(matrix);
        double num = ratio(tn + 1, fp + 1);
        double den = ratio(fn + 1, tp + 1);
        double retVal = (den > 0 ? num / den : Double.POSITIVE_INFINITY);
        return Math.log10(retVal);
    }

    /**
     * Compute the positive likelihood ratio.
     *
     * @param matrix	confusion matrix for this result
     *
     * @return a measure of how likely a positive prediction is correct
     */
    protected double positiveLikelihood(ConfusionMatrix<Integer> matrix) {
        int tp = truePositive(matrix);
        int tn = trueNegative(matrix);
        int fp = falsePositive(matrix);
        int fn = falseNegative(matrix);
        double den = ratio(fp + 1, tn + 1);
        double num = ratio(tp + 1, fn + 1);
        double retVal = (den > 0 ? num / den : Double.POSITIVE_INFINITY);
        return Math.log10(retVal);
    }

    /**
     * Compute a metric from a numerator and a denominator increment.
     *
     * @param num	numerator
     * @param den	denominator increment
     *
     * @return 0.0 if the numerator is 0, else num / (num + den)
     */
    protected static double ratio(int num, int den) {
        double retVal = 0.0;
        if (num > 0)
            retVal = num / (double) (num + den);
        return retVal;
    }



    /**
     * @return the value of this metric
     *
     * @param matrix	confusion matrix
     */
    public abstract double compute(ConfusionMatrix<Integer> matrix);

    /**
     * Compute the number of true positives.
     *
     * @param matrix	incoming confusion matrix
     *
     * @return the number of positive predictions that are correct
     */
    protected int truePositive(ConfusionMatrix<Integer> matrix) {
        final int nClasses = matrix.getClasses().size();
        int retVal = 0;
        for (int i = 1; i < nClasses; i++)
            retVal += matrix.getCount(i, i);
        return retVal;
    }

    /**
     * Compute the number of true negatives.
     *
     * @param matrix	incoming confusion matrix
     *
     * @return the number of negative predictions that are correct
     */
    protected int trueNegative(ConfusionMatrix<Integer> matrix) {
        return matrix.getCount(0, 0);
    }

    /**
     * Compute the number of false positives.
     *
     * @param matrix	incoming confusion matrix
     *
     * @return the number of positive predictions that are incorrect
     */
    protected int falsePositive(ConfusionMatrix<Integer> matrix) {
        final int nClasses = matrix.getClasses().size();
        int retVal = 0;
        for (int i = 1; i < nClasses; i++) {
            for (int j = 0; j < nClasses; j++) {
                if (i != j)
                    retVal += matrix.getCount(j, i);
            }
        }
        return retVal;
    }

    /**
     * Compute the number of false negatives.
     *
     * @param matrix	incoming confusion matrix
     *
     * @return the number of negative predictions that are incorrect
     */
    protected int falseNegative(ConfusionMatrix<Integer> matrix) {
        final int nClasses = matrix.getClasses().size();
        int retVal = 0;
        for (int j = 1; j < nClasses; j++)
            retVal += matrix.getCount(j, 0);
        return retVal;
    }

}
