/**
 *
 */
package org.theseed.dl4j.decision;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This class contains a proposal for splitting a choice node.  The best splitter has
 * the highest information gain.  Note that this is not a set-capable ordering, since
 * two different split schemes can compare equal.
 */
public class Splitter implements Comparable<Splitter> {

    /** index of feature to split on */
    private int feature;
    /** split point */
    private double limit;
    /** left entropy */
    private double leftEntropy;
    /** right entropy */
    private double rightEntropy;
    /** left node count */
    private int leftCount;
    /** right node count */
    private int rightCount;
    /** information gain */
    private double gain;
    /** null splitter, indicating do not split */
    public static final Splitter NULL = new Splitter();

    /**
     * Create a null splitter.
     */
    private Splitter() {
        this.feature = -1;
        this.gain = 0.0;
    }

    /**
     * Create a splitter from known label sums.
     */

    /**
     * Compute a split proposal for a feature from a set of data rows.
     *
     * @param feature		index of the feature being used to split
     * @param limit			value to split on
     * @param data			data rows to split
     * @param oldEntropy	entropy at the current node
     */
    public static Splitter computeSplitter(int feature, double limit, int nClasses, List<DataSet> data, double oldEntropy) {
        // Split the data rows on the left and the right.
        INDArray leftLabels = Nd4j.zeros(nClasses);
        INDArray rightLabels = Nd4j.zeros(nClasses);
        for (DataSet row : data) {
            INDArray labelArray = row.getLabels();
            if (row.getFeatures().getDouble(feature) <= limit)
                leftLabels.addi(labelArray);
            else
                rightLabels.addi(labelArray);
        }
        return computeSplitter(feature, limit, oldEntropy, leftLabels, rightLabels);
    }

    /**
     * Compute a split proposal for a feature from the left and right label sums.
     *
     * @param feature		index of the feature being used to split
     * @param limit			value to split on
     * @param oldEntropy	entropy at the current node
     * @param leftLabels	sum of the labels on the left
     * @param rightLabels	sum of the labels on the right
     */
    public static Splitter computeSplitter(int feature, double limit, double oldEntropy, INDArray leftLabels,
            INDArray rightLabels) {
        Splitter retVal;
        int leftCount = leftLabels.sumNumber().intValue();
        int rightCount = rightLabels.sumNumber().intValue();
        if (leftCount <= 0 || rightCount <= 0)
            retVal = NULL;
        else {
            // Here we have a useful split.
            retVal = new Splitter();
            retVal.feature = feature;
            retVal.limit = limit;
            retVal.leftEntropy = DecisionTree.labelEntropy(leftLabels);
            retVal.rightEntropy = DecisionTree.labelEntropy(rightLabels);
            retVal.gain = oldEntropy - (retVal.leftEntropy * leftCount + retVal.rightEntropy * rightCount) / (leftCount + rightCount);
            retVal.leftCount = leftCount;
            retVal.rightCount = rightCount;
        }
        return retVal;
    }

    /**
     * Here we sort better splits to the beginning, so a negative number is returned if
     * this is the better split.
     */
    @Override
    public int compareTo(Splitter o) {
        // Start by comparing the better gain.
        int retVal = Double.compare(o.gain, this.gain);
        if (retVal == 0) {
            // If the gain is the same, look for the more even split.
            retVal = Math.abs(this.leftCount - this.rightCount) - Math.abs(o.leftCount - o.rightCount);
        }
        return retVal;
    }

    /**
     * @return a choice node created from this splitter
     */
    protected DecisionTree.ChoiceNode createNode(double entropy) {
        return new DecisionTree.ChoiceNode(this.feature, this.limit, entropy, this.gain);
    }

    /**
     * @return the number of rows that would split left
     */
    public int getLeftCount() {
        return this.leftCount;
    }

    /**
     * @return the nubmer of rows that would split right
     */
    public int getRightCount() {
        return this.rightCount;
    }

    /**
     * @return the entropy on the left
     */
    public double getLeftEntropy() {
        return this.leftEntropy;
    }

    /**
     * @return the entropy on the right
     */
    public double getRightEntropy() {
        return this.rightEntropy;
    }

    /**
     * @return TRUE if the specified dataset row would split left, else FALSE
     *
     * @param row		row to check
     */
    public boolean splitsLeft(DataSet row) {
        double value = row.getFeatures().getDouble(this.feature);
        return (value <= limit);
    }

}
