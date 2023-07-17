/**
 *
 */
package org.theseed.dl4j.decision;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A decision tree is a serializable data structure that can be used to classify an item based on a set of
 * features.  Each node of the tree specifies a feature and a threshold.  Values less than or equal to the
 * threshold are classified on the left and those greater than the threshold are classified on the right.
 * At the leaf level an output class is specified.
 *
 * @author Bruce Parrello
 *
 */
public class DecisionTree implements Serializable {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(DecisionTree.class);
    /** number of features */
    private int nFeatures;
    /** number of classes */
    private int nClasses;
    /** root node */
    private Node root;
    /** number of nodes in tree */
    private int size;
    /** hyperparameters */
    private transient RandomForest.Parms parms;
    /** feature selector factory for training */
    private transient TreeFeatureSelectorFactory factory;
    /** log base 2 factor */
    private static double LOG2BASE = Math.log(2.0);
    /** object ID for serialization */
    private static final long serialVersionUID = 4184229432605504479L;

    /**
     * This nested class represents a tree node.
     */
    public static abstract class Node implements Serializable {

        /** serialization ID */
        private static final long serialVersionUID = 6974234637504879773L;
        /** entropy value at this node */
        private double entropy;

        protected Node(double entropy) {
            this.entropy = entropy;
        }

        /**
         * @return the entropy
         */
        public double getEntropy() {
            return this.entropy;
        }

        /**
         * Accumulate the impact of this node's feature on the result.
         *
         * @param vector	vector of feature impacts
         */
        protected abstract void addImpact(INDArray vector);

        /**
         * The score of a node is its entropy if it is a leaf.  If it is a choice node, it is
         * the mean score of the two children.
         *
         * @return the score of this node
         */
        public abstract double getScore();

    }

    /**
     * This is a decision node, with two children.
     */
    public static class ChoiceNode extends Node {

        /** serialization ID */
        private static final long serialVersionUID = -3772567268579272613L;
        /** index of deciding feature */
        private int iFeature;
        /** threshold (inclusive on the left) */
        private double limit;
        /** impurity gain */
        private double gain;
        /** left child */
        private Node left;
        /** right child */
        private Node right;

        /**
         * Create a new node with the specified decision criteria.
         *
         * @param iFeat		index of deciding feature
         * @param lim		maximum feature value for left children
         * @param entropy	entropy of this node
         * @param gain		impurity gain
         */
        protected ChoiceNode (int iFeat, double lim, double entropy, double gain) {
            super(entropy);
            this.iFeature = iFeat;
            this.limit = lim;
            this.gain = gain;
            this.left = null;
            this.right = null;
        }

        /**
         * Return the child of this node relevant to the specified input row.
         *
         * @param features	array of features containing the input row
         * @param idx		row to test
         *
         * @return the child node for the selected feature
         */
        public Node choose(INDArray features, int idx) {
            double value = features.getDouble(idx, this.iFeature);
            Node retVal = (value > limit ? this.right : this.left);
            return retVal;
        }

        /**
         * Attach the left-side child.
         *
         * @param left 	the left-side child node
         */
        public void setLeft(Node left) {
            this.left = left;
        }

        /**
         * Attach the right-side child.
         *
         * @param right 	the right-side child node
         */
        public void setRight(Node right) {
            this.right = right;
        }

        /**
         * @return the impurity gain for this node
         */
        public double getGain() {
            return this.gain;
        }

        /**
         * @return the decision feature index for this node
         */
        public int getFeatureIdx() {
            return this.iFeature;
        }

        @Override
        protected void addImpact(INDArray vector) {
            vector.putScalar(this.iFeature, this.gain + vector.getDouble(this.iFeature));
            this.left.addImpact(vector);
            this.right.addImpact(vector);
        }

        @Override
        public String toString() {
            return "ChoiceNode@" + Integer.toHexString(System.identityHashCode(this)) +
                    "[iFeature=" + this.iFeature + ", limit=" + this.limit + "]";
        }

        @Override
        public double getScore() {
            return (this.left.getScore() + this.right.getScore()) / 2.0;
        }

    }

    /**
     * This class represents a leaf node.
     */
    public static class LeafNode extends Node {

        // FIELDS
        /** index of predicted class */
        private int iClass;
        /** serialization type ID */
        private static final long serialVersionUID = 3328808945708275642L;

        /**
         * Construct a leaf node.
         *
         * @param iClass	index of class decided by leaf node
         * @param entropy	entropy of this node
         */
        protected LeafNode(int iClass, double entropy) {
            super(entropy);
            this.iClass = iClass;
        }

        /**
         * @return the class decided by this leaf
         */
        public int getiClass() {
            return this.iClass;
        }

        @Override
        protected void addImpact(INDArray vector) {
            // A leaf has no impact.
        }

        @Override
        public String toString() {
            return "LeafNode@" + Integer.toHexString(System.identityHashCode(this)) +
                    "[iClass=" + this.iClass + "]";
        }

        @Override
        public double getScore() {
            return this.getEntropy();
        }

    }

    /**
     * Create a decision tree for the specified dataset.
     *
     * @param dataset	dataset to use for training the tree
     * @param parms		hyperparameter specification
     * @param finder	split point finder
     */
    public DecisionTree(DataSet dataset, RandomForest.Parms parms, TreeFeatureSelectorFactory factory) {
        this.nClasses = dataset.numOutcomes();
        this.nFeatures = dataset.numInputs();
        this.parms = parms;
        this.factory = factory;
        this.size = 0;
        // Compute the number of features to use in each tree.
        int arraySize = parms.getNumFeatures();
        if (this.nFeatures < arraySize) arraySize = this.nFeatures;
        // Compute the starting entropy.
        double entropy = DecisionTree.entropy(dataset);
        // Split the dataset into rows.
        List<DataSet> rows = dataset.asList();
        // Create the root node.
        this.root = this.computeNode(rows, 0, entropy);
    }

    /**
     * @return the entropy value of the dataset
     *
     * @param dataset	dataset whose entropy is desired
     */
    public static double entropy(DataSet dataset) {
        double total = dataset.numExamples();
        double retVal = 0.0;
        if (total > 0) {
            INDArray labelCounts = getLabelSums(dataset);
            retVal = labelEntropy(labelCounts);
        }
        return retVal;
    }

    /**
     * @return the entropy indicated by an array of label counts
     *
     * @param labelSum	array of label counts
     */
    public static double labelEntropy(INDArray labelSum) {
        double total = labelSum.sumNumber().doubleValue();
        double retVal = 0.0;
        for (int i = 0; i < labelSum.columns(); i++) {
            double pI = labelSum.getDouble(i) / total;
            if (pI > 0.0)
                retVal -= pI * Math.log(pI);
        }
        retVal /= LOG2BASE;
        return retVal;
    }

    /**
     * Recursively compute the tree node from the specified list of features.
     *
     * @param rows				dataset rows to be classified by this node
     * @param depth				depth of the node in question
     * @param entropy			entropy of the set
     *
     * @return a node for deciding this set
     */
    private Node computeNode(List<DataSet> rows, int depth, double entropy) {
        Node retVal;
        // Is this a leaf?
        if (rows.size() <= this.parms.getLeafLimit() || entropy <= 0.0 || depth >= this.parms.getMaxDepth()) {
            // Yes.  Assign the best label value.
            retVal = createLeaf(rows, entropy);
        } else {
            // Loop through the features, looking for the one that creates the greatest entropy decrease.
            // These variables contain the data we need to split the dataset for the children.
            Splitter best = Splitter.NULL;
            // Loop through the features left to examine.
            FeatureSelector selector = this.factory.getSelector(depth);
            SplitPointFinder finder = selector.getFinder();
            for (int i : selector.getFeaturesToUse()) {
                Splitter test = finder.computeSplit(i, this.nClasses, rows, entropy);
                if (test.compareTo(best) < 0)
                    best = test;
            }
            // If we were not able to gain anything, this is a leaf.
            if (best == Splitter.NULL)
                retVal = this.createLeaf(rows, entropy);
            else {
                // Here we can split the node.
                ChoiceNode newNode = best.createNode(entropy);
                // Split the incoming dataset.
                List<DataSet> left = new ArrayList<DataSet>(best.getLeftCount());
                List<DataSet> right = new ArrayList<DataSet>(best.getRightCount());
                for (DataSet row : rows) {
                    if (best.splitsLeft(row))
                        left.add(row);
                    else
                        right.add(row);
                }
                // Create the left node.
                newNode.setLeft(this.computeNode(left, depth + 1, best.getLeftEntropy()));
                // Create the right node.  Here we can reuse the old bitmap.
                newNode.setRight(this.computeNode(right, depth + 1, best.getRightEntropy()));
                // Return the new node.
                retVal = newNode;
            }
        }
        this.size++;
        return retVal;
    }

    /**
     * @return the mean value of the specified feature in the list of dataset rows
     *
     * @param rows	list of dataset rows
     * @param i		column index of the desired feature
     */
    public static double featureMean(List<DataSet> rows, int i) {
        int count = 0;
        double retVal = 0.0;
        for (DataSet row : rows) {
            retVal += row.getFeatures().getDouble(i);
            count++;
        }
        if (count > 0) retVal /= count;
        return retVal;
    }

    /**
     * @return a leaf node for a dataset
     *
     * @param rows	dataset represented by the leaf
     * @param entropy	entropy of the dataset
     */
    private Node createLeaf(List<DataSet> rows, double entropy) {
        int label = DecisionTree.bestLabel(rows);
        this.size++;
        return new LeafNode(label, entropy);
    }

    /**
     * @return the index of the most popular label in the list of dataset rows
     *
     * @param rows	list of dataset rows
     */
    private static int bestLabel(List<DataSet> rows) {
        int retVal = 0;
        if (rows.size() > 0) {
            INDArray labelSums = rows.get(0).getLabels().dup();
            for (int i = 1; i < rows.size(); i++)
                labelSums.addi(rows.get(i).getLabels());
            retVal = computeBest(labelSums);
        }
        return retVal;
    }

    /**
     * @return the index of the most popular label in the dataset
     *
     * @param dataset	dataset to process
     */
    public static int bestLabel(DataSet dataset) {
        INDArray labelSums = getLabelSums(dataset);
        int retVal = computeBest(labelSums);
        return retVal;
    }

    /**
     * @return the highest value in a vector of label sums
     *
     * @param labelSums		vector to search
     */
    private static int computeBest(INDArray labelSums) {
        int retVal = 0;
        double max = labelSums.getDouble(0);
        for (int i = 1; i < labelSums.columns(); i++) {
            double val = labelSums.getDouble(i);
            if (val > max) {
                retVal = i;
                max = val;
            }
        }
        return retVal;
    }

    /**
     * @return a vector containing the number of occurrences of each label in the dataset
     *
     * @param dataset	dataset of interest
     */
    private static INDArray getLabelSums(DataSet dataset) {
        return dataset.getLabels().sum(0);
    }

    /**
     * @return the index of the predicted label for the specified data row
     *
     * @param features	array of input rows
     * @param idx		index of row whose label is desired
     */
    public int predict(INDArray features, int idx) {
        Node current = root;
        while (current instanceof ChoiceNode) {
            ChoiceNode curr = (ChoiceNode) current;
            current = curr.choose(features, idx);
        }
        LeafNode curr = (LeafNode) current;
        return curr.getiClass();
    }

    /**
     * Add this tree's vote to the predictions of a dataset.
     *
     * @param features	array of input rows
     * @param labels	array in which to put label votes
     */
    public void vote(INDArray features, INDArray labels) {
        for (int i = 0; i < features.rows(); i++) {
            // Compute the prediction for this row.
            int label = this.predict(features, i);
            // Add our vote to its label output.
            labels.putScalar(i, label, labels.getDouble(i, label) + 1.0);
        }
    }

    /**
     * @return an estimate of relative variable impact for each input variable
     */
    public INDArray computeImpact() {
        // Denote each variable initially has zero impact.
        INDArray retVal = Nd4j.zeros(this.nFeatures);
        accumulateImpact(retVal);
        // Return the result.
        return retVal;
    }

    /**
     * Add this tree's impact into an impact array.
     *
     * @param impactArray	array in which to accumulate this tree's impact
     */
    protected void accumulateImpact(INDArray impactArray) {
        // Process all the nodes.
        this.root.addImpact(impactArray);
    }

    /**
     * @return the number of nodes in the tree
     */
    public int size() {
        return this.size;
    }

    /**
     * @return the score of this tree, which is a function of the leaf entropy
     */
    public double score() {
        return this.root.getScore();
    }

}
