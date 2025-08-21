/**
 *
 */
package org.theseed.dl4j.decision;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.dl4j.train.ClassPredictError;
import org.theseed.dl4j.train.ITrainReporter;
import org.theseed.utils.IDescribable;

/**
 * A random forest is a set of decision trees, each trained on a randomly-selected subset of the
 * full training set.  An entire forest predicts an outcome by voting.
 *
 * @author Bruce Parrello
 *
 */
public class RandomForest implements Serializable {

    // FIELDS
    /** logging facility */
    private static final Logger log = LoggerFactory.getLogger(RandomForest.class);
    /** serialization version ID */
    private static final long serialVersionUID = -5802362692626598850L;
    /** random number generator */
    private static Random rand = new Random();
    /** hyperparameters */
    private transient Parms parms;
    /** trees in this forest */
    private List<DecisionTree> trees;
    /** number of labels for this tree */
    private final int nLabels;
    /** number of input features for this tree */
    private int nFeatures;
    /** randomizer for selecting training sets */
    private transient IRandomizer randomizer;
    /** split point finder */
    private transient Iterator<TreeFeatureSelectorFactory> factoryIter;
    /** progress monitor for creation */
    private transient ITrainReporter monitor;
    /** number of trees built during creation */
    private transient int treesDone;

    /**
     * type of randomization
     *
     * BALANCED-- random with replacement, equal numbers of each class
     * UNIQUE-- random without replacement
     * RANDOM-- random with replacement
     */
    public static enum Method implements IDescribable {
        BALANCED {
            @Override
            IRandomizer create() {
                return new BalancedRandomizer();
            }

            @Override
            public String getDescription() {
                return "Class-balanced example sets with replacement.";
            }
        }, UNIQUE {
            @Override
            IRandomizer create() {
                return new NonReplacingRandomizer();
            }

            @Override
            public String getDescription() {
                return "Random example sets without replacement.";
            }
        }, RANDOM {
            @Override
            IRandomizer create() {
                return new ReplacingRandomizer();
            }

            @Override
            public String getDescription() {
                return "Random example sets with replacement.";
            }
        };

        /**
         * @return a randomizer of the appropriate type
         */
        abstract IRandomizer create();


    }

    /**
     * Initialize the randomizer with a specified seed.
     *
     * @param seed	randomization seed to use
     */
    public static void setSeed(int seed) {
        rand = new Random(seed);
    }

    /**
     * This class represents the hyperparameters for the random forest.
     */
    public static class Parms {

        /** number of trees */
        private int nTrees;
        /** number of features to use at each node */
        private int nFeatures;
        /** minimum number of features for a leaf node */
        private int leafLimit;
        /** number of examples to use for each tree */
        private int nExamples;
        /** type of randomization */
        private Method method;
        /** maximum tree depth */
        private int maxDepth;


        /**
         * Construct hyperparameters with default values.
         */
        public Parms() {
            this.nTrees = 50;
            this.nFeatures = 10;
            this.leafLimit = 1;
            this.nExamples = 1000;
            this.method = Method.RANDOM;
            this.maxDepth = 50;
        }

        /**
         * Construct hyperparameters with reasonable values for a specified training set.
         *
         * @param dataset	training set to use for computing the parameters
         */
        public Parms(DataSet dataset) {
            this.setup(dataset.numExamples(), dataset.numInputs());
        }

        /**
         * Construct hyperparameters with reasonable values for a training set with the
         * specified characteristics.
         *
         * @param nExamples		number of input rows
         * @param nInputs		number of feature columns
         */
        public Parms(int nExamples, int nInputs) {
            this.setup(nExamples, nInputs);
        }

        /**
         * Initialize the hyperparameters with reasonable values for a training set with the
         * specified characteristics.
         *
         * @param nRows		number of input rows
         * @param nInputs	number of feature columns
         */
        private void setup(int nRows, int nInputs) {
            this.nTrees = 50;
            int min = nInputs * 4 / this.nTrees;
            int middle = (int) Math.sqrt(nInputs) + 1;
            int max = nInputs / 2;
            this.nFeatures = (middle < min ? min : middle);
            if (this.nFeatures > max) this.nFeatures = max;
            this.leafLimit = 1;
            this.nExamples = nRows / 5;
            this.method = Method.RANDOM;
            this.maxDepth = 2 * nInputs;
        }

        /**
         * @return the number of trees to build
         */
        public int getNumTrees() {
            return this.nTrees;
        }

        /**
         * Specify the number of trees to build.
         *
         * @param nTrees 	the number of trees to build
         */
        public Parms setNumTrees(int nTrees) {
            this.nTrees = nTrees;
            return this;
        }

        /**
         * @return the number of features to test at each choice node
         */
        public int getNumFeatures() {
            return this.nFeatures;
        }

        /**
         * Set the number of features to test at each choice node.
         *
         * @param nFeatures 	the number of features to set
         */
        public Parms setNumFeatures(int nFeatures) {
            this.nFeatures = nFeatures;
            return this;
        }

        /**
         * @return the number of examples that triggers formation of a leaf node
         */
        public int getLeafLimit() {
            return this.leafLimit;
        }

        /**
         * Specify the number of examples that triggers formation of a leaf node.
         *
         * @param leafLimit 	the leafLimit to set
         */
        public Parms setLeafLimit(int leafLimit) {
            this.leafLimit = leafLimit;
            return this;
        }

        /**
         * @return the number of examples to use for each tree
         */
        public int getNumExamples() {
            return this.nExamples;
        }

        /**
         * Specify the number of examples to use for each tree.
         *
         * @param nExamples the number of examples to use
         */
        public Parms setNumExamples(int nExamples) {
            this.nExamples = nExamples;
            return this;
        }

        /**
         * Specify the type of randomizer.
         *
         * @param method	randomization method
         */
        public Parms setMethod(Method method) {
            this.method = method;
            return this;
        }

        /**
         * @return the randomizer to use for selecting training sets
         */
        public IRandomizer getRandomizer() {
            return this.method.create();
        }

        /**
         * @return the randomizing method
         */
        public Method getMethod() {
            return this.method;
        }

        /**
         * @return the maximum permissible tree depth
         */
        public int getMaxDepth() {
            return this.maxDepth;
        }

        /**
         * Specify the maximum permissible tree depth
         *
         * @param maxDepth 	the depth to set
         */
        public Parms setMaxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }
    }

    /**
     * Construct a forest based on the specified training set.
     *
     * @param dataset		training set to use
     * @param parms			hyper-parameters
     * @param factoryIter	iterator for feature selector factories to be used in producing the forest
     * @param mon			progress monitor (or NULL if none)
     */
    public RandomForest(DataSet dataset, Parms parms, Iterator<TreeFeatureSelectorFactory> factoryIter, ITrainReporter mon) {
        this.nLabels = dataset.numOutcomes();
        buildForest(dataset, parms, factoryIter, mon);
    }

    /**
     * Build a forest based on the specified training set.
     *
     * @param dataset		training set to use
     * @param parms			hyper-parameters
     * @param factoryIter	iterator for feature selector factories to be used in producing the forest
     * @param mon			progress monitor
     */
    private void buildForest(DataSet dataset, Parms parms, Iterator<TreeFeatureSelectorFactory> factoryIter, ITrainReporter mon) {
        this.nFeatures = dataset.numInputs();
        this.parms = parms;
        this.factoryIter = factoryIter;
        this.randomizer = parms.getRandomizer();
        this.treesDone = 0;
        this.monitor = mon;
        // Initialize the randomizer.
        log.debug("Initializing randomizer for {} examples.", this.parms.getNumExamples());
        this.randomizer.initializeData(this.nLabels, this.parms.getNumExamples(), dataset);
        // Create an array of randomizer seeds.
        log.debug("Initializing seeds for {} trees.", this.parms.getNumTrees());
        long[] seeds = rand.longs(this.parms.getNumTrees()).toArray();
        // Create an array of tree selector factories.  This is a sequential operation, so we have to do it here,
        // not in the parallel stream below.
        log.debug("Creating factories.");
        TreeFeatureSelectorFactory[] factories = IntStream.range(0, this.parms.getNumTrees())
                .mapToObj(i -> this.factoryIter.next()).toArray(TreeFeatureSelectorFactory[]::new);
        // Create the decision trees in the random forest.
        log.debug("Creating trees.");
        this.trees = IntStream.range(0, this.parms.getNumTrees()).parallel()
                .mapToObj(i -> this.buildTree(i, seeds[i], factories[i]))
                .collect(Collectors.toList());
    }

    /**
     * Construct a random forest with the standard tree feature-selection factory.
     *
     * @param dataset	training set to process
     * @param hParms	hyper-parameters
     */
    public RandomForest(DataSet dataset, Parms hParms) {
        this.nLabels = dataset.numOutcomes();
        int[] idxes = getUsefulFeatures(dataset);
        Iterator<TreeFeatureSelectorFactory> treeIter = new NormalTreeFeatureSelectorFactory(rand.nextLong(),
                idxes, hParms.getNumFeatures(), hParms.getNumTrees());
        this.buildForest(dataset, hParms, treeIter, null);
    }

    /**
     * Create a decision tree from a balanced random subset of the rows in this dataset.
     * Note that all the trees are built in parallel, so care has been taken not to modify
     * the incoming parameters.
     *
     * @param id			ID number of the tree
     * @param seed			seed to use for data randomization
     * @param factory		feature selector factory for feature randomization
     *
     * @return a decision tree for the sampled subset
     */
    private DecisionTree buildTree(int id, long seed, TreeFeatureSelectorFactory factory) {
        // Get the sampling to use for training this tree.
        DataSet sample = this.randomizer.getData(seed);
        // Build the decision tree.
        DecisionTree retVal = new DecisionTree(sample, this.parms, factory);
        if (this.monitor != null)
            this.reportTree(retVal);
        return retVal;
    }

    /**
     * Report completion of a new tree to the progress monitor.  Here the epoch is the number of trees completed,
     * the score is 1 minus the new tree's accuracy, and the rating is the max accuracy so far.
     *
     * @param tree	new decision tree
     */
    private synchronized void reportTree(DecisionTree tree) {
        this.treesDone++;
        double score = tree.score();
        try {
            this.monitor.displayEpoch(this.treesDone, score, 0.0, false);
        } catch (InterruptedException e) {
            // Just ignore the exception.
            log.error(e.toString());
        }


    }

    /**
     * Predict the classifications for a set of features.
     *
     * @param features		array of features to predict
     */
    public INDArray predict(INDArray features) {
        // Get an empty result matrix.
        INDArray retVal = Nd4j.zeros(features.rows(), this.nLabels);
        // Ask each tree to vote.
        for (DecisionTree tree : this.trees)
            tree.vote(features, retVal);
        return retVal;
    }

    /**
     * @return the accuracy of a classifier for the specified testing set
     *
     * @param testSet		testing set to check
     */
    public double getAccuracy(DataSet testSet) {
        // Get the predictions.
        INDArray predictions = this.predict(testSet.getFeatures());
        // Compare to the expectations.
        int good = 0;
        int total = 0;
        INDArray expected = testSet.getLabels();
        for (int r = 0; r < predictions.rows(); r++) {
            if (ClassPredictError.computeBest(predictions, r) == ClassPredictError.computeBest(expected, r))
                good++;
            total++;
        }
        return ((double) good) / total;
    }

    /**
     * Compute the impact of each input on the classifications.  This is a one-dimensional array with
     * a number for each input.
     */
    public INDArray computeImpact() {
        INDArray retVal = Nd4j.zeros(this.nFeatures);
        // Accumulate each tree's impact.
        for (DecisionTree tree : this.trees)
            tree.accumulateImpact(retVal);
        // Take the mean.
        retVal.divi(this.trees.size());
        return retVal;
    }

    /**
     * Save this model to the specified file.
     *
     * @param saveFile	output file to contain this random forest
     *
     * @throws IOException
     * @throws FileNotFoundException
     */
    public void save(File saveFile) throws FileNotFoundException, IOException {
        try (FileOutputStream fileStream = new FileOutputStream(saveFile)) {
            ObjectOutputStream outStream = new ObjectOutputStream(fileStream);
            outStream.writeObject(this);
        }
    }

    /**
     * @return a random forest model loaded from the specified file
     *
     * @param loadFile		file containing the model
     */
    public static RandomForest load(File loadFile) throws IOException {
        RandomForest retVal;
        try (FileInputStream fileStream = new FileInputStream(loadFile)) {
            ObjectInputStream inStream = new ObjectInputStream(fileStream);
            retVal = (RandomForest) inStream.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException("Invalid format for " + loadFile + ": " + e.toString());
        }
        return retVal;
    }

    /**
     * Convert a feature array from the 4-dimensional shape used by neural nets to a standard 2 dimensions.
     *
     * @param features		feature array to reshape
     *
     * @return the incoming list of examples, flattened to a row/column matrix
     */
    public static INDArray flattenFeatures(INDArray features) {
        return features.reshape(features.size(0), features.size(1) * features.size(3));
    }

    /**
     * Convert a dataset's feature array from the 4-dimensional shape used by neural nets to a standard 2 dimensions.
     *
     * @param dataset		dataset to convert
     */
    public static void flattenDataSet(DataSet dataset) {
        dataset.setFeatures(flattenFeatures(dataset.getFeatures()));
    }

    /**
     * Apply the model to an input file to make predictions, and write them to an output file.
     *
     * @param inFile		input file containing samples
     * @param outFile		output file for predictions
     * @param metaCols		list of metadata column names
     *
     * @throws IOException
     */
    public void makePredictions(File inFile, File outFile, List<String> metaCols, List<String> labels)
            throws IOException {
        // Open the input and output files.
        TabbedDataSetReader batches = new TabbedDataSetReader(inFile, metaCols);
        try (PrintWriter writer = new PrintWriter(outFile)) {
            // Write the output header.
            writer.println(StringUtils.join(metaCols, '\t') + "\tpredicted");
            // Loop through the input data.
            for (DataSet batch : batches) {
                // Get the features in this batch.
                INDArray features = RandomForest.flattenFeatures(batch.getFeatures());
                // Make the predictions and extract the metadata values.
                INDArray output = this.predict(features);
                List<String> metaRows = batch.getExampleMetaData(String.class);
                // Now we write the output rows.
                for (int i = 0; i < features.rows(); i++) {
                    int labelIdx = ClassPredictError.computeBest(output, i);
                    writer.println(metaRows.get(i) + "\t" + labels.get(labelIdx));
                }
            }
        } finally {
            batches.close();
        }
    }

    /**
     * Compute the useful columns in a training set.  Only columns with a variance in the input values
     * will be included in the output.
     *
     * @param trainingSet	training set to scan
     *
     * @return an array of the column indices for the useful columns
     */
    public static int[] getUsefulFeatures(DataSet trainingSet) {
        // We will use this array to identify the useful columns.
        boolean[] flags = new boolean[trainingSet.numInputs()];
        // Analyze each column.
        INDArray features = trainingSet.getFeatures();
        final int n = features.rows();
        for (int i = 0; i < flags.length; i++) {
            // Get the first value.  Check every other row until we find a different one.
            double val0 = features.getDouble(0, i);
            flags[i] = false;
            for (int j = 1; j < n && ! flags[i]; j++)
                flags[i] = features.getDouble(j, i) != val0;
        }
        // Now every nontrivial feature column has TRUE in the flag array.
        int[] retVal = IntStream.range(0, flags.length).filter(i -> flags[i]).toArray();
        return retVal;
    }

}
