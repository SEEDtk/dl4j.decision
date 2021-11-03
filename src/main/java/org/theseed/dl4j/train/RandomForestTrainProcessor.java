/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.math3.stat.inference.OneWayAnova;
import org.apache.commons.text.TextStringBuilder;
import org.kohsuke.args4j.Option;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.counters.Rating;
import org.theseed.dl4j.DistributedOutputStream;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.dl4j.decision.DecisionTree;
import org.theseed.dl4j.decision.RandomForest;
import org.theseed.dl4j.decision.TreeFeatureSelectorFactory;
import org.theseed.io.Shuffler;
import org.theseed.io.TabbedLineReader;
import org.theseed.reports.ClassTestValidationReport;
import org.theseed.reports.ClassValidationReport;
import org.theseed.reports.IValidationReport;
import org.theseed.reports.TestValidationReport;
import org.theseed.utils.ParseFailureException;

/**
 * This class represents a random-forest model.  This is the fastest and simplest of the three main ML models.
 *
 * The positional parameter is the name of the model directory.  The directory should contain the training data
 * in the file "training.tbl" and the label values in the file "labels.txt".  The model itself will be stored
 * in "model.ser".  The following command-line parameters are supported.
 *
 * -c	index (1-based) or name of the column containing the classification labels; the
 * 		default is 1 (first column)
 * -t	size of the testing set, which is the first batch read and is used to compute
 * 		normalization; the default is 2000
 * -s	seed value to use for random number generation; the default is to use the last
 * 		20 bits of the current time
 * -i	name of the training set file; the default is "training.tbl" in the model directory
 *
 * --meta			a comma-delimited list of the metadata columns; these columns are ignored during training; the
 * 					default is none
 * --comment		a comment to display at the beginning of the trial log
 * --name			name for the model file; the default is "model.ser" in the model directory
 * --parms			name of a file to contain a dump of the current parameters
 * --help			display the command-line options and parameters
 * --id				name of the ID column in the training file; this is also a meta-data column and must be unique
 * --method			randomization method (BALANCED, UNIQUE, RANDOM)
 * --maxFeatures	number of features to use at each tree node
 * --nEstimators	number of trees in the forest
 * --minSplit		minimum number of examples allowed in a leaf node
 * --maxDepth		maximum tree depth
 * --sampleSize		number of examples to use for each tree's subset
 * --selection		feature selection mode (default NORMAL)
 * --rootFile		if feature selection mode is ROOTED, the name of a file containing the tree root feature names in the first column
 *
 * @author Bruce Parrello
 *
 */
public class RandomForestTrainProcessor extends ModelProcessor implements ITrainingProcessor {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(RandomForestTrainProcessor.class);
    /** trained model */
    private RandomForest model;
    /** hyper-parameters */
    private RandomForest.Parms hParms;
    /** result report */
    private String resultReport;
    /** impact report */
    private SortedSet<Rating<String>> impact;
    /** tree feature selection factory iterator */
    private Iterator<TreeFeatureSelectorFactory> factoryIter;

    // COMMAND-LINE OPTIONS

    /** parameter dump */
    @Option(name="--parms", usage="write command-line parameters to a configuration file")
    private File parmFile;

    /** label column ID or number */
    @Option(name = "-c", aliases = { "--col" }, metaVar = "result", usage = "label column name")
    private String labelCol;

    /** randomization method for choosing example rows */
    @Option(name = "--method", usage = "randomization method")
    private RandomForest.Method method;

    /** number of features to use at each tree node */
    @Option(name = "--maxFeatures", metaVar = "10", usage = "number of features to interrogate at each splitting tree node")
    private int maxFeatures;

    /** minimum number of examples allowed in a choice node */
    @Option(name = "--minSplit", metaVar = "2", usage = "minimum number of examples allowed in a leaf node")
    private int minSplit;

    /** number of trees in the forest */
    @Option(name = "--nEstimators", metaVar = "100", usage = "number of trees in the forest")
    private int nEstimators;

    /** maximum permissible tree depth */
    @Option(name = "--maxDepth", metaVar = "10", usage = "maximum tree depth")
    private int maxDepth;

    /** size of each input batch */
    @Option(name = "-b", aliases = { "--batchSize" }, metaVar = "1000", usage = "size of each input batch")
    protected int batchSize;

    /** number of samples to use for each tree's subset */
    @Option(name = "--sampleSize", metaVar = "100", usage = "sample size to use for each tree")
    private int sampleSize;

    /** method for determining feature selection */
    @Option(name = "--selection", usage = "method for determining feature selection in individual trees")
    private TreeFeatureSelectorFactory.Type selection;

    /** file containing root features for ROOTED selection */
    @Option(name = "--rootFile", usage = "name of tab-delimited file (with headers) in model directory containing list of feature column names for root selection")
    private String rootFile;

    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Set the defaults.
        this.setAllDefaults();
        // Parse the command line.
        try {
            if (this.parseArgs(args)) {
                this.setupTraining();
                configureReader();
                // We made it this far, we can run the application.
                retVal = true;
            }
        } catch (IOException e) {
            System.err.println(e.toString());
        }
        return retVal;
    }

    /**
     * Prepare the training reader to read the training file.
     *
     * @throws IOException
     */
    private void configureReader() throws IOException {
        // Verify the model directory and read the labels.
        TabbedDataSetReader myReader = this.openReader(this.trainingFile, this.labelCol);
        // Configure the model for training.
        this.configureTraining(myReader);
    }

    @Override
    public void run() {
        try {
            // Convert the testing set to decision tree format.
            RandomForest.flattenDataSet(this.testingSet);
            // Now we read the training set.
            DataSet trainingSet = this.readTrainingSet();
            // Build the model from the training set.
            long start = System.currentTimeMillis();
            log.info("Creating selection factories");
            this.factoryIter = this.selection.create(this.hParms.getNumTrees(), this);
            log.info("Building the model.");
            this.model = new RandomForest(trainingSet, this.hParms, this.factoryIter);
            // Test the accuracy.
            log.info("Testing the model.");
            // Create a label array for output.
            INDArray predictions = this.model.predict(this.testingSet.getFeatures());
            String duration = DurationFormatUtils.formatDuration(System.currentTimeMillis() - start, "mm:ss");
            // Get the actual labels.
            INDArray expectations = this.testingSet.getLabels();
            Evaluation accuracy = new Evaluation(this.getLabels());
            accuracy.eval(expectations, predictions);
            log.info("Writing the report.");
            TextStringBuilder reportBuilder = new TextStringBuilder(800);
            // Describe the model.
            reportBuilder.appendNewLine();
            reportBuilder.appendln(
                            "Model file is %s%n" +
                            "=========================== Parameters ===========================%n" +
                            "     maxFeatures = %12d, minSplit      = %12d%n" +
                            "     nEstimators = %12d, maxDepth      = %12d%n" +
                            "     sampleSize  = %12d, batchSize     = %12d%n" +
                            "     --------------------------------------------------------%n" +
                            "     Randomization strategy is %s with seed %d.%n" +
                            "     %s minutes to train model.",
                           this.modelName.getCanonicalPath(), this.maxFeatures, this.minSplit,
                           this.nEstimators, this.maxDepth, this.sampleSize, this.batchSize,
                           this.method.toString(), this.seed, duration);
            this.produceAccuracyReport(reportBuilder, accuracy, predictions, expectations);
            this.bestRating = accuracy.accuracy();
            reportBuilder.appendNewLine();
            INDArray impactArray = this.model.computeImpact();
            this.impact = this.computeImpactList(impactArray);
            // Output the 10 most impactful columns.
            reportBuilder.appendln("Impact       Column Name");
            reportBuilder.appendln("---------------------------------------------------------");
            Iterator<Rating<String>> iter = this.impact.iterator();
            int count = 0;
            while (iter.hasNext() && count < 20) {
                Rating<String> item = iter.next();
                reportBuilder.appendln("%12.4f %s", item.getRating(), item.getKey());
                count++;
            }
            reportBuilder.appendNewLine();
            String report = reportBuilder.toString();
            log.info(report);
            RunLog.writeTrialReport(this.getTrialFile(), this.comment, report);
            this.resultReport = this.comment + report;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

    }

    /**
     * @return a sorted list of the impact values for all the columns
     *
     * @param impact	impact vector, indexed by input column
     */
    private SortedSet<Rating<String>> computeImpactList(INDArray impact) {
        List<String> colNames = getColNames();
        // Now we fill in the output set.
        SortedSet<Rating<String>> retVal = new TreeSet<Rating<String>>();
        for (int i = 0; i < impact.columns(); i++)
            retVal.add(new Rating<String>(colNames.get(i), impact.getDouble(i)));
        return retVal;
    }

    /**
     * @return the list of column names (in order) for this model's input columns
     */
    public List<String> getColNames() {
        // The tricky part here is we need to compute the column names.  We go through the feature
        // columns one by one.  If we are in channel mode, each channel element gets a number suffix.
        List<String> featureNames = this.reader.getFeatureNames();
        int channels = this.getChannelCount();
        List<String> colNames = new ArrayList<String>(featureNames.size() * channels);
        for (String col : this.reader.getFeatureNames()) {
            if (! this.isChannelMode())
                colNames.add(col);
            else
                IntStream.rangeClosed(1, channels).mapToObj(i -> String.format("%s|%02d", col, i)).forEach(x -> colNames.add(x));
        }
        return colNames;
    }

    /**
     * Compute the ANOVA F-measure for each input column.  This is an indicator of the degree to which each input
     * column's values sort according to the classification.  We do this for both the training and testing sets.
     *
     * @param modelDir		model directory
     *
     * @return a sorted set of ratings for the input columns
     *
     * @throws IOException
     * @throws ParseFailureException
     */
    public SortedSet<Rating<String>> computeAnovaFValues(File modelDir) throws IOException, ParseFailureException {
        SortedSet<Rating<String>> retVal = new TreeSet<Rating<String>>();
        if (! this.initializeForPredictions(modelDir))
            throw new ParseFailureException("Invalid parameter file in " + modelDir + ".");
        else {
            OneWayAnova computer = new OneWayAnova();
            // Get the nubmer of labels.
            int nLabels = this.getLabels().size();
            // Set up the training file reader.
            this.configureReader();
            // Get the column names.
            List<String> colNames = this.getColNames();
            // We need to merge the training set with the testing set.  Note the order doesn't matter.
            DataSet fullSet = this.readTrainingSet();
            List<DataSet> rows = fullSet.asList();
            rows.addAll(this.testingSet.asList());
            // Now separate the dataset rows by classification.
            List<List<DataSet>> groups = IntStream.range(0, nLabels).mapToObj(i -> new ArrayList<DataSet>(rows.size()))
                    .collect(Collectors.toList());
            for (DataSet row : rows) {
                int labelIdx = DecisionTree.bestLabel(row);
                groups.get(labelIdx).add(row);
            }
            // Build a double[] for each classification.  We use this to build the anova input data.
            List<double[]> arrays = IntStream.range(0, nLabels).mapToObj(j -> new double[groups.get(j).size()])
                    .collect(Collectors.toList());
            // For each column, compute the Anova.
            for (int i = 0; i < colNames.size(); i++) {
                for (int j = 0; j < nLabels; j++) {
                    List<DataSet> group = groups.get(j);
                    double[] array = arrays.get(j);
                    for (int k = 0; k < group.size(); k++) {
                        INDArray features = group.get(k).getFeatures();
                        array[k] = features.getDouble(i);
                    }
                }
                // Now we have a double[] for each class.  From this we get the F-measure.
                double rating = computer.anovaFValue(arrays);
                retVal.add(new Rating<String>(colNames.get(i), rating));
            }
        }
        return retVal;
    }

    /**
     * @return the full training set for this model
     */
    private DataSet readTrainingSet() {
        log.info("Reading training set.");
        List<DataSet> batches = new ArrayList<DataSet>();
        for (DataSet batch : this.reader) {
            RandomForest.flattenDataSet(batch);
            batches.add(batch);
        }
        DataSet retVal = DataSet.merge(batches);
        log.info("{} records read from training set.", retVal.numExamples());
        return retVal;
    }

    @Override
    public DistributedOutputStream getDistributor() {
        return new DistributedOutputStream.Discrete();
    }

    @Override
    public IValidationReport getValidationReporter(OutputStream outStream) {
        return new ClassValidationReport(outStream);
    }

    @Override
    public List<String> getLabelCols() {
        return Collections.singletonList(this.labelCol);
    }

    @Override
    public void setAllDefaults() {
        RandomForest.Parms hyperParms = new RandomForest.Parms();
        this.copyHyperParmsToOptions(hyperParms);
        this.help = false;
        this.testSize = 2000;
        this.seed = (int) (System.currentTimeMillis() & 0xFFFFF);
        this.setChannelMode(false);
        this.metaCols = "";
        this.channelCount = 1;
        this.modelName = null;
        this.comment = null;
        this.idCol = null;
        this.batchSize = 100;
        this.selection = TreeFeatureSelectorFactory.Type.NORMAL;
        this.rootFile = "roots.tbl";
        // Clear the rating value.
        this.bestRating = 0.0;
    }

    @Override
    public List<String> computeAvailableHeaders(List<String> headers, Collection<String> labels) {
        // Every header is available for a classification model.  One will be selected as the label column.
        return headers;
    }

    @Override
    public void setMetaCols(String[] cols) {
        if (cols.length > 1)
            this.metaCols = StringUtils.join(cols, ',', 0, cols.length - 1);
        else
            this.metaCols = "";
        if (cols.length > 0)
            this.labelCol = cols[cols.length - 1];
        else
            this.labelCol = "1";
    }

    @Override
   public void writeParms(File outFile) throws IOException {
       PrintWriter writer = new PrintWriter(outFile);
       writer.format("--col %s\t# input column for class name%n", this.labelCol);
       writeBaseModelParms(writer);
       String typeList = Stream.of(RandomForest.Method.values()).map(RandomForest.Method::name).collect(Collectors.joining(", "));
       writer.format("# Valid randomization methods are %s.%n", typeList);
       writer.format("--method %s\t# randomization method for picking subsamples%n", this.method);
       writer.format("--maxFeatures %d\t# number of features to check at each tree node%n", this.maxFeatures);
       writer.format("--minSplit %d\t# minimum number of examples allowed in a leaf node%n", this.minSplit);
       writer.format("--nEstimators %d\t# number of trees in the forest%n", this.nEstimators);
       writer.format("--maxDepth %d\t# maximum allowable tree depth%n", this.maxDepth);
       writer.format("--sampleSize %d\t# number of examples to use in each subsample%n", this.sampleSize);
       writer.format("--batchSize %d\t# size of each input batch%n", this.batchSize);
       typeList = Stream.of(TreeFeatureSelectorFactory.Type.values()).map(TreeFeatureSelectorFactory.Type::name).collect(Collectors.joining(", "));
       writer.format("# Valid split point finder methods are %s.%n", typeList);
       writer.format("--selection %s\t# selection method for features in trees%n", this.selection.toString());
       writer.format("--rootFile %s\t# file (tab-delimited with headers, column 1) containing root features for ROOTED selection%n", this.rootFile);
       writer.close();
   }

    @Override
    public TabbedDataSetReader openReader(List<String> strings) throws IOException {
        return this.openReader(strings, this.labelCol);
    }

    @Override
    public TabbedDataSetReader openReader(File inFile) throws IOException {
        return this.openReader(inFile, this.labelCol);
    }

    @Override
    public void configureTraining(TabbedDataSetReader myReader) throws IOException {
        this.initializeReader(myReader);
        this.initializeTraining();
    }

    @Override
    public TestValidationReport getTestReporter() {
        return new ClassTestValidationReport();
    }

    @Override
    public IPredictError testBestPredictions(List<String> mainFile, IValidationReport testErrorReport)
            throws IOException {
        // Get access to the input data.
        TabbedDataSetReader batches = this.openReader(mainFile, this.labelCol);
        // Set the batch size.
        batches.setBatchSize(this.batchSize);
        // Initialize the error predictor.
        IPredictError retVal = new ClassPredictError(this.getLabels());
        // Initialize the output report.
        testErrorReport.startReport(this.getMetaList(), this.getLabels());
        // Loop through the data, making predictions.
        for (DataSet batch : batches) {
            INDArray features = RandomForest.flattenFeatures(batch.getFeatures());
            INDArray output = this.model.predict(features);
            retVal.accumulate(batch.getLabels(), output);
            testErrorReport.reportOutput(batch.getExampleMetaData(String.class), batch.getLabels(), output);
        }
        // Finish predicting.
        retVal.finish();
        testErrorReport.finishReport(retVal);
        // Close the report output stream.
        testErrorReport.close();
        // Return the evaluation of the predictions.
        return retVal;
    }

    @Override
    public String getResultReport() {
        return this.resultReport;
    }

    @Override
    public void saveModelForced() throws IOException {
        this.model.save(this.modelName);
        // Also, store the full impact report in the output directory.
        File impactFile = new File(this.modelDir, "impact.tbl");
        try (PrintWriter impactStream = new PrintWriter(impactFile)) {
            impactStream.println("col_name\tinfo_gain");
            for (Rating<String> colItem : this.impact)
                impactStream.format("%s\t%14.6f%n", colItem.getKey(), colItem.getRating());
        }

    }

    @Override
    public void setupTraining() throws IOException {
        this.setupTraining(this.labelCol);
    }

    @Override
    protected void testModelPredictions(IValidationReport reporter, Shuffler<String> inputData) throws IOException {
        this.model = RandomForest.load(this.modelName);
        this.testBestPredictions(inputData, reporter);
    }

    @Override
    protected void initializeModelParameters() throws FileNotFoundException, IOException {
        // Copy the parameters to the hyper-parameters.
        this.hParms = new RandomForest.Parms();
        this.hParms.setMethod(this.method);
        this.hParms.setLeafLimit(this.minSplit);
        this.hParms.setMaxDepth(this.maxDepth);
        this.hParms.setNumExamples(this.sampleSize);
        this.hParms.setNumFeatures(this.maxFeatures);
        this.hParms.setNumTrees(this.nEstimators);
        // Set the randomizer seed.
        RandomForest.setSeed(seed);
        // Initialize the low-level computed parameters.
        initializeBaseModelParms();
        // If the user asked for a configuration file, write it here.
        if (this.parmFile != null) writeParms(this.parmFile);
    }

    @Override
    public void setSizeParms(int inputSize, int featureCols) {
        this.testSize = inputSize / 10;
        if (testSize < 1) testSize = 1;
        int nExamples = inputSize - testSize;
        RandomForest.Parms hyperParms = new RandomForest.Parms(nExamples, featureCols);
        copyHyperParmsToOptions(hyperParms);
    }

    /**
     * Copy hyper-parameters to this object's command-line options.
     *
     * @param hyperParms	hyper-parameters to copy
     */
    protected void copyHyperParmsToOptions(RandomForest.Parms hyperParms) {
        this.method = hyperParms.getMethod();
        this.maxDepth = hyperParms.getMaxDepth();
        this.nEstimators = hyperParms.getNumTrees();
        this.minSplit = hyperParms.getLeafLimit();
        this.maxFeatures = hyperParms.getNumFeatures();
        this.sampleSize = hyperParms.getNumExamples();
    }

    /**
     * @return the tuning parameters for this random forest
     */
    public RandomForest.Parms getParms() {
        return this.hParms;
    }

    /**
     * @return the random number seed for this process
     */
    public int getSeed() {
        return this.seed;
    }

    /**
     * @return the number of input columns
     */
    public int getWidth() {
        return this.reader.getWidth() * this.getChannelCount();
    }

    /**
     * @return the feature column names listed in the root file
     */
    public List<String> getImpactCols() throws IOException {
        File impactFile = new File(this.modelDir, this.rootFile);
        List<String> retVal = TabbedLineReader.readColumn(impactFile, "1");
        return retVal;
    }

}
