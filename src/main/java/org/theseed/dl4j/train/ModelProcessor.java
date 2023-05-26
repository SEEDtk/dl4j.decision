/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.text.TextStringBuilder;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.evaluation.classification.ConfusionMatrix;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.counters.Shuffler;
import org.theseed.dl4j.ChannelDataSetReader;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.io.LineReader;
import org.theseed.io.TabbedLineReader;
import org.theseed.reports.IValidationReport;
import org.theseed.reports.NullTrainReporter;
import org.theseed.utils.Parms;

/**
 * @author Bruce Parrello
 *
 */
public abstract class ModelProcessor {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ModelProcessor.class);
    /** input file reader */
    protected TabbedDataSetReader reader;
    /** array of labels */
    private List<String> labels;
    /** label column specification (NULL if none) */
    private String labelSpec;
    /** testing set */
    protected DataSet testingSet;
    /** number of input channels */
    protected int channelCount;
    /** TRUE if we have channel input */
    private boolean channelMode;
    /** best accuracy */
    private double bestRating;
    /** list of metadata column names */
    protected List<String> metaList;
    /** progress monitor */
    private ITrainReporter progressMonitor;
    /** TRUE if models should be saved, else FALSE */
    protected boolean production;

    // COMMAND-LINE OPTIONS

    /** help option */
    @Option(name = "-h", aliases = { "--help" }, help = true)
    protected boolean help;
    /** size of testing set */
    @Option(name = "-t", aliases = { "--testSize" }, metaVar = "1000", usage = "size of the testing set")
    protected int testSize;
    /** initialization seed */
    @Option(name = "-s", aliases = { "--seed" }, metaVar = "12765", usage = "random number seed")
    protected int seed;
    /** comma-delimited list of metadata column names */
    @Option(name = "--meta", metaVar = "name,date", usage = "comma-delimited list of metadata columns")
    protected String metaCols;
    /** comment to display in trial log */
    @Option(name = "--comment", metaVar = "changed bias rate", usage = "comment to display in trial log")
    protected String comment;
    /** model file name */
    @Option(name = "--name", usage = "model file name (default is model.ser in model directory)")
    protected File modelName;
    /** input training set */
    @Option(name = "-i", aliases = { "--input" }, metaVar = "training.tbl", usage = "input training set file")
    protected File trainingFile;
    /** if specified, the ID column for identifying the rows used to train */
    @Option(name = "--id", metaVar = "row_id", usage = "ID column for input rows, specified if trained.tbl is to be written")
    protected String idCol;
    /** model directory */
    @Argument(index = 0, metaVar = "modelDir", usage = "model directory", required = true)
    protected File modelDir;

    public ModelProcessor() {
        this.production = true;
        this.progressMonitor = new NullTrainReporter();
    }

    /**
     * @return the progress monitor
     */
    protected ITrainReporter getProgressMonitor() {
        return this.progressMonitor;
    }

    /**
     * Specify a new progress monitor.
     *
     * @param monitor	new ITrainReporter instance
     */
    public void setProgressMonitor(ITrainReporter monitor) {
        this.progressMonitor = monitor;
    }

    /**
     * Show a progress message.
     *
     * @param msg	progress message to show
     */
    public void showProgressMessage(String msg) {
        if (this.progressMonitor != null)
            this.progressMonitor.showMessage(msg);
        log.info(msg);
    }

    /**
     * Denote that the model should not be saved to disk.
     */
    public void setSearchMode() {
        this.production = false;
    }

    /**
     * @return the rating of the best epoch
     */
    public double getRating() {
        return this.bestRating;
    }

    /**
     * Store a new best rating.
     */
    public void setRating(double newValue) {
        this.bestRating = newValue;
    }

    /**
     * Reset the rating indicator to show failure to achieve a result.
     */
    public void clearRating() {
        this.bestRating = Double.NEGATIVE_INFINITY;
    }

    /**
     * @return the labels
     */
    public List<String> getLabels() {
        return labels;
    }

    /**
     * @param labels the labels to set
     */
    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    /**
     * @return the channelMode
     */
    public boolean isChannelMode() {
        return channelMode;
    }

    /**
     * @param channelMode the channelMode to set
     */
    public void setChannelMode(boolean channelMode) {
        this.channelMode = channelMode;
    }

    /**
     * Initialize the reader for reading a training set.
     *
     * @param myReader  reader containing the training/testing data.
     *
     * @throws IOException
     */
    public void initializeReader(TabbedDataSetReader myReader) throws IOException {
        // Determine the input type and get the appropriate reader.
        this.checkChannelMode();
        this.reader = myReader;
    }

    /**
     * Set up the common input parameters.
     *
     * @throws FileNotFoundException
     * @throws IOException
     */
    protected abstract void initializeModelParameters() throws FileNotFoundException, IOException;

    /**
     * Determine whether or not this model uses channel input.
     */
    public void checkChannelMode() {
        File channelFile = new File(this.modelDir, "channels.tbl");
        this.channelMode = channelFile.exists();
    }

    /**
     * Initialize a reader for reading a training/testing set.
     *
     * @param inFile	file containing training/testing set
     * @param labelCol	column specifier for label column, or NULL if there is none
     *
     * @throws IOException
     */
    public TabbedDataSetReader openReader(File inFile, String labelCol) throws IOException {
        TabbedDataSetReader retVal;
        // Determine the input type and get the appropriate reader.
        if (! this.channelMode) {
            log.info("Normal input from {}.", inFile);
            // Normal situation.  Read scalar values.
            retVal = new TabbedDataSetReader(inFile, labelCol, this.getLabels(), this.metaList);
            this.channelCount = 1;
        } else {
            // Here we have channel input.
            File channelFile = new File(this.modelDir, "channels.tbl");
            Map<String, double[]> channelMap = ChannelDataSetReader.readChannelFile(channelFile);
            retVal = new ChannelDataSetReader(inFile, labelCol,
                    this.getLabels(), this.metaList, channelMap);
            this.channelCount = ((ChannelDataSetReader) retVal).getChannels();
            log.info("Channel input with {} channels from {}.", this.getChannelCount(), inFile);
        }
        return retVal;
    }

    /**
     * @return a reader for reading a training/testing set from a list of in-memory strings.
     *
     * @param strings		in-memory list of strings (including the header line)
     * @param labelCol		column specifier for label column, or NULL if there is none
     *
     * @throws IOException
     */
    public TabbedDataSetReader openReader(List<String> strings, String labelCol) throws IOException {
        TabbedDataSetReader retVal;
        // Determine the input type and get the appropriate reader.
        if (! this.channelMode) {
            // Normal situation.  Read scalar values.
            retVal = new TabbedDataSetReader(strings, labelCol, this.getLabels(), this.metaList);
            this.channelCount = 1;
        } else {
            // Here we have channel input.
            File channelFile = new File(this.modelDir, "channels.tbl");
            Map<String, double[]> channelMap = ChannelDataSetReader.readChannelFile(channelFile);
            retVal = new ChannelDataSetReader(strings, labelCol,
                    this.getLabels(), this.metaList, channelMap);
            this.channelCount = ((ChannelDataSetReader) retVal).getChannels();
        }
        return retVal;
    }

    /**
     * @return the channelCount
     */
    public int getChannelCount() {
        return channelCount;
    }

    /**
     * @return the trial file name
     */
    public File getTrialFile() {
        return new File(this.modelDir, "trials.log");
    }

    /**
     * Specify the comment string for reports.
     *
     * @param comment 	the comment to set
     */
    public void setComment(String comment) {
        this.comment = comment;
    }

    /**
     * @return the list of meta-data column names
     */
    public List<String> getMetaList() {
        return this.metaList;
    }

    /**
     * @return the ID column
     */
    public String getIdCol() {
        return this.idCol;
    }

    /**
     * Specify the ID column.
     *
     * @param idCol		new ID column
     */
    public void setIdCol(String idCol) {
        this.idCol = idCol;
    }

    /**
     * @return the model directory
     */
    public File getModelDir() {
        return this.modelDir;
    }

    /**
     * Specify a new testing size.
     *
     * @param size	proposed new testing size
     */
    public void setTestSize(int size) {
        this.testSize = size;
    }

    /**
     * Set the model directory.
     *
     * @param modelDir	proposed model directory
     */
    public void setModelDir(File modelDir) {
        this.modelDir = modelDir;
    }

    /**
     * Save the IDs from the input training file to the trained.tbl file, one per line.
     *
     * @throws IOException
     */
    public void saveTrainingMeta() throws IOException {
        try (TabbedLineReader reader = new TabbedLineReader(this.trainingFile)) {
            this.saveTrainingMeta(reader);
        }
    }

    /**
     * Save the IDs from the input training data to the trained.tbl file, one per line.
     *
     * @param trainList		list of strings containing the training data.
     *
     * @throws IOException
     */
    public void saveTrainingMeta(List<String> trainList) throws IOException {
        try (TabbedLineReader reader = new TabbedLineReader(trainList)) {
            this.saveTrainingMeta(reader);
        }
    }

    /**
     * @return the IDs from the training data
     *
     * @param reader	reader containing the training and testing data
     *
     * @throws IOException
     */
    public List<String> getTrainingMeta(TabbedLineReader reader) throws IOException {
        List<String> retVal = new ArrayList<String>(500);
        // Get the ID column.
        int idColIdx = reader.findField(this.getIdCol());
        // Skip over the testing set.
        Iterator<TabbedLineReader.Line> iter = reader.iterator();
        for (int i = 0; i < this.testSize; i++)
            iter.next();
        // Now accumulate the IDs for the training set.
        while (iter.hasNext()) {
            TabbedLineReader.Line line = iter.next();
            String id = line.get(idColIdx);
            retVal.add(id);
        }
        return retVal;
    }

    /**
     * Save the IDs from the training data to the trained.tbl file.
     *
     * @param reader	reader containing the training and testing data
     *
     * @throws IOException
     */
    protected void saveTrainingMeta(TabbedLineReader reader) throws IOException {
        List<String> trained = getTrainingMeta(reader);
        File outFile = new File(this.modelDir, "trained.tbl");
        try (PrintWriter writer = new PrintWriter(outFile)) {
            for (String id : trained)
                writer.println(id);
        }
    }

    /**
     * Configure this processor with the specified parameters.
     *
     * @param parms				configuration parameters
     * @param modelDirectory	directory containing the model
     *
     * @return TRUE if successful, FALSE if the parameters were invalid
     *
     * @throws IOException
     */
    public boolean setupParameters(Parms parms, File modelDirectory) throws IOException {
        String[] parmArray = new String[parms.size() + 1];
        setAllDefaults();
        // Process the parameters.
        List<String> parmValues = parms.get();
        parmValues.add(modelDirectory.toString());
        parmArray = parmValues.toArray(parmArray);
        boolean retVal = this.parseArgs(parmArray);
        if (retVal) {
            // Setup the training configuration.
            this.setupTraining();
        }
        return retVal;
    }

    /**
     * Setup the label and metadata columns.
     *
     * @throws IOException
     */
    public abstract void setupTraining() throws IOException;

    /**
     * Parse the parameters in the specified array.
     *
     * @param theseParms	array of parameters
     *
     * @return TRUE if successful, FALSE if the command should be aborted
     */
    public boolean parseArgs(String[] theseParms) {
        boolean retVal = false;
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(theseParms);
            if (this.help)
                parser.printUsage(System.err);
            else
                retVal = true;
            // Default the model name.
            if (this.modelName == null)
                this.modelName = new File(this.modelDir, "model.ser");
            // Default the input.
            if (this.trainingFile == null)
                this.trainingFile = new File(this.modelDir, "training.tbl");
        } catch (CmdLineException e) {
            System.err.println(e.toString());
            // For parameter errors, we display the command usage.
            parser.printUsage(System.err);
            this.getProgressMonitor().showResults("Error in parameters: " + e.toString());
        }
        return retVal;
    }

    /**
     * Set up the training file for processing.
     *
     * @param myReader	reader containing the data to use for training
     * @oaram labelCol	column specifier for the label column, or NULL if there is none
     *
     * @throws FileNotFoundException
     * @throws IOException
     */
    public void setupTraining(String labelCol) throws FileNotFoundException, IOException {
        if (! this.modelDir.isDirectory()) {
            throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
        } else {
            // Read in the labels from the label file.
            File labelFile = new File(this.modelDir, "labels.txt");
            if (! labelFile.exists())
                throw new FileNotFoundException("Label file not found in " + this.modelDir + ".");
            this.setLabels(TabbedDataSetReader.readLabels(labelFile));
            log.info("{} labels read from label file.", this.getLabels().size());
            // Save the label column specification.
            this.labelSpec = labelCol;
            // Parse the metadata column list.
            this.metaList = Arrays.asList(StringUtils.split(this.metaCols, ','));
            // Finally, we initialize the input to get the label and metadata columns handled.
            if (this.trainingFile == null) {
                this.trainingFile = new File(this.modelDir, "training.tbl");
                if (! this.trainingFile.exists())
                    throw new FileNotFoundException("Training file " + this.trainingFile + " not found.");
            }
        }
    }

    /**
     * Run predictions and output to a specific reporter.
     *
     * @param reporter	validation reporter
     * @param inFile	input file
     *
     * @throws IOException
     */
    public void runPredictions(IValidationReport reporter, File inFile) throws IOException {
        String idCol = this.getIdCol();
        if (idCol != null)
            reporter.setupIdCol(this.modelDir, idCol, this.getMetaList(), null);
        // Get the input data.
        Shuffler<String> inputData = new Shuffler<String>(1000);
        try (LineReader inStream = new LineReader(inFile)) {
            inputData.addSequence(inStream);
        }
        log.info("{} input data lines.", inputData.size() - 1);
        testModelPredictions(reporter, inputData);
    }

    /**
     * Run the current model's predictions against a sequence of input strings and output to a specific reporter.
     *
     * @param reporter		validation reporter
     * @param inputData		list of input strings
     */
    protected abstract void testModelPredictions(IValidationReport reporter, Shuffler<String> inputData) throws IOException;

    /**
     * Initialize this model processor for a prediction run.
     *
     * @param modelDir	target model directory
     *
     * @return TRUE if successful, FALSE if the parameters were invalid
     *
     * @throws IOException
     */
    public boolean initializeForPredictions(File modelDir) throws IOException {
        this.setModelDir(modelDir);
        File parmsPrm = new File(modelDir, "parms.prm");
        Parms parms = new Parms(parmsPrm);
        boolean retVal = this.setupParameters(parms, modelDir);
        if (retVal)
            this.checkChannelMode();
        return retVal;
    }

    /**
     * Set all the default parameters for this processor.
     */
    public abstract void setAllDefaults();

    /**
     * Write the basic parameters to the parm file.
     *
     * @param writer	writer to which the parameters should be output
     */
    public void writeBaseModelParms(PrintWriter writer) {
        String commentFlag = "";
        commentFlag = (this.metaCols.isEmpty() ? "# " : "");
        writer.format("%s--meta %s\t# comma-delimited list of meta-data columns%n", commentFlag, this.metaCols);
        String idName;
        if (this.getIdCol() == null) {
            String[] meta = StringUtils.split(this.metaCols, ',');
            if (meta.length == 0)
                idName = "id";
            else
                idName = meta[0];
            commentFlag = "# ";
        } else {
            idName = this.getIdCol();
            commentFlag = "";
        }
        writer.format("%s--id %s\t# ID column for validation reports%n", commentFlag, idName);
        writer.format("--testSize %d\t# size of the testing set, taken from the beginning of the file%n", this.testSize);
        writer.format("--seed %d\t# random number initialization seed%n", this.seed);
        String modelPath = (this.modelName == null ? "model.ser" : this.modelName.toString());
        writer.format("# --name %s\t# model file name%n", modelPath);
        String trainPath = (this.trainingFile == null ? "training.tbl" : this.trainingFile.toString());
        writer.format("# --input %s\t# training file name%n", trainPath);
        if (this.comment == null)
            writer.println("# --comment The comment appears in the trial log.");
        else
            writer.format("--comment %s%n", this.comment);
    }

    /**
     * Read in the testing set.
     */
    protected void readTestingSet() {
        // Get the testing set.
        log.info("Reading testing set (size = {}).", this.testSize);
        this.reader.setBatchSize(this.testSize);
        this.testingSet = this.reader.next();
        if (! this.reader.hasNext()) {
            log.warn("Training set contains only test data. Batch size = {} but {} records in input.",
                    this.testSize, this.testingSet.numExamples());
            throw new IllegalArgumentException("No training data.  Testing set size was " + this.testSize + " but only "
                    + this.testingSet.numExamples() + " records in input.");
        }
    }

    /**
     * Initialize for training.
     *
     * @throws IOException
     */
    protected void initializeTraining() throws IOException {
        // Read in the testing set.
        readTestingSet();
        // Set up the common parameters.
        initializeModelParameters();
    }

    /**
     * Update the base model parameters.
     *
     * @throws IOException
     */
    protected void initializeBaseModelParms() throws IOException {
        // Compute the model file name if it is defaulting.
        if (this.modelName == null)
            this.modelName = new File(this.modelDir, "model.ser");
        // Write out the comment.
        if (this.comment != null)
            log.info("*** {}", this.comment);
    }

    /**
     * Store an accuracy report in the specified text buffer.
     *
     * @param buffer		output text buffer
     * @param eval			classification evaluation object
     * @param output		predicted values
     * @param expect		expected values
     */
    public void produceAccuracyReport(TextStringBuilder buffer, Evaluation eval, INDArray output, INDArray expect) {
        // Output the evaluation.
        buffer.appendln(eval.stats());
        ConfusionMatrix<Integer> matrix = eval.getConfusion();
        // Analyze the negative results.
        int actualNegative = matrix.getActualTotal(0);
        if (actualNegative == 0) {
            buffer.appendln("No \"%s\" results were found.", this.getLabels().get(0));
        } else {
            double specificity = ((double) matrix.getCount(0, 0)) / actualNegative;
            buffer.appendln("Model specificity is %11.4f.%n", specificity);
        }
        // Write the header.
        buffer.appendln("%-11s %11s %11s %11s %11s %11s", "class", "accuracy", "sensitivity", "precision", "fallout", "MAE");
        buffer.appendln(StringUtils.repeat('-', 71));
        // The classification accuracy is 1 - (false negative + false positive) / total,
        // sensitivity is true positive / actual positive, precision is true positive / predicted positive,
        // and fall-out is false positive / actual negative.  The L1 Error is the trickiest.  It is the absolute
        // difference between the expected values and the output, divided by the number of examples.
        for (int i = 1; i < this.getLabels().size(); i++) {
            String label = this.getLabels().get(i);
            double accuracy = 1 - ((double) (matrix.getCount(0, i) + matrix.getCount(i,  0))) / this.testSize;
            double l1_error = 0.0;
            for (long r = 0; r < this.testSize; r++) {
                double diff = expect.getDouble(r, i) - output.getDouble(r, i);
                l1_error += Math.abs(diff);
            }
            String sensitivity = formatRatio(matrix.getCount(i, i), matrix.getActualTotal(i));
            String precision = formatRatio(matrix.getCount(i, i), matrix.getPredictedTotal(i));
            String fallout = formatRatio(matrix.getCount(0,  i), actualNegative);
            String l1Error = formatRatio(l1_error, this.testSize);
            buffer.appendln("%-11s %11.4f %11s %11s %11s %11s", label, accuracy, sensitivity, precision, fallout, l1Error);
        }
    }

    /**
     * Format a ratio for display in the evaluation metrics matrix.
     *
     * @param value		numerator
     * @param total		denominator
     *
     * @return a formatted fraction, or an empty string if the denominator is 0
     */
    protected static String formatRatio(double value, int total) {
        String retVal = "";
        if (total > 0) {
            retVal = String.format("%11.4f", value / total);
        }
        return retVal;
    }

    /**
     * Format a ratio for display in the evaluation metrics matrix.
     *
     * @param count		numerator
     * @param total		denominator
     *
     * @return a formatted fraction, or an empty string if the denominator is 0
     */
    protected static String formatRatio(int count, int total) {
        String retVal = "";
        if (total > 0) {
            retVal = String.format("%11.4f", ((double) count) / total);
        }
        return retVal;
    }

    /**
     * @return the label column index from the parameter file.
     *
     * @param modelDir	model directory
     *
     * @throws IOException
     */
    public int getLabelIndex(File modelDir) throws IOException {
        int retVal;
        // Set up the parameters without reading the testing set.
        this.initializeForPredictions(modelDir);
        // Get the label column name.
        String labelName = this.labelSpec;
        if (labelName == null)
            labelName = this.labels.get(0);
        // Read the training file headers to get the column index.
        try (TabbedLineReader reader = new TabbedLineReader(this.trainingFile)) {
            retVal = reader.findField(labelName);
        }
        return retVal;
    }

}
