/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.List;

import org.theseed.basic.ICommand;
import org.theseed.dl4j.DistributedOutputStream;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.io.TabbedLineReader;
import org.theseed.reports.IValidationReport;
import org.theseed.reports.TestValidationReport;
import org.theseed.utils.Parms;

/**
 * This is an interface that describes the methods used by machine learning tools managed by dl4j.jfx.
 *
 * @author Bruce Parrello
 */
public interface ITrainingProcessor extends ICommand {

    /**
     * @return the model directory
     */
    public File getModelDir();

    /**
     * @return the labels
     */
    public List<String> getLabels();

    /**
     * Run predictions and output to a specific reporter.
     *
     * @param reporter	validation reporter
     * @param inFile	input file
     *
     * @throws IOException
     */
    public void runPredictions(IValidationReport reporter, File inFile) throws IOException;

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
    public boolean setupParameters(Parms parms, File modelDirectory) throws IOException;

    /**
     * @return a distributed output stream for this processor
     */
    public DistributedOutputStream getDistributor();

    /**
     * @return an appropriate validation reporter for this processor
     *
     * @param outStream		output stream for the report
     */
    public IValidationReport getValidationReporter(OutputStream outStream);

    /**
     * @return the relevant label column names for this model
     */
    public List<String> getLabelCols();

    /**
     * Set the default values for the parameters.
     */
    public void setAllDefaults();

    /**
     * Compute the size-dependent default parameters
     *
     * @param inputSize		number of rows in input training file
     * @param featureCols	number of input columns
     */
    public void setSizeParms(int inputSize, int featureCols);

    /**
     * @return the list of headers available for use as meta-columns
     *
     * @param headers	full list of column headers
     * @param labels	list of labels for this model
     */
    public List<String> computeAvailableHeaders(List<String> headers, Collection<String> labels);

    /**
     * Specify the meta-column information.
     *
     * @param cols		array of metadata column names; the first is the sample ID, the last is the label (if any)
     */
    public void setMetaCols(String[] cols);

    /**
     * Specify the ID column.
     *
     * @param idCol		new ID column
     */
    public void setIdCol(String idCol);

    /** Write all the parameters to a configuration file.
    *
    * @param outFile	file to be created for future use as a configuration file
    *
    * @throws IOException */
    public void writeParms(File outFile) throws IOException;

    /**
     * Specify a new progress monitor.
     *
     * @param monitor	new ITrainReporter instance
     */
    public void setProgressMonitor(ITrainReporter monitor);

    /**
     * Initialize this model processor for a prediction run.
     *
     * @param modelDir	target model directory
     *
     * @return TRUE if successful, FALSE if the parameters were invalid
     *
     * @throws IOException
     */
    public boolean initializeForPredictions(File modelDir) throws IOException;

    /**
     * Determine whether or not this model uses channel input.
     */
    public void checkChannelMode();

    /**
     * Set the model directory.
     *
     * @param modelDir	proposed model directory
     */
    public void setModelDir(File modelDir);

    /**
     * @return a data set reader for the specified file
     *
     * @param inFile	file to use for training and testing
     *
     * @throws IOException
     */
    public TabbedDataSetReader openReader(File inFile) throws IOException;

    /**
     * @return a data set reader for the specified string list
     *
     * @param strings	string list to use for training and testing
     *
     * @throws IOException
     */
    public abstract TabbedDataSetReader openReader(List<String> strings) throws IOException;

    /**
     * Denote that the model should not be saved to disk.
     */
    public void setSearchMode();

    /**
     * Configure the model for training.  This includes parsing the header of the training/testing file,
     * reading the testing set, and initializing the model parameters.
     *
     * @param	reader for the testing/training data
     *
     * @throws IOException
     */
    public void configureTraining(TabbedDataSetReader myReader) throws IOException;

    /**
     * @return a testing-set error calculator for this processor
     */
    public TestValidationReport getTestReporter();

    /**
     * @return the IDs from the training data
     *
     * @param reader	reader containing the training and testing data
     *
     * @throws IOException
     */
    public List<String> getTrainingMeta(TabbedLineReader reader) throws IOException;

    /**
     * @return the prediction error from the best model found during a validation or search
     *
     * @param mainFile			list of testing-set records
     * @param testErrorReport	reporting facility for validation
     */
    public IPredictError testBestPredictions(List<String> mainFile, IValidationReport testErrorReport) throws IOException;

    /**
     * Save the IDs from the input training data to the trained.tbl file, one per line.
     *
     * @param trainList		list of strings containing the training data.
     *
     * @throws IOException
     */
    public void saveTrainingMeta(List<String> trainList) throws IOException;

    /**
     * @return the result report
     */
    public String getResultReport();

    /**
     * @return the trial file name
     */
    public File getTrialFile();

    /**
     * @return the ID column
     */
    public String getIdCol();

    /**
     * @return the list of meta-data column names
     */
    public List<String> getMetaList();

    /**
     * Specify the comment string for reports.
     *
     * @param comment 	the comment to set
     */
    public void setComment(String comment);

    /**
     * Force saving of the model.
     */
    public void saveModelForced() throws IOException;

    /**
     * @return the model's performance rating (higher is better)
     */
    public double getRating();

    /**
     * Save the IDs from the input training file to the trained.tbl file, one per line.
     *
     * @throws IOException
     */
    public void saveTrainingMeta() throws IOException;

    /**
     * @return the label column index
     *
     * @param parmFile	parameter file
     *
     * @throws IOException
     */
    public int getLabelIndex(File parmFile) throws IOException;

}
