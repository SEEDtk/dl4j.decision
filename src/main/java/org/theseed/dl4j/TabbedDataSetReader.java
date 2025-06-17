/**
 *
 */
package org.theseed.dl4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;
import org.theseed.io.TabbedLineReader;

/**
 * This class reads a dataset from a tab-delimited file.  A dataset consists of features (inputs),
 * labels (classifications), and metadata.  In a training set the metadata is ignored.  In a
 * prediction set the labels are ignored.
 *
 * The primary processing function is to output a dataset representing a batch of input.  This
 * dataset can then be used to train or test a model.  The default batch size is 100.  This can
 * be modified by the client.
 *
 * @author Bruce Parrello
 *
 */
public class TabbedDataSetReader implements Iterable<DataSet>, Iterator<DataSet>, AutoCloseable {

    // FIELDS
    /** input tabbed file */
    private TabbedLineReader reader;
    /** list of valid labels */
    private ArrayList<String> labels;
    /** map of label columns for regression labels; array index is column index, value is label column */
    private int[] labelMap;
    /** column index of label column */
    private int labelIdx;
    /** normalizer to be applied to all batches of input */
    private DataNormalization normalizer;
    /** current batch size */
    private int batchSize;
    /** buffer array for holding input */
    private ArrayList<Entry> buffer;
    /** array of meta-column indices, or -1 if the column is not meta-data */
    private int[] metaColFlag;
    /** number of inputs */
    private int width;
    /** number of meta-data columns */
    private int metaWidth;

    /** null array index */
    private static final int ANULL = -1;

    /** This is a simple class for holding a feature, its metadata, and its label. */
    protected class Entry {
        String[] feature;
        String[] metaData;
        INDArray labelVals;

        /** Create a blank entry. */
        public Entry() {
            this.feature = new String[getWidth()];
            this.labelVals = Nd4j.zeros(TabbedDataSetReader.this.labels.size());
            this.metaData = new String[getMetaWidth()];
        }

    }

    /**
     * Construct a training/testing dataset reader for a file (with metadata).
     *
     * @param file		the file containing the training set, or NULL to use the standard input
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param metaCols	a list of metadata column names and/or indexes; these columns are ignored
     *
     * @throws IOException
     */
    public TabbedDataSetReader(File file, String labelCol, List<String> labels, List<String> metaCols) throws IOException {
        this.reader = (file == null ? new TabbedLineReader(System.in) : new TabbedLineReader(file));
        this.setup(labelCol, labels, metaCols);
    }

    /**
     * Construct a training/testing dataset reader for an input stream (with metadata).
     *
     * @param stream	the stream containing the training set
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param metaCols	a list of metadata column names and/or indexes; these columns are ignored
     *
     * @throws IOException
     */
    public TabbedDataSetReader(InputStream stream, String labelCol, List<String> labels, List<String> metaCols) throws IOException {
        this.reader = new TabbedLineReader(stream);
        this.setup(labelCol, labels, metaCols);
    }

    /**
     * Construct a training/testing dataset reader for a file (without metadata).
     *
     * @param file		the file containing the training set, or NULL to use the standard input
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     *
     * @throws IOException
     */
    public TabbedDataSetReader(File file, String labelCol, List<String> labels) throws IOException {
        this.reader = (file == null ? new TabbedLineReader(System.in) : new TabbedLineReader(file));
        this.setup(labelCol, labels, Collections.emptyList());
    }

    /**
     * Construct a training/testing dataset reader for an input stream (without metadata).
     *
     * @param stream	the stream containing the training set
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     *
     * @throws IOException
     */
    public TabbedDataSetReader(InputStream stream, String labelCol, List<String> labels) throws IOException {
        this.reader = new TabbedLineReader(stream);
        this.setup(labelCol, labels, Collections.emptyList());
    }

    /**
     * Construct a prediction dataset reader for a file (with metadata).
     *
     * @param file		the file containing the training set, or NULL to use the standard input
     * @param metaCols	a list of metadata column names and/or indexes
     *
     * @throws IOException
     */
    public TabbedDataSetReader(File file, List<String> metaCols) throws IOException {
        this.reader = (file == null ? new TabbedLineReader(System.in) : new TabbedLineReader(file));
        this.setup(null, Collections.emptyList(), metaCols);
    }

    /**
     * Construct a prediction reader for an input stream (with metadata).
     *
     * @param stream	the stream containing the training set
     * @param metaCols	a list of metadata column names and/or indexes
     *
     * @throws IOException
     */
    public TabbedDataSetReader(InputStream stream, List<String> metaCols) throws IOException {
        this.reader = new TabbedLineReader(stream);
        this.setup(null, Collections.emptyList(), metaCols);
    }

    /**
     * Construct a training/testing dataset reader for a string list.
     *
     * @param strings	the list of strings containing the headers and the data rows
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param metaCols	a list of metadata column names and/or indexes; these columns are ignored
     *
     * @throws IOException
     */
    public TabbedDataSetReader(List<String> strings, String labelCol, List<String> labels, List<String> metaCols) throws IOException {
        this.reader = new TabbedLineReader(strings);
        this.setup(labelCol, labels, metaCols);
    }

    /**
     * Initialize the fields of this object.
     *
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param metaCols	a list of metadata column names and/or indexes; these columns are ignored
      *
     * @throws IOException
     */
    protected void setup(String labelCol, List<String> labels, List<String> metaCols) throws IOException {
        // Save the label array.
        this.labels = new ArrayList<String>(labels);
        // Find the field where we expect the labels to be.
        this.labelIdx = (labelCol != null ? this.reader.findField(labelCol) : ANULL);
        // Denote we are not normalizing.
        this.normalizer = null;
        // Denote everything is an input.
        this.width = this.reader.size();
        this.metaWidth = 0;
        // Create the label map array.
        this.labelMap = new int[this.width];
        Arrays.fill(this.labelMap, ANULL);
        // Account for the label.
        if (this.labelIdx != ANULL) this.width--;
        // Set up the metadata columns.  Each one subtracts from the input width.
        this.metaColFlag = new int[this.reader.size()];
        Arrays.fill(this.metaColFlag, ANULL);
        for (int i = 0; i < metaCols.size(); i++) {
            String metaCol = metaCols.get(i);
            int colIdx = this.reader.findField(metaCol);
            this.metaColFlag[colIdx] = i;
            this.width--;
            this.metaWidth++;
        }
        // Initialize the batch size.
        this.setBatchSize(100);
    }

    /**
     * Set up this input stream for regression input.  Columns having the same name as the labels will be identified
     * and used as output value columns.
     *
     * @throws IOException
     */
    public void setRegressionColumns() throws IOException {
        for (int i = 0; i < this.labels.size(); i++) {
            String labelCol = this.labels.get(i);
            int colIdx = this.reader.findField(labelCol);
            this.labelMap[colIdx] = i;
            this.width--;
        }
    }

    /**
     * @return a list of the names for each feature column
     */
    public List<String> getFeatureNames() {
        List<String> retVal = new ArrayList<String>(this.width);
        String[] names = this.reader.getLabels();
        for (int i = 0; i < this.reader.size(); i++) {
            if (i != this.labelIdx && this.labelMap[i] < 0 && this.metaColFlag[i] == ANULL)
                retVal.add(names[i]);
        }
        return retVal;
    }

    @Override
    public Iterator<DataSet> iterator() {
        return this;
    }

    /**
     * Return TRUE if there is more data in this file, else FALSE.
     */
    @Override
    public boolean hasNext() {
        return this.reader.hasNext();
    }

    /**
     * Read all the batches into a single dataset.
     */
    public DataSet readAll() {
        int saved = this.batchSize;
        this.batchSize = Integer.MAX_VALUE;
        DataSet retVal = this.next();
        this.batchSize = saved;
        return retVal;
    }

    /**
     * Return the next batch of data.
     */
    @Override
    public DataSet next() {
        // Get the number of fields in each record.
        int n = this.reader.size();
        // Remember if we have metadata and/or labels.
        boolean haveMeta = (this.getMetaWidth() > 0);
        boolean haveLabels = false;
        // The array list should be empty.  Fill it from the input.
        for (int numRead = 0; numRead < this.batchSize && this.hasNext(); numRead++) {
            TabbedLineReader.Line line = this.reader.next();
            Entry record = new Entry();
            int pos = 0;
            for (int i = 0; i < n; i++) {
                if (i == this.labelIdx) {
                    // We have a class label.  Translate from a string to a number.
                    String labelName = line.get(i);
                    int label = this.labels.indexOf(labelName);
                    if (label < 0) {
                        throw new IllegalArgumentException("Invalid label " + labelName);
                    } else {
                        record.labelVals.putScalar(label, 1.0);
                    }
                    haveLabels = true;
                } else if (this.labelMap[i] >= 0) {
                    double value = line.getDouble(i);
                    record.labelVals.putScalar(this.labelMap[i], value);
                    haveLabels = true;
                } else if (this.metaColFlag[i] != ANULL) {
                    // Here we have a metadata column.
                    record.metaData[this.metaColFlag[i]] = line.get(i);
                } else {
                    // Here we have a feature column.
                    record.feature[pos++] = line.get(i);
                }
            }
            this.getBuffer().add(record);
        }
        // Create and fill the feature and label arrays.
        INDArray features = createFeatureArray();
        INDArray labels = Nd4j.zeros(this.getBuffer().size(), this.labels.size());
        ArrayList<String> metaData = (haveMeta ? new ArrayList<String>(this.getBuffer().size()) : null);
        int row = 0;
        for (Entry record : this.getBuffer()) {
            storeFeature(features, row, record);
            if (haveLabels) {
                labels.putRow(row, record.labelVals);
            }
            if (metaData != null) metaData.add(String.join("\t", record.metaData));
            row++;
        }
        this.getBuffer().clear();
        // Build the dataset.
        DataSet retVal = new DataSet();
        retVal.setFeatures(features);
        if (haveLabels) retVal.setLabels(labels);
        if (haveMeta) retVal.setExampleMetaData(metaData);
        if (this.normalizer != null)
            this.normalizer.transform(retVal);
        return retVal;
    }

    /**
     * This is the method for formatting features into an example row.  Each input column
     * is converted to a vector of floating-point numbers and stored in the specified row of the
     * feature matrix.
     *
     * @param features	feature matrix
     * @param row		target row index
     * @param record	record containing the input strings
     *
     * @return an array representing a single feature
     */
    protected void storeFeature(INDArray features, int row, Entry record) {
        // Create the output array:  one row for each channel (depth), one column for each input column, and a unit height in the middle.
        int[] indices = new int[] { row, 0, 0, 0 };
        // Loop through the indices, copying values.
        for (int i = 0; i < this.getWidth(); i++) {
            indices[3] = i;
            double[] vector = this.stringToVector(record.feature[i]);
            for (int j = 0; j < this.getChannels(); j++) {
                indices[1] = j;
                features.putScalar(indices, vector[j]);
            }
        }
    }

    /**
     * Convert a column string to a floating-point vector to be put into the output feature.
     *
     * @param string	input string
     *
     * @return	a vector containing the feature data; the default is a vector of one element
     */
    protected double[] stringToVector(String string) {
        double[] retVal = new double[1];
        retVal[0] = Double.valueOf(string);
        return retVal;
    }

    /**
     * @return the number of channels for this dataset; the default is 1
     */
    protected int getChannels() {
        return 1;
    }

    /**
     * This is the default method for pre-allocating the feature array.  It is four-dimensional,
     * one column per input column, one row per input record, and unit dimensions for the height
     * and channel size.
     *
     * @return an empty feature array for the output
     */
    protected INDArray createFeatureArray() {
        int[] shape = new int[] { this.getBuffer().size(), 1, 1,
                this.getWidth() };
        return Nd4j.createUninitialized(shape);
    }

    /**
     * @return the normalizer
     */
    public DataNormalization getNormalizer() {
        return this.normalizer;
    }

    /**
     * @return the batch size
     */
    public int getBatchSize() {
        return this.batchSize;
    }

    /**
     * @param normalizer 	normalizer to apply to each incoming set
     */
    public TabbedDataSetReader setNormalizer(DataNormalization normalizer) {
        this.normalizer = normalizer;
        return this;
    }

    /**
     * @param batchSize 	new batch size for reading
     */
    public TabbedDataSetReader setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        this.buffer = new ArrayList<Entry>(batchSize);
        return this;
    }

    /**
     * @return the number of sensor values per feature
     */
    public int getWidth() {
        return this.width;
    }

    /**
     * @return the number of metadata values per feature;
     */
    public int getMetaWidth() {
        return this.metaWidth;
    }


    /**
     * @return the list of labels read from the specified label file
     *
     * @param labelFile		file containing the labels
     *
     * @throws FileNotFoundException
     */
    public static List<String> readLabels(File labelFile) throws FileNotFoundException {
        Scanner labelsIn = new Scanner(labelFile);
        List<String> retVal = new ArrayList<String>();
        while (labelsIn.hasNext()) {
            retVal.add(labelsIn.nextLine());
        }
        labelsIn.close();
        return retVal;
    }

    /**
     * @return the buffer
     */
    public ArrayList<Entry> getBuffer() {
        return buffer;
    }

    /**
     * Close this dataset reader.
     */
    @Override
    public void close() {
        this.reader.close();
    }

}
