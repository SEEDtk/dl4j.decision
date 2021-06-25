/**
 *
 */
package org.theseed.dl4j;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.theseed.io.TabbedLineReader;
import org.theseed.io.TabbedLineReader.Line;

/**
 * This reader creates a two-dimensional training or prediction set input from string labels.  The client must
 * specify a configuration file that contains each possibly input string and the vector of values to create
 * from it.  This should be a tab-delimited file with headers, but the label value MUST be in the first
 * column.  The headers are only used to fix the dimensions of the channel vectors.
 *
 * @author Bruce Parrello
 *
 */
public class ChannelDataSetReader extends TabbedDataSetReader {

    // FIELDS
    /** hash mapping each input value to its array of channel values */
    private Map<String, double[]> channelMap;
    /** number of channels */
    private int channels;

    /**
     * Construct a training/testing dataset reader for a file (with metadata).
     *
     * @param file		the file containing the training set
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param metaCols	a list of metadata column names and/or indexes; these columns are ignored
     * @param dict		mapping from input strings to output vectors
     * @throws IOException
     */
    public ChannelDataSetReader(File file, String labelCol, List<String> labels, List<String> metaCols,
            Map<String, double[]> dict)
            throws IOException {
        super(file, labelCol, labels, metaCols);
        setDictionary(dict);
    }

    /**
     * Construct a training/testing dataset reader for an input stream (with metadata).
     *
     * @param stream	the stream containing the training set
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param metaCols	a list of metadata column names and/or indexes; these columns are ignored
     * @param dict		mapping from input strings to output vectors
     *
     * @throws IOException
     */
    public ChannelDataSetReader(InputStream stream, String labelCol, List<String> labels, List<String> metaCols,
            Map<String, double[]> dict)
            throws IOException {
        super(stream, labelCol, labels, metaCols);
        setDictionary(dict);
    }

    /**
     * Construct a training/testing dataset reader for a file (without metadata).
     *
     * @param file		the file containing the training set
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param dict		mapping from input strings to output vectors
     *
     * @throws IOException
     */
    public ChannelDataSetReader(File file, String labelCol, List<String> labels,
            Map<String, double[]> dict)
            throws IOException {
        super(file, labelCol, labels);
        setDictionary(dict);
    }

    /**
     * Construct a training/testing dataset reader for an input stream (without metadata).
     *
     * @param stream	the stream containing the training set
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param dict		mapping from input strings to output vectors
     *
     * @throws IOException
     */
    public ChannelDataSetReader(InputStream stream, String labelCol, List<String> labels,
            Map<String, double[]> dict)
            throws IOException {
        super(stream, labelCol, labels);
        setDictionary(dict);
    }

    /**
     * Construct a prediction dataset reader for a file (with metadata).
     *
     * @param file		the file containing the prediction set
     * @param metaCols	a list of metadata column names and/or indexes
     * @param dict		mapping from input strings to output vectors
     *
     * @throws IOException
     */
    public ChannelDataSetReader(File file, List<String> metaCols,
            Map<String, double[]> dict)
            throws IOException {
        super(file, metaCols);
        setDictionary(dict);
    }

    /**
     * Construct a prediction dataset reader for an input stream (with metadata).
     *
     * @param stream	the stream containing the training set
     * @param metaCols	a list of metadata column names and/or indexes
     * @param dict		mapping from input strings to output vectors
     *
     * @throws IOException
     */
    public ChannelDataSetReader(InputStream stream, List<String> metaCols,
            Map<String, double[]> dict)
            throws IOException {
        super(stream, metaCols);
        setDictionary(dict);
    }

    /**
     * Construct a training/testing dataset reader for a list of strings.
     *
     * @param strings	the list of strings including the header and the data rows
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     * @param metaCols	a list of metadata column names and/or indexes; these columns are ignored
     * @param dict		mapping from input strings to output vectors
     * @throws IOException
     */
    public ChannelDataSetReader(List<String> strings, String labelCol, List<String> labels, List<String> metaCols,
            Map<String, double[]> dict)
            throws IOException {
        super(strings, labelCol, labels, metaCols);
        setDictionary(dict);
    }

    /**
     * Store the channel dictionary for this reader.
     *
     * @param dict	hash mapping input strings to channel vectors
     */
    public void setDictionary(Map<String, double[]> dict) {
        this.channelMap = dict;
        // Get the number of channels.  This is not the same as the number of values, as some values
        // may distribute among channels. (For example, in genomics, an ambiguity character would not
        // increase the number of channels, but would post fractional numbers in multiple channels.)
        double[] value1 = dict.values().iterator().next();
        this.channels = value1.length;
    }


    /**
     * This is the default method for pre-allocating the feature array.  It is two-dimensional,
     * one column per input column, one row per input record.
     *
     * @return an empty feature array for the output
     */
    @Override
    protected INDArray createFeatureArray() {
        int[] shape = new int[] { this.getBuffer().size(), this.channels, 1,
                this.getWidth() };
        return Nd4j.createUninitialized(shape);
    }

    /**
     * @return the vector of values corresponding to the specified data string
     *
     * @param string	input string to convert
     */
    protected double[] stringToVector(String string) {
        double[] retVal = this.channelMap.get(string);
        if (retVal == null)
            throw new IllegalArgumentException("Invalid channel value \"" + string + "\" in input.");
        return retVal;
    }

    /**
     * Read the channel specifications from a tab-delimited file.
     *
     * @param inFile	input file containing the channel specifications
     *
     * @return a hash mapping each input value to the channel values
     *
     * @throws IOException
     */
    public static Map<String, double[]> readChannelFile(File inFile) throws IOException {
        HashMap<String, double[]> retVal = new HashMap<String, double[]>();
        TabbedLineReader inStream = new TabbedLineReader(inFile);
        while (inStream.hasNext()) {
            double[] vector = new double[inStream.size() - 1];
            Line thisLine = inStream.next();
            String key = thisLine.get(0);
            for (int i = 1; i < inStream.size(); i++) {
                vector[i-1] = Double.parseDouble(thisLine.get(i));
            }
            retVal.put(key, vector);
        }
        inStream.close();
        return retVal;
    }

    /**
     * @return the number of input channels
     */
    @Override
    public int getChannels() {
        return this.channels;
    }

}
