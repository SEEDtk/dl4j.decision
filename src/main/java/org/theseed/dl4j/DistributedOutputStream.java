/**
 *
 */
package org.theseed.dl4j;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.StringUtils;
import org.theseed.counters.Shuffler;
import org.theseed.dl4j.train.ITrainingProcessor;

/**
 * This is ia class that produces a scrambled output stream for deep learning.  It buffers the entire stream in memory
 * and then writes it out in a different order.  One column will be selected as a label, and its vallues will be
 * distributed across the output file as evenly as possible.  If the label is discrete (CLASS), each value will be distributed.
 * If the label is continuous (REGRESSION), it will be divided into 10 range categories.
 *
 * @author Bruce Parrello
 *
 */
public abstract class DistributedOutputStream implements Closeable, AutoCloseable {

    // FIELDS
    /** underlying output writer */
    private PrintWriter writer;
    /** label column index */
    protected int labelIdx;
    /** saved header record */
    private String header;
    /** data line counter */
    private int outCount;
    /** number of fields expected in each input line */
    private int width;

    /**
     * Open a distributed output file.
     *
     * @param outputFile	file to receive the output
     * @param processor		processor model being run
     * @param label			name of the label field
     * @param headers		array of fields in the header record, containing column names
     *
     * @return the open output object
     *
     * @throws IOException
     */
    public static DistributedOutputStream create(File outputFile, ITrainingProcessor processor, String label, String[] headers) throws IOException {
        // Create the stream object.
        DistributedOutputStream retVal = processor.getDistributor();
        // Find the label.
        OptionalInt labelIdx0 = IntStream.range(0, headers.length).filter(i -> headers[i].contentEquals(label)).findFirst();
        if (! labelIdx0.isPresent())
            throw new IOException("Label column \"" + label + "\" not found in headers.");
        retVal.labelIdx = labelIdx0.getAsInt();
        // Save the header.
        retVal.header = StringUtils.join(headers, '\t');
        retVal.width = headers.length;
        // Open the output file.
        retVal.writer = new PrintWriter(outputFile);
        // Denote no data lines have been written.
        retVal.outCount = 0;
        return retVal;
    }

    /**
     * This is a utility comparator for sorting lists by size (largest to smallest).
     */
    private static class SizeSorter implements Comparator<List<String>> {

        @Override
        public int compare(List<String> o1, List<String> o2) {
            int retVal = o2.size() - o1.size();
            if (retVal == 0) {
                for (int i = 0; retVal == 0 && i < o1.size(); i++)
                    retVal = o1.get(i).compareTo(o2.get(i));
            }
            return retVal;
        }

    }

    /**
     * Queue a line for output.
     *
     * @param line	array of fields in the line
     *
     * @throws IOException
     */
    public void write(String[] line) throws IOException {
        // Verify the line length.
        if (line.length != this.width)
            throw new IOException("Incorrect number of fields in input line beginning with " + line[0] + ".");
        // Get the label.
        String label = line[this.labelIdx];
        // Join the line into a string.
        String outLine = StringUtils.join(line, '\t');
        // Put the line in the buffer.
        this.add(label, outLine);
        this.outCount++;
    }

    /**
     * @return the number of lines written
     */
    public int getOutputCount() {
        return this.outCount;
    }


    /**
     * Add a line to the output buffer.
     *
     * @param label		label of the line
     * @param line		value of the line
     */
    protected abstract void add(String label, String line);

    @Override
    public void close() {
        this.flush();
        writer.close();
    }

    /**
     * Peel off enough elements from list2 to increase the size of list1 to the desired value.
     *
     * @param list1		target list
     * @param list2		source list
     * @param size1		desired size of list1
     */
    public static void peel(List<String> list1, List<String> list2, int size1) {
        while(list1.size() < size1 && list2.size() > 0)
            list1.add(list2.remove(list2.size() - 1));
    }

    /**
     * Distribute a list into N pieces of approximately equal size.
     *
     * @param list1		source list
     * @param num		number of pieces
     */

    /**
     * Write all the accumulated output.
     */
    private void flush() {
        // Start with the header.
        this.writer.println(header);
        // Now write all the data lines.
        if (this.outCount > 0) {
            // Get a map of classes to data lines.
            Map<String, List<String>> classMap = this.getClassMap();
            // First we need to randomize all the classes.
            for (List<String> classLines : classMap.values())
                Collections.shuffle(classLines);
            // Sort the classes from smallest to largest.
            List<List<String>> sortedLists = classMap.values().stream().sorted(new SizeSorter()).collect(Collectors.toList());
            // Create the master list. At each stage we distribute the master list's pieces among the lists at the current stage.
            // It starts empty.
            List<List<String>> master = new ArrayList<List<String>>();
            for (int pos = 0; pos < sortedLists.size(); pos++) {
                // Turn the current level into a list of lists.
                List<List<String>> newMaster = sortedLists.get(pos).stream().map(x -> new Shuffler<String>(10).add1(x)).collect(Collectors.toList());
                // Add each of the existing lists to the new list items.
                int i = 0;
                for (List<String> oldList : master) {
                    newMaster.get(i).addAll(oldList);
                    i++;
                    if  (i >= newMaster.size()) i = 0;
                }
                master = newMaster;
            }
            // Now write the master lists in order.
            for (List<String> list : master)
                for (String line : list)
                    this.writer.println(line);
            // Make sure the writes actually happen.
            this.writer.flush();
        }
    }

    /**
     * Get a map of all the classes to write.
     */
    protected abstract Map<String, List<String>> getClassMap();

    /**
     * This handles discrete data sets, where every class is a string.
     */
    public static class Discrete extends DistributedOutputStream {

        // FIELDS
        /** buffer of input lines */
        private Map<String, List<String>> lineMap;

        public Discrete() {
            // Create the line buffer.
            this.lineMap = new HashMap<String, List<String>>();
        }

        @Override
        protected void add(String label, String line) {
            List<String> classList = this.lineMap.computeIfAbsent(label, x -> new ArrayList<String>(1000));
            classList.add(line);
        }

        @Override
        protected Map<String, List<String>> getClassMap() {
            return this.lineMap;
        }

    }

    /**
     * This handles continuous data sets, where every class is a portion of the data range.
     */
    public static class Continuous extends DistributedOutputStream {

        // FIELDS
        /** buffer of input lines */
        private SortedMap<Double, List<String>> lineMap;
        /** number of output classes to distribute */
        private static final int CONTINUOUS_CLASSES = 10;

        public Continuous() {
            // Create the line buffer.
            this.lineMap = new TreeMap<Double, List<String>>();
        }

        @Override
        protected void add(String label, String line) {
            Double labelNum = Double.valueOf(label);
            List<String> classList = this.lineMap.computeIfAbsent(labelNum, x -> new ArrayList<String>(1000));
            classList.add(line);
        }

        @Override
        protected Map<String, List<String>> getClassMap() {
            // Compute the size of each output class.
            int total = this.getOutputCount();
            int size = total / CONTINUOUS_CLASSES + 1;
            // Now merge the existing classes into bigger ones, each of similar size.  (They don't
            // have to be exact, or even close.)
            Map<String, List<String>> retVal = new HashMap<String, List<String>>(CONTINUOUS_CLASSES);
            int classIdx = 0;
            List<String> classList = createClassList(0, retVal, size);
            for (List<String> currList : this.lineMap.values()) {
                if (classList.size() >= size) {
                    classIdx++;
                    classList = createClassList(classIdx, retVal, size);
                }
                classList.addAll(currList);
            }
            return retVal;
        }

        /**
         * @return a string list integrated into the specified map to be filled with output lines
         *
         * @param classIdx		index to be assigned to the list
         * @param classMap		map into which the list should be kept
         * @param size			expected size of the list
         */
        private List<String> createClassList(int classIdx, Map<String, List<String>> classMap, int size) {
            List<String> classList;
            classList = new ArrayList<String>(size);
            classMap.put(Integer.toString(classIdx), classList);
            return classList;
        }

    }
}
