/**
 *
 */
package org.theseed.dl4j;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.train.ITrainingProcessor;

/**
 * This is a class that produces a scrambled output stream for deep learning.  It buffers the entire stream in memory
 * and then writes it out in a different order.  One column will be selected as a label, and its values will be
 * distributed across the output file as evenly as possible.  If the label is discrete (CLASS), each value will be distributed.
 * If the label is continuous (REGRESSION), it will be divided into 10 range categories.
 *
 * The client can specify quantity balancing.  If this is chosen, a limit will be placed on class (or category) size based on
 * the size of the smallest class.  A balance number of 2.0 means each class can be no more than twice the size of the smallest.
 * A balance number of 0.5 means each class can be no more than half the size of the smallest.
 *
 * @author Bruce Parrello
 *
 */
public abstract class DistributedOutputStream implements AutoCloseable {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(DistributedOutputStream.class);
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
    /** maximum ratio of a class set in terms of the smallest set, 0 to ignore */
    private double balanced;

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
        // Default to unbalanced.
        retVal.balanced = 0.0;
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
     * Denote whether the output should be balanced in terms of the smallest class.
     *
     * @param ratio		0 for unbalanced, otherwise the maximum ratio of other classes to the smallest
     */
    public void setBalanced(double ratio) {
        this.balanced = ratio;
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
            // If the lists need to be balanced, we do that here.
            if (this.balanced > 0.0) {
                // Update the list sizes.
                var sortedLists = new ArrayList<List<String>>(classMap.values());
                Collections.sort(sortedLists, new SizeSorter());
                this.trimLists(sortedLists);
            }
            // Distribute the lines into a master list.
            List<String> lineList = this.distribute(classMap);
            // Write out the list.
            for (String line : lineList)
                this.writer.println(line);
            // Make sure the writes actually happen.
            this.writer.flush();
        }
    }

    /**
     * Distribute the contents of the specified lists evenly into the output list.  This is a recursive method.  At each
     * level, we split the bigger lists evenly among the smaller lists and then concatenate the results.
     *
     * @param classMap	a map of output values to line lists
     *
     * @return a single distributed list of lines
     */
    private List<String> distribute(Map<String, List<String>> classMap) {
        // If there is only one key, we are done.
        List<String> retVal;
        if (classMap.size() <= 1)
             retVal = classMap.values().iterator().next();
        else {
            // Get the key with the fewest associated lines.
            String shortKey = getShortest(classMap);
            // We are going to build a new class map for each line from the shortest list.
            // These will be arranged in a list that we will move through in a circular
            // fashion.  The maps are tree maps because the number of keys is expected to
            // be very small.
            final List<String> shortestList = classMap.get(shortKey);
            List<Map<String, List<String>>> mapList = shortestList.stream().map(x -> new TreeMap<String, List<String>>())
                    .collect(Collectors.toList());
            // Initialize each map with all the keys other than the shortest-list's key.
            for (var map : mapList) {
                for (String key : classMap.keySet()) {
                    if (! key.contentEquals(shortKey))
                        map.put(key, new ArrayList<String>());
                }
            }
            // This will be our current position in the output maps.
            int i0 = 0;
            // Now loop through the class map, transferring lines.
            for (var mapEntry : classMap.entrySet()) {
                // This key will determine which key gets these lines in the smaller maps.
                String key = mapEntry.getKey();
                if (! key.contentEquals(shortKey)) {
                    // Loop through the lines for this key, transferring them to the smaller maps.
                    for (String line : mapEntry.getValue()) {
                        var smallMap = mapList.get(i0);
                        smallMap.get(key).add(line);
                        i0++;
                        if (i0 >= mapList.size()) i0 = 0;
                    }
                }
            }
            // Now we produce the output list.  We distribute each smaller map and append the result
            // to a line from the shortest-key list.
            retVal = new ArrayList<String>();
            for (int i = 0; i < mapList.size(); i++) {
                // Add the shortest-list line to the output.
                retVal.add(shortestList.get(i));
                // Distribute its class map and add that.
                List<String> distributedList = this.distribute(mapList.get(i));
                retVal.addAll(distributedList);
            }
        }
        return retVal;
    }

    /**
     * Remove excess lines from each list so that they all have the same length.
     *
     * @param sortedLists	list of line lists, sorted from longest to shortest
     */
    private void trimLists(List<List<String>> sortedLists) {
        // Get the length of the shortest list.
        int n1 = sortedLists.size() - 1;
        // Compute the maximum size of all other lists.
        final int len = (int) Math.ceil(sortedLists.get(n1).size() * this.balanced);
        // Remove excess members from all the lists.
        for (int i = 0; i < n1; i++) {
            var lineList = sortedLists.get(i);
            while (lineList.size() > len)
                lineList.remove(lineList.size() - 1);
        }
    }

    /**
     * Find the key with the shortest value list in a class map.
     *
     * @param classMap	map of values (keys) to lists of lines with the value
     *
     * @return the value with the shortest list in the specified class map,
     * 		   or NULL if the map is has no non-empty lists
     */
    private static String getShortest(Map<String, List<String>> classMap) {
        int len = Integer.MAX_VALUE;
        String retVal = null;
        for (var mapEntry : classMap.entrySet()) {
            int newLen = mapEntry.getValue().size();
            if (newLen < len) {
                retVal = mapEntry.getKey();
                len = newLen;
            }
        }
        return retVal;
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
