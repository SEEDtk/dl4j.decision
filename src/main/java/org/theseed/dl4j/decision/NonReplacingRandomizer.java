/**
 *
 */
package org.theseed.dl4j.decision;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.nd4j.linalg.dataset.DataSet;

/**
 * This randomizer selects a training subset consisting of unique elements from the original training set.
 *
 * @author Bruce Parrello
 *
 */
public class NonReplacingRandomizer implements IRandomizer {

    // FIELDS
    /** number of examples to put in each subset */
    private int nSize;
    /** full list of examples */
    private List<DataSet> allRows;

    @Override
    public void initializeData(int nClasses, int nSize, DataSet trainingSet) {
        this.allRows = trainingSet.asList();
        int maxSize = this.allRows.size() / 2;
        this.nSize = (nSize <= maxSize ? nSize : maxSize);
    }

    @Override
    public DataSet getData(long seed) {
        // Choose the elements to pull.  Note that we cannot shuffle "allRows", because it would not be thread-safe.
        int n = allRows.size();
        int[] shuffler = IntStream.range(0, n).toArray();
        Random rand = new Random(seed);
        for (int i = 0; i < nSize; i++) {
            int j = rand.nextInt(n - i);
            int buffer = shuffler[i];
            shuffler[i] = shuffler[j];
            shuffler[j] = buffer;
        }
        // Create a dataset from the selected elements.
        List<DataSet> sample = IntStream.range(0, nSize).mapToObj(i -> this.allRows.get(shuffler[i]))
                .collect(Collectors.toList());
        return DataSet.merge(sample);
    }

}
