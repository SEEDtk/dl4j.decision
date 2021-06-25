/**
 *
 */
package org.theseed.dl4j.decision;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.nd4j.linalg.dataset.DataSet;

/**
 * This randomly selects data rows with replacement, which is the simplest of all the possible algorithms.
 *
 * @author Bruce Parrello
 *
 */
public class ReplacingRandomizer implements IRandomizer {

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
        Random rand = new Random(seed);
        List<DataSet> sample = rand.ints(this.nSize, 0, this.allRows.size()).mapToObj(i -> this.allRows.get(i))
                .collect(Collectors.toList());
        return DataSet.merge(sample);
    }

}
