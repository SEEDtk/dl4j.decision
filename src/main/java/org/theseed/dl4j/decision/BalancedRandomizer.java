/**
 *
 */
package org.theseed.dl4j.decision;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


import org.nd4j.linalg.dataset.DataSet;

/**
 * This randomizer selects subsets of the training sets balanced to have an equal number of members of each class.
 * Since some classes may have fewer members than required to fill the necessary slots, the choice method is
 * randomization with replacement.
 *
 * @author Bruce Parrello
 *
 */
public class BalancedRandomizer implements IRandomizer {

    // FIELDS
    /** number of classifications */
    private int nClasses;
    /** number of examples desired for each training subset */
    private int nSize;
    /** list of examples for each nonempty class */
    private List<List<DataSet>> outcomeSets;
    /** number of examples to choose for each class */
    private int oRows;

    @Override
    public void initializeData(int nClasses, int nSize, DataSet trainingSet) {
        this.nSize = nSize;
        this.nClasses = nClasses;
        // Split the dataset into outcome groups.
        this.outcomeSets = this.splitByOutcome(trainingSet);
        // Determine the number of rows to use for each outcome.
        this.oRows = (nSize + outcomeSets.size() - 1) / outcomeSets.size();
    }

    /**
     * Split the training set into subsets by outcome.
     *
     * @param dataset	dataset to split
     *
     * @return a list of split datasets by outcome
     */
    protected List<List<DataSet>> splitByOutcome(DataSet dataset) {
        // Create a list of dataset rows.
        List<DataSet> allRows = dataset.asList();
        int n = allRows.size();
        // Create lists for each output label.
        List<List<DataSet>> retVal = IntStream.range(0, this.nClasses)
                .mapToObj(i -> new ArrayList<DataSet>(n))
                .collect(Collectors.toList());
        // Copy each row into its label's list.
        for (DataSet row : allRows) {
            int label = DecisionTree.bestLabel(row);
            retVal.get(label).add(row);
        }
        // Remove the empty sets.
        Iterator<List<DataSet>> iter = retVal.iterator();
        while (iter.hasNext()) {
            List<DataSet> curr = iter.next();
            if (curr.isEmpty())
                iter.remove();
        }
        return retVal;
    }


    @Override
    public DataSet getData(long seed) {
        Random rand = new Random(seed);
        List<DataSet> rows = new ArrayList<DataSet>(this.nSize);
        for (List<DataSet> outcomeSet : this.outcomeSets) {
            for (int i = 0; i < this.oRows; i++) {
                int idx = rand.nextInt(outcomeSet.size());
                rows.add(outcomeSet.get(idx));
            }
        }
        DataSet retVal = DataSet.merge(rows);
        return retVal;
    }

}
