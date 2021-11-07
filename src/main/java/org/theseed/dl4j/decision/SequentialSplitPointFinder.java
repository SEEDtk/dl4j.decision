/**
 *
 */
package org.theseed.dl4j.decision;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This method uses an exhaustive search to find the best split point.
 *
 * @author Bruce Parrello
 */
public class SequentialSplitPointFinder extends SplitPointFinder {

    @Override
    public Splitter computeSplit(int iFeature, int nClasses, List<DataSet> rows, double entropy) {
        Splitter retVal = Splitter.NULL;
        // If there are only two rows, split on the mean.
        if (rows.size() <= 2) {
            double mean = DecisionTree.featureMean(rows, iFeature);
            retVal =  Splitter.computeSplitter(iFeature, mean, nClasses, rows, entropy);
        } else {
            // These arrays will contain the label sums for each side of the split.
            INDArray leftLabelSums = Nd4j.zeros(nClasses);
            INDArray rightLabelSums = Nd4j.zeros(nClasses);
            // Sort the rows by the feature value.  For each value, we count the number of occurrences of each label.
            SortedMap<Double, INDArray> rowMap = new TreeMap<Double, INDArray>();
            for (DataSet row : rows) {
                double value = row.getFeatures().getDouble(iFeature);
                INDArray valueGroup = rowMap.computeIfAbsent(value, v -> Nd4j.zeros(nClasses));
                // We use ravel here to convert a 1-row matrix to a vector.
                valueGroup.addi(row.getLabels().ravel());
                rightLabelSums.addi(row.getLabels().ravel());
            }
            // Start with the first value on the left.  Eliminate it from the right label sums.
            Iterator<Map.Entry<Double, INDArray>> iter = rowMap.entrySet().iterator();
            Map.Entry<Double, INDArray> curr = iter.next();
            while (iter.hasNext()) {
                leftLabelSums.addi(curr.getValue());
                rightLabelSums.subi(curr.getValue());
                Map.Entry<Double, INDArray> next = iter.next();
                double mean = (curr.getKey() + next.getKey()) / 2;
                Splitter test = Splitter.computeSplitter(iFeature, mean, entropy, leftLabelSums, rightLabelSums);
                if (test.compareTo(retVal) < 0)
                    retVal = test;
            }
        }
        return retVal;
    }

}
