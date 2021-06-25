/**
 *
 */
package org.theseed.reports;

import org.theseed.dl4j.train.ITrainReporter;

/**
 * This is a no-op for training reporting.  It is used as the default when no listening is needed.
 *
 * @author Bruce Parrello
 *
 */
public class NullTrainReporter implements ITrainReporter {

    @Override
    public void showMessage(String message) {
    }

    @Override
    public void showResults(String paragraph) {
    }

    @Override
    public void displayEpoch(int epoch, double score, double rating, boolean saved) {
    }

}
