/**
 *
 */
package org.theseed.dl4j.train;

/**
 * This interface represents an object that can be used to report the progress of a training run.
 *
 * @author Bruce Parrello
 *
 */
public interface ITrainReporter {

    /**'
     * Display a general message showing a major status change.
     *
     * @param message	message to display
     */
    public void showMessage(String message);

    /**
     * Report the results of a training run.
     *
     * @param paragraph		long, multiline string containing results
     */
    public void showResults(String paragraph);

    /**
     * Report the progress of the training.
     *
     * @param epoch		number of this epoch
     * @param score		score of this epoch
     * @param rating	rating of this epoch
     * @param saved		TRUE if the epoch model was saved
     * @throws InterruptedException
     */
    public void displayEpoch(int epoch, double score, double rating, boolean saved) throws InterruptedException;

}
