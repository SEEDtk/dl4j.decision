/**
 *
 */
package org.theseed.reports;

import java.io.File;
import java.util.Collection;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.dl4j.train.IPredictError;

/**
 * This is a version of the validation report that produces no output, used for cross-validation.
 *
 * @author Bruce Parrello
 *
 */
public class NullValidationReport implements IValidationReport {

    @Override
    public void startReport(List<String> metaCols, List<String> labels) {
    }

    @Override
    public void reportOutput(List<String> metaData, INDArray expected, INDArray output) {
    }

    @Override
    public void finishReport(IPredictError errors) {
    }

    @Override
    public void close() {
    }

    @Override
    public void setupIdCol(File modelDir, String idCol, List<String> metaList, Collection<String> trainList) {
    }

}
