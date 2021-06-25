/**
 *
 */
package org.theseed.reports;

import java.io.OutputStream;
import java.util.List;

import org.apache.commons.text.TextStringBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This class produces the validation report for a regression model.  For each output, the report will contain the
 * expected output (original name), the prediction output ("o." + name), and the error ("e." + name).
 *
 * @author Bruce Parrello
 *
 */
public class RegressionValidationReport extends BaseValidationReport {

    // FIELDS
    /** buffer for building output lines */
    private TextStringBuilder buffer;

    /**
     * Construct a regression validation report for a specified output stream.
     *
     * @param output	stream to receive the report
     */
    public RegressionValidationReport(OutputStream output) {
        super(output);
    }

    @Override
    public void startReport(List<String> metaCols, List<String> labels) {
       // Build the header line;
        this.buffer = new TextStringBuilder(14 * metaCols.size() + 52 * labels.size());
        for (String meta : metaCols)
            this.buffer.appendSeparator('\t').append(meta);
        // Note there are three columns per label.
        for (String label : labels) {
            this.buffer.append('\t');
            this.buffer.append(label).append("\to-").append(label).append("\te-").append(label);
        }
        String header = this.buffer.toString();
        this.println(this.fixHeaders(header));
    }

    @Override
    public void reportOutput(List<String> metaData, INDArray expected, INDArray output) {
        // Loop through the rows of output.
        for (int r = 0; r < metaData.size(); r++) {
            this.buffer.clear();
            String rowMeta = metaData.get(r);
            this.buffer.append(rowMeta);
            for (int i = 0; i < expected.columns(); i++) {
                double e = expected.getDouble(r, i);
                double o = output.getDouble(r, i);
                double error = o - e;
                this.buffer.append("\t%1.4f\t%1.4f\t%1.6f", e, o, error);
            }
            String dataLine = this.buffer.toString();
            this.println(this.fixDataLine(rowMeta, dataLine));
        }

    }

}
