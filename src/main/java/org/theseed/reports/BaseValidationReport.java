/**
 *
 */
package org.theseed.reports;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.train.IPredictError;
import org.theseed.io.LineReader;

/**
 * This is a base class for validation reports.  It contains a few methods common to all of them.
 * @author Bruce Parrello
 *
 */
public abstract class BaseValidationReport extends BaseReporter implements IValidationReport {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ClassValidationReport.class);
    /** training set ID list */
    private Set<String> trained;
    /** column index of ID column in metadata columns, or -1 if there is no ID column */
    private int idColIdx;

    /**
     * Construct a validation reporter for a specified output stream.
     *
     * @param output	output stream to receive the report
     */
    public BaseValidationReport(OutputStream output) {
        super(output);
        this.idColIdx = -1;
    }

    @Override
    public void finishReport(IPredictError errors) {
        log.info("Mean error from validation = {}.", errors.getError());
        log.info("**{}", Arrays.stream(errors.getTitles()).map(x -> String.format(" %14s", x)).collect(Collectors.joining()));
        log.info("**{}", Arrays.stream(errors.getStats()).mapToObj(x -> String.format(" %14.8f", x)).collect(Collectors.joining()));
    }

    /**
     * Set up to track which input rows were included in the model's original training set.
     *
     * @param modelDir	model directory
     * @param idCol		ID column name
     * @param metaList	list of metadata column names
     * @param trainList	if specified, a list containing the training set
     *
     * @throws IOException
     */
    @Override
    public void setupIdCol(File modelDir, String idCol, List<String> metaList, Collection<String> trainList) throws IOException {
        this.idColIdx = metaList.indexOf(idCol);
        if (this.idColIdx < 0)
            throw new IllegalArgumentException("ID column \"" + idCol + "\" not found in metadata list.");
        if (trainList != null) {
            this.trained = new HashSet<String>(trainList);
        } else {
            File trainedFile = new File(modelDir, "trained.tbl");
            this.trained = LineReader.readSet(trainedFile);
        }
    }

    /**
     * Fix up the header line if we are marking data rows from the training set.
     *
     * @param headers	raw header line
     *
     * @return the header line to use
     */
    protected String fixHeaders(String headers) {
        String retVal = headers;
        if (this.idColIdx >= 0)
            retVal += "\ttrained";
        return retVal;
    }

    /**
     * Fix up a data line if we are marking data rows from the training set.
     *
     * @param metadata	metadata label
     * @param dataLine	raw data line
     *
     * @return the data line to use
     */
    protected String fixDataLine(String metadata, String dataLine) {
        String retVal = dataLine;
        if (this.idColIdx >= 0) {
            // Get the ID from the metadata.
            String id = StringUtils.split(metadata, '\t')[this.idColIdx];
            // Mark the row if the ID was found in the training set IDs.
            String mark = (this.trained.contains(id) ? "\tY" : "\t");
            retVal += mark;
        }
        return retVal;
    }


}
