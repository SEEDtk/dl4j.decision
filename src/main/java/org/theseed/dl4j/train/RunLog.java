/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.ZonedDateTime;

/**
 * This class encapsulates methods for writing to the trial log.
 *
 * @author Bruce Parrello
 *
 */
public class RunLog {

    // FIELDS
    /** marking constant for a section of the trial log */
    public static final String TRIAL_SECTION_MARKER = "******************************************************************";
    /** marking constant for a job start in the trial log */
    public static final String JOB_START_MARKER =     "##################################################################";

    /**
     * Write a job marker to the trial log.
     *
     * @param logFile	file to contain the trial log
     * @param type		typer of job
     *
     * @throws IOException
     */
    public static void writeTrialMarker(File logFile, String type) throws IOException {
        // Open the trials lot in append mode and write the job marker.
        try (PrintWriter trialWriter = new PrintWriter(new FileWriter(logFile, true))) {
            trialWriter.println(JOB_START_MARKER);
            // Compute the current time and date.
            ZonedDateTime now = ZonedDateTime.now();
            trialWriter.format("%s job at %s.%n%n", type, now.toString());
        }

    }

    /**
     * Write a report to the trial log.
     *
     * @param logFile file to contain the trial log
     * @param label   heading comment, if any
     * @param report  text of the report to write, with internal new-lines
     *
     * @throws IOException
     */
    public static void writeTrialReport(File logFile, String label, String report) throws IOException {
        // Open the trials log in append mode and write the information about this run.
        try (PrintWriter trialWriter = new PrintWriter(new FileWriter(logFile, true))) {
            trialWriter.println(TRIAL_SECTION_MARKER);
            if (label != null)
                trialWriter.print(label);
            trialWriter.println(report);
        }
    }


}
