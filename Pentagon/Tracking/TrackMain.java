import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
/**
 * 
 * @author Yuanyuan Xu
 *
 */
public class TrackMain {
	static int r;
	static int row, col;
	static int interval;
	static int startIndex, endIndex;
	static String inputPath;
	static String outputPath;
	protected static JFileChooser ourChooser = new JFileChooser(System.getProperties().getProperty("user.dir"));

	
	public static void initialize() {
		// choose input path
		ourChooser.setDialogTitle("SELECT DATA DIRECTORY");
		ourChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		int retval = ourChooser.showOpenDialog(null);
		if (retval == JFileChooser.APPROVE_OPTION) {
			File file = ourChooser.getSelectedFile();
			inputPath = file.getAbsolutePath();
		}
		
		// enter parameters
		String input;
		input = JOptionPane.showInputDialog("ENTER PICTURE SIZE (ROW COL)", "1 1");
		String temp[] = input.split("\\s+");
		row = Integer.parseInt(temp[0]);
		col = Integer.parseInt(temp[1]);
		input = JOptionPane.showInputDialog("ENTER RADIUS OF PENTAGON", "62");
		r = Integer.parseInt(input);
		input = JOptionPane.showInputDialog("ENTER INITIAL STEP", "1");
		startIndex = Integer.parseInt(input);
		input = JOptionPane.showInputDialog("ENTER INTERVAL STEP", "1");
		interval = Integer.parseInt(input);
		input = JOptionPane.showInputDialog("ENTER FINAL STEP", "1");
		endIndex = Integer.parseInt(input);
		
		// choose output path
		ourChooser.setDialogTitle("SElECT A DIRECTORY TO SAVE DATA");
		ourChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		retval = ourChooser.showOpenDialog(null);
		if (retval == JFileChooser.APPROVE_OPTION) {
			File file = ourChooser.getSelectedFile();
			outputPath = file.getPath();
		}
	}
	
	public static String numToStr(int num) {
		String str = Integer.toString(num);
		for (int i = str.length(); i < 4; i++)
			str = '0' + str;
		return str;
	}
	
	public static void visualization() throws IOException {
		Object[] options ={ "No", "Yes" };
		int m = JOptionPane.showOptionDialog(null, "Visualization", "Pentagon Experiment", JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE, null, options, options[1]);
		System.out.println(m);
		Process proc = Runtime.getRuntime().exec("python " + "visualization.py");
	}
	
	public static void main(String[] args) throws IOException {
		Tracking track = new Tracking();
		initialize();
		int[] val = new int[(endIndex - startIndex) / interval + 1];
		File file = new File(inputPath + "/data_" + numToStr(startIndex));
		Scanner in = new Scanner(file);
		track.initialize(r, row, col, startIndex, in);
		for (int step = startIndex + interval; step <= endIndex; step += interval) {
			file = new File(inputPath + "/data_" + numToStr(step));
			in = new Scanner(file);
			val[(step - startIndex) / interval] = track.add(step, in);
			track.update();
		}
		track.printData(outputPath);
		int error = 0;
		for (int i = 0; i < val.length; i++)
			error += val[i];
		JOptionPane.showMessageDialog(null, "Average Error: " + (error/val.length)); 
		visualization();
	}
}
