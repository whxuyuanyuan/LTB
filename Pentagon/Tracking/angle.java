import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

public class angle {
	final static int N = 1499;
	
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
		for (int i = str.length(); i < 3; i++)
			str = '0' + str;
		return str;
	}
	public static double trueangle(double angle) {
		if (angle >= 0 && angle <= 72)
			return angle;
		if (angle > 72)
			return trueangle(angle - 72);
		else
			return trueangle(angle + 72);
	}
	public static void main(String[] args) throws IOException {
		double[] a = new double[N + 1];
		Arrays.fill(a, 0.0);
		initialize(); 
		
		for (int i = 1; i <= 835; i++) {
			File file = new File(inputPath + "/particle_" + numToStr(i) + ".txt");
			Scanner in = new Scanner(file);
			double temp, angle;
			double x, y;
			temp = in.nextDouble();
			x = in.nextDouble();
			y = in.nextDouble();
			angle = in.nextDouble();
			angle = trueangle(angle);
			for (int j = 1; j <= N; j++) {
				double currx, curry, currangle;
				temp = in.nextDouble();
				currx = in.nextDouble();
				curry = in.nextDouble();
				currangle = in.nextDouble();
				currangle = trueangle(currangle);
				a[j] += Math.min(Math.abs(currangle - angle), Math.abs(Math.abs(currangle - angle) - 72));
				angle = currangle;
			}
		}
		FileOutputStream out = new FileOutputStream(outputPath + "/angle.txt");
		for (int i = 1; i <= N; i++) {
			byte[] buf = (a[i] + "\n").getBytes();
			out.write(buf);
		}
		out.close();	
		


	}
}
