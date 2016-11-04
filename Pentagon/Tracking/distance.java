import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

public class distance {
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
	
	public static void main(String[] args) throws IOException {
		double[] dis = new double[N + 1];
		Arrays.fill(dis, 0.0);
		initialize(); 
		
		for (int i = 1; i <= 835; i++) {
			File file = new File(inputPath + "/particle_" + numToStr(i) + ".txt");
			Scanner in = new Scanner(file);
			double temp;
			double x, y;
			temp = in.nextDouble();
			x = in.nextDouble();
			y = in.nextDouble();
			temp = in.nextDouble();
			for (int j = 1; j <= N; j++) {
				double currx, curry;
				temp = in.nextDouble();
				currx = in.nextDouble();
				curry = in.nextDouble();
				temp = in.nextDouble();
				dis[j] += Math.sqrt((currx - x)*(currx - x) + (curry - y)*(curry - y));
				x = currx;
				y = curry;
			}
		}
		FileOutputStream out = new FileOutputStream(outputPath + "/distance.txt");
		for (int i = 1; i <= N; i++) {
			byte[] buf = (dis[i] + "\n").getBytes();
			out.write(buf);
		}
		out.close();	
		


	}
}
