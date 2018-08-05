import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Scanner;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
/*
 * 
 * @author Alexandra Puchko
 * CSCI597I Machine Learning
 * Program 1
 * Due date: April 25,2018
 * 
 */


/** 
	IOHelper class is responsible for reading a data from file and writing it to a file
*/
public class IOHelper {	
	
	//parse file and write data to RealMatrix
	public RealMatrix writeFromFileToMatrix(String fileName,int datapoints_numb,int columns) {
	 	RealMatrix matrix =  null;
	 	double [][] datapoints = new double[datapoints_numb][columns + 1];
	    try {
	    	String file_name = this.getClass().getClassLoader().getResource("").getPath() + fileName;
	        BufferedReader file = new BufferedReader(new FileReader(file_name));
	        LineNumberReader lr = new LineNumberReader(file);
	        String line = "";
	        int row_numb = 0;
	        while(((line = lr.readLine()) != null)) {
	        		String[] splited = line.split(" ");
	        		datapoints[row_numb][0] = 1.0;
		        	for(int i = 1; i <= splited.length;i++) {
		        		datapoints[row_numb][i] = Double.parseDouble(splited[i - 1]);
		        	}
	        	row_numb++;
	        }	
	        matrix = new Array2DRowRealMatrix(datapoints);
	        file.close();
	        
	    } catch (IOException e) {
	    }
	    
	    return matrix;

   }
	
	//parse file and write data to RealVector
	public RealVector writeFromFileToVector(String fileName, int rows, boolean fromModel) {
	 	 double[] datapoints = new double[rows];
	 	 String file_name = this.getClass().getClassLoader().getResource("").getPath() + fileName;
	 	 RealVector vc = null;
	 	 Scanner scanner = null;
	     try {
	        BufferedReader file = new BufferedReader(new FileReader(file_name));
	        scanner = new Scanner(file);
	        int i = 0;
	        while (scanner.hasNext()) {
	        	String datapt = scanner.next();
	        	if(datapt != null) {
	        		if(fromModel) {
	        			datapoints[i] = (Double.parseDouble(datapt));
	        		} else {
	        			datapoints[i] = (Double.parseDouble(datapt));
	        		}	
		        	i++;
	        	}	
	        }
	        file.close();
	        vc =  new ArrayRealVector(datapoints);
	    } catch (IOException e) {
	    }
	    return vc;
	}
	
	
	//write output data to file
	public void writeModelToFile(String fileName,RealVector outputVector) {
		Writer writer = null;
		String file_name = this.getClass().getClassLoader().getResource("").getPath() + fileName;
		try {
		    writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file_name), "utf-8"));
		    for(int i = 0; i < outputVector.getDimension();i++) {
		    	writer.write(String.format("%.3E",outputVector.getEntry(i)) + "\n");
		    }
		} catch (IOException ex) {
		    // Report
		} finally {
		   try {
			   writer.close();
		} 
		   catch (Exception ex) {/*ignore*/}
		}
	}
	
	//print Mean-squareed error in scientific notation to stdout
	public void printMSE(double MSE){
		System.out.println("The MSE is: " + String.format("%.3E",MSE));
	}
}
