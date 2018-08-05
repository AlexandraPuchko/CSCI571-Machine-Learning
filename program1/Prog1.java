import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

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


public class Prog1 {
	
	public static void main(String[] args) {
			long startTime = System.nanoTime();
		
			Modes modes = new Modes();				
			IOHelper helper = new IOHelper();			
			List<String> commands = new ArrayList<>(); 
			Collections.addAll(commands, args); 
            
            /*Parameters to be initialized from the command line arguments*/
            String input_file = commands.get(1);
            RealMatrix X_mx = null;
            RealVector Y_vc = null;
            RealVector W_trained = null;
            RealVector outputVector = null;
            String target_file = null;
            String model_file = null;
            String algorithm = null;
            int datapoints__numb = 0;
            int dimension = 0;
            double step_size = 0.0;
            double stop_treshold = 0.0;
            int polinomial_order = 0;
            /*Parameters to be initialized from the command line arguments*/
            
            //EVALUATION MODE//
            if(commands.contains(Constants.EVAL_MODE)) { 
            	target_file = commands.get(2);
            	model_file = commands.get(3);
            	datapoints__numb = Integer.parseInt(commands.get(4));
            	dimension = Integer.parseInt(commands.get(5));           	
                Y_vc = helper.writeFromFileToVector(target_file,datapoints__numb,false);
                polinomial_order = Integer.parseInt(commands.get(6));  
              	if(polinomial_order > 1) {
            		if(dimension != 1) {
            			System.err.println("Uncompatible dimensions");
            			System.exit(0);	
            		}else {	                   	
                    	int w_trained_dim = polinomial_order +1;
                    	W_trained = helper.writeFromFileToVector(model_file,w_trained_dim,true);
            			RealVector X_vc = helper.writeFromFileToVector(input_file,datapoints__numb,false);
                		X_mx = modes.performPolynomialFitting(X_vc, polinomial_order, datapoints__numb);
            		}
            	}else if(polinomial_order == 1) {
            		int w_trained_dim = dimension  +1;         	
                	W_trained = helper.writeFromFileToVector(model_file,w_trained_dim,true);
            		X_mx = helper.writeFromFileToMatrix(input_file, datapoints__numb,dimension);
            	}
            	double MSE = modes.evaluationMode(X_mx,Y_vc,W_trained);
            	helper.printMSE(MSE);
            }
            //PREDICTION MODE//
            else if(commands.contains(Constants.PRED_MODE)) {
            	model_file = commands.get(2);
            	String predictions_file = commands.get(3);
            	datapoints__numb = Integer.parseInt(commands.get(4));
            	dimension = Integer.parseInt(commands.get(5));
            	polinomial_order = Integer.parseInt(commands.get(6));
            	if(polinomial_order > 1) {
            		if(dimension != 1) {
            			System.err.println("Uncompatible dimensions");
            			System.exit(0);	
            		}else {
            			RealVector X_vc = helper.writeFromFileToVector(input_file,datapoints__numb,false);
                		X_mx = modes.performPolynomialFitting(X_vc, polinomial_order, datapoints__numb);
            		}
            	}else if(polinomial_order == 1) {
            		X_mx = helper.writeFromFileToMatrix(input_file, datapoints__numb,dimension);
            	}
            	int w_trained_dim = dimension  + 1;
            	W_trained = helper.writeFromFileToVector(model_file,w_trained_dim,true);
            	outputVector = modes.predictionMode(X_mx, W_trained);
            	helper.writeModelToFile(predictions_file, outputVector);
            }
          //TRAINING MODE//
            else if(commands.contains(Constants.TRAIN_MODE)){
            	target_file = commands.get(2);
            	model_file = commands.get(3);
            	algorithm = commands.get(4);
            	//TRAINING MODE: GRADIENT DESCENT//
            	if(algorithm.equals(Constants.GRADIENT_DESCENT)) {
            		step_size = Double.parseDouble(commands.get(5));
                	stop_treshold = Double.parseDouble(commands.get(6));
                	datapoints__numb = Integer.parseInt(commands.get(7));
                	dimension = Integer.parseInt(commands.get(8));
                	polinomial_order = Integer.parseInt(commands.get(9));
                	if(polinomial_order > 1) {
                		if(dimension != 1) {
                			System.err.println("Uncompatible dimensions");
                			System.exit(0);	
                		}else {
                			RealVector X_vc = helper.writeFromFileToVector(input_file,datapoints__numb,false);
                    		X_mx = modes.performPolynomialFitting(X_vc, polinomial_order, datapoints__numb);
                		}
                	}else if(polinomial_order == 1) {
                		X_mx = helper.writeFromFileToMatrix(input_file, datapoints__numb,dimension);
                	}
            	}else {
            		//TRAINING MODE: Analytical Solution//
            		datapoints__numb = Integer.parseInt(commands.get(5));
                	dimension = Integer.parseInt(commands.get(6));
                	polinomial_order = Integer.parseInt(commands.get(7));
                	if(polinomial_order > 1) {
                		if(dimension != 1) {
                			System.err.println("Uncompatible dimensions");
                			System.exit(0);	
                		}else {
                			RealVector X_vc = helper.writeFromFileToVector(input_file,datapoints__numb,false);
                    		X_mx = modes.performPolynomialFitting(X_vc, polinomial_order, datapoints__numb);
                		}
                	}
                	else if(polinomial_order == 1) {
                		X_mx = helper.writeFromFileToMatrix(input_file, datapoints__numb,dimension);
                	}
            	 }  	
            	Y_vc = helper.writeFromFileToVector(target_file,datapoints__numb,false);
            	outputVector = modes.trainingMode(X_mx, Y_vc, algorithm, step_size,stop_treshold);
            	helper.writeModelToFile(model_file, outputVector);
            }   
            long endTime   = System.nanoTime();
        	long totalTime = endTime - startTime;
        	TimeUnit.MILLISECONDS.convert(totalTime, TimeUnit.NANOSECONDS);
        	System.out.println("total time = "+ totalTime);
		
	}

}
