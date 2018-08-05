import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
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
	Modes class contains methods which are used to support three required modes of a program:
	1) prediction and evaluation modes
	2) the analytical solution training mode
	3) the gradient descent training mode
	+  polynomial features
*/
public class Modes{
	
	
	//get value of h(x)
	public double getRegFunc(RealVector X_ith, RealVector W_ith) {
		return X_ith.dotProduct(W_ith);
	}
	
	//get gradient as the vector of partial derivatives
	public RealVector getGradient(RealMatrix X_mx, RealVector Y_vc, RealVector W_vc) {
		int N_tr = Y_vc.getDimension();		
		RealVector gradient = X_mx.transpose().operate(X_mx.operate(W_vc).subtract(Y_vc));
		gradient.mapMultiplyToSelf(2.0).mapDivideToSelf(N_tr);
		return gradient;	
	}
	
	//get R(x)
	public double getEmpiricalRisk(RealMatrix X_mx,RealVector Y_vc, RealVector W_vc){
		double result = 0.0;
		int N_tr = Y_vc.getDimension();
		double sums = 0.0;
		for(int i = 0; i < N_tr;i++) {
			double Y_ith = Y_vc.getEntry(i);
			RealVector X_vc = X_mx.getRowVector(i);
			sums += ((getRegFunc(X_vc,W_vc) - Y_ith)* (getRegFunc(X_vc,W_vc) - Y_ith));
		}
		result = sums / N_tr;
		return result;
	}
	
	//Training mode: Gradient descent
	public RealVector gradientDescentForLinearRegression(double stepSize,RealMatrix X_mx, RealVector Y_vc, double stop_treshold){
		boolean converged = false;
		/*Init W_hat vector with 0-----*/
		double[] features = new double[X_mx.getColumnDimension()];
		RealVector W_hat = new ArrayRealVector(features);
		/*-------*/	
		int i = 0;
		while(!converged) {
			//oldObjectiveValue from previous iteration
			double oldObjectiveValue = getEmpiricalRisk(X_mx, Y_vc, W_hat);
			W_hat = W_hat.subtract(getGradient(X_mx,Y_vc,W_hat).mapMultiply(stepSize));
			//newObjectiveValue from current iteration
			double newObjectiveValue = getEmpiricalRisk(X_mx, Y_vc, W_hat); 
			converged = checkConvergence(oldObjectiveValue, newObjectiveValue, stop_treshold);
			i++;			
		}
		System.out.println(i);
		return W_hat;
	}
	
	/* Stop training when the decrease in objective value gets sufficiently small */
	public boolean checkConvergence(double oldObjectiveValue, double newObjectiveValue, double stop_treshold) {
		double relativeReduction = (oldObjectiveValue - newObjectiveValue) / oldObjectiveValue;
		boolean converged = (relativeReduction < stop_treshold) ? true : false;
		return converged;
	}
	
	//Training mode: Analytical Solution
	public RealVector getAnalyticalSolution(double stepSize, RealMatrix X_mx, RealVector Y_vc, double stop_treshold){
		RealVector resultVector = null;
		RealMatrix matrix = (X_mx.transpose());
		RealMatrix tempMx_1 =	matrix.multiply(X_mx);
		DecompositionSolver solver = new LUDecomposition(tempMx_1).getSolver();
		if(solver.isNonSingular()) {
			RealMatrix inverseMx = solver.getInverse();
			RealMatrix tempMx_2 = inverseMx.multiply(X_mx.transpose());
			resultVector = tempMx_2.operate(Y_vc);
		} else {//if singular - > gradient descent solution
			gradientDescentForLinearRegression(stepSize, X_mx, Y_vc, stop_treshold);
		}
		return resultVector;
	}
	
	//Training mode: General method
	public RealVector trainingMode(RealMatrix X_mx, RealVector Y_vc,String algorithm, double stepSize, double stop_treshold){
		RealVector outputVector = null;
		if(algorithm.equals(Constants.GRADIENT_DESCENT)) {
			outputVector = gradientDescentForLinearRegression(stepSize, X_mx, Y_vc,stop_treshold);
		} else {
			outputVector = getAnalyticalSolution(stepSize,X_mx,Y_vc,stop_treshold);
		}
		return outputVector;
	}
	
	/*Evaluation mode*/
	public double evaluationMode(RealMatrix X_mx, RealVector Y_vc, RealVector W_trained){
		double MSE = 0.0;
		int N_tr = Y_vc.getDimension();
		double sums = 0.0;
		for(int i = 0; i < N_tr;i++) {
			RealVector X_vc = X_mx.getRowVector(i);
			double Y_ith = Y_vc.getEntry(i);
			sums += ((getRegFunc(X_vc,W_trained) - Y_ith)*(getRegFunc(X_vc,W_trained) - Y_ith));
		}
		MSE = sums / N_tr;
		return MSE;
	}
	
	/*Prediction mode*/
	public RealVector predictionMode(RealMatrix X_mx, RealVector w_hat){
		RealVector resultPredictions = null;
		double[] predictions = new double[w_hat.getDimension()];
			for(int i = 0; i < w_hat.getDimension();i++) {
				predictions[i] = getRegFunc(X_mx.getRowVector(i), w_hat);
			}
			resultPredictions = new ArrayRealVector(predictions);
		return resultPredictions;
	}
	
	/*Polynomial fitting*/
	public RealMatrix performPolynomialFitting(RealVector X_vc, int K, int datapoints_numb){
		RealMatrix X_mx = null;
		int dimension = K + 1;
		double [][] datapoints = new double[datapoints_numb][dimension];
		for(int i = 0; i < datapoints_numb;i++) {
			datapoints[i][0] = 1.0;
			for(int j = 1; j <=K; j++) {
				datapoints[i][j] = Math.pow(X_vc.getEntry(i), j);
			}	
		}
		X_mx = new Array2DRowRealMatrix(datapoints);
		return X_mx;	
	}
	
	

}
