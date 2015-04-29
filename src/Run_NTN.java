

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLNumericArray;

import edu.umass.nlp.optimize.IDifferentiableFn;
import edu.umass.nlp.optimize.IOptimizer;
import edu.umass.nlp.optimize.LBFGSMinimizer;
import edu.umass.nlp.utils.BasicPair;
import edu.umass.nlp.utils.DoubleArrays;
import edu.umass.nlp.utils.IPair;
public class Run_NTN {

	public static void main(String[] args) throws IOException {
		//Data path
		String data_path = "C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet//";
		
		//Paramters
		int batchSize = 1000; 			//training batch size, socherr: 20.000
		int numWVdimensions = 100; 		// size of the dimension of a word vector
		int numIterations = 10; 		// number of optimization iterations, every iteration with a new training batch job, socherr: 500
		int batch_iterations = 5;		// number of optimazation iterations for each batch	, currently not implemented
		int sliceSize = 3; 				//number of slices in the tensor w and v
		int corrupt_size = 10; 			// corruption size
		int activation_function=1; 		//not implemented always sigmoid, tanh or [x] sigmoid
		float lamda = 0.0001F;			// regulariization parameter
		
		//for help
		Util u = new Util();

		//Load data entities, relation, traingsdata, word vectors ...
		DataFactory tbj = DataFactory.getInstance(batchSize, corrupt_size);
		tbj.loadEntitiesFromSocherFile(data_path +"entities.txt");
		tbj.loadRelationsFromSocherFile(data_path + "//relations.txt");	
		tbj.loadTrainingDataTripplesE1rE2(data_path + "//train.txt");
		tbj.loadDevDataTripplesE1rE2Label(data_path + "//dev.txt");
		tbj.loadTestDataTripplesE1rE2Label(data_path + "//test.txt");
		tbj.loadWordVectorsFromMatFile(data_path + "\\initEmbed.mat");
		
		// Create the NTN and set the parameters for the NTN
		NTN t = new NTN(numWVdimensions, tbj.getNumOfentities(), tbj.getNumOfRelations(), batchSize, sliceSize, activation_function, 0.0001F);
		t.connectDatafactory(tbj);
		
		//Load initialized parameters
		double[] theta = t.getTheta_inital().data().asDouble();		
		
		//Train
		for (int i = 0; i < numIterations; i++) { 
			//Create a training batch by picking up (random) samples from training data	
			tbj.generateNewTrainingBatchJob();
			
			//Optimize the network using the training batch
			IOptimizer.Result res = (new LBFGSMinimizer()).minimize(t, theta, new LBFGSMinimizer.Opts());
			//System.out.println("result: " + DoubleArrays.toString(res.minArg));
			//TODO Minimize iterations of optimizer to 5 (like in original)
			//System.out.println("result of iteration i is computed: "+i);
			
			theta = res.minArg;
			
			//Storing paramters to start from this iteration again:
			Nd4j.writeTxt( u.convertDoubleArrayToFlattenedINDArray(theta), "C:\\Users\\Patrick\\Documents\\master arbeit\\theta_opt_iteration_"+i+".txt", ",");
			
		}		
		// save optimized theta paramters
		Nd4j.writeTxt(u.convertDoubleArrayToFlattenedINDArray(theta) , "C:\\Users\\Patrick\\Documents\\master arbeit\\theta_opt"+Calendar.getInstance().get(Calendar.DATE)+".txt", ",");
		System.out.println("model saved!");
		
		//Test
		// Load test data to calculate predictions
		INDArray best_theresholds = t.computeBestThresholds(u.convertDoubleArrayToFlattenedINDArray(theta), tbj.getDevTripples());
		System.out.println("best_theresholds: "+best_theresholds);
		
		//Calculate accuracy of the predictions and accuracy
		INDArray predictions = t.getPrediction(u.convertDoubleArrayToFlattenedINDArray(theta), tbj.getTestTripples(), best_theresholds);
		
		
	}
	


}
