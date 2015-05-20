

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;

import org.nd4j.linalg.api.buffer.DataBuffer;
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
		Random rand = new Random();
		
		//Restrict data type to float to save memory
		Nd4j.dtype = DataBuffer.FLOAT;
		
		//Data path
		String data_path="";String theta_save_path=""; String theta_load_path="";
		try {
			data_path = args[0];
			theta_load_path = args[1];
			theta_save_path = args[2];
		} catch (Exception e) {
			data_path = "C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet//";
			theta_save_path = "C://Users//Patrick//Documents//master arbeit//";
		}

		//Paramters
		int batchSize = 20000; 				// training batch size, org: 20000
		int numWVdimensions = 100; 			// size of the dimension of a word vector org: 100
		int numIterations = 5; 			// number of optimization iterations, every iteration with a new training batch job, org: 500
		int batch_iterations = 500;			// number of optimazation iterations for each batchs, org: 5
		int sliceSize = 3; 					// number of slices in the tensor w and v
		int corrupt_size = 10; 				// corruption size, org: 10
		String activation_function= "tanh"; // [x] tanh or [] sigmoid, org:tanh
		float lamda = 0.0001F;				// regularization parameter, org: 0.0001
		boolean optimizedLoad=true;			// only load word vectors that are neede for entity vectors (>50% less), org: false
		
		System.out.println("NTN: batchSize: "+batchSize+" | SliceSize: "+sliceSize+" | numIterations:"+numIterations+" | corrupt_size: "+corrupt_size+"| activation func: "+ activation_function);
		
		//support utilities
		Util u = new Util();	
		//Load data entities, relation, traingsdata, word vectors ...
		DataFactory tbj = DataFactory.getInstance(batchSize, corrupt_size, numWVdimensions);
		tbj.loadEntitiesFromSocherFile(data_path +"entities.txt");
		tbj.loadRelationsFromSocherFile(data_path + "relations.txt");	
		tbj.loadTrainingDataTripplesE1rE2(data_path + "train.txt");
		tbj.loadDevDataTripplesE1rE2Label(data_path + "dev.txt");
		tbj.loadTestDataTripplesE1rE2Label(data_path + "test.txt");
		tbj.loadWordVectorsFromMatFile(data_path + "initEmbed.mat",optimizedLoad);
		
		// Create the NTN and set the parameters for the NTN
		NTN t = new NTN(numWVdimensions, tbj.getNumOfentities(), tbj.getNumOfRelations(), tbj.getNumOfWords(), batchSize, sliceSize, activation_function, tbj, lamda);
		t.connectDatafactory(tbj);
		
		//Load initialized parameters
		double[] theta = t.getTheta_inital().data().asDouble();
		//double[] theta = Nd4j.readTxt(theta_load_path, ",").data().asDouble();
		
		//Train
		for (int i = 0; i < numIterations; i++) { 
			//Create a training batch by picking up (random) samples from training data	
			tbj.generateNewTrainingBatchJob();
			
			
			LBFGSMinimizer.Opts optimizerOpts = new LBFGSMinimizer.Opts();
			//Set optimizer options: 5 iterations
			optimizerOpts.maxIters=batch_iterations;
			
			//Optimize the network using the training batch
			IOptimizer.Result res = (new LBFGSMinimizer()).minimize(t, theta, optimizerOpts);
			//System.out.println("result: " + DoubleArrays.toString(res.minArg));
			System.out.println("Paramters for batchjob optimized, current iteration: "+i);
			
			theta = res.minArg;
			
			//Storing paramters to start from this iteration again:
			Nd4j.writeTxt( u.convertDoubleArrayToFlattenedINDArray(theta), theta_save_path+"//theta_opt_iteration_"+i+".txt", ",");		
		}		
		// save optimized theta paramters
		Nd4j.writeTxt(u.convertDoubleArrayToFlattenedINDArray(theta) , theta_save_path+"//theta_opt"+Calendar.getInstance().get(Calendar.DATE)+".txt", ",");
		System.out.println("Model saved!");
		
		//Test
		// Load test data to calculate predictions
		INDArray best_theresholds = t.computeBestThresholds(u.convertDoubleArrayToFlattenedINDArray(theta), tbj.getDevTripples());
		System.out.println("Best theresholds: "+best_theresholds);
		
		//Calculate accuracy of the predictions and accuracy
		INDArray predictions = t.getPrediction(u.convertDoubleArrayToFlattenedINDArray(theta), tbj.getTestTripples(), best_theresholds);
	}
}
