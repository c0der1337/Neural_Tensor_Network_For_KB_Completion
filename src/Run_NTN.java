

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
	//TODO all hashmaps are needed?
	private static HashMap<Integer, String> entitiesNumWord = new HashMap<Integer, String>();
	private static HashMap<Integer, String> relationsNumWord = new HashMap<Integer, String>();
	private static HashMap<String, Integer> entitiesWordNum = new HashMap<String, Integer>();
	private static HashMap<String, Integer> relationsWordNum = new HashMap<String, Integer>();
	
	//number for every tripple - necessay?
	private static HashMap<Integer, Tripple> trainingDataNumTripple = new HashMap<Integer, Tripple>();
	private static HashMap<Integer, Tripple> devDataNumTripple = new HashMap<Integer, Tripple>();
	private static HashMap<Integer, Tripple> testDataNumTripple = new HashMap<Integer, Tripple>();
	
	private static HashMap<Integer, String> vocabNumWord = new HashMap<Integer, String>(); //vocab of word vectors
	private static HashMap<String, Integer> vocabWordNum = new HashMap<String, Integer>();
	
	private static HashMap<String, INDArray> worvectorWordVec = new HashMap<String, INDArray>();
	private static HashMap<Integer, INDArray> worvectorNumVec = new HashMap<Integer, INDArray>();
	private static int numEntities;
	private static int numRelations;
	static ArrayList<Tripple> traingDataTripples;
	static ArrayList<Tripple> devDataTripples;
	static ArrayList<Tripple> testDataTripples;
	
	public static void main(String[] args) {
		//Paramters
		int batchSize = 200; 			//training batch size, socherr: 20.000
		int numWVdimensions = 100; 		// size of the dimension of a word vector
		int numIterations = 10; 		// number of optimization iterations, every iteration with a new training batch job, socherr: 500
		int batch_iterations = 5;		// number of optimazation iterations for each batch	, currently not implemented
		int sliceSize = 3; 				//number of slices in the tensor w and v
		int corrupt_size = 10; 			// corruption size
		int activation_function; 		//not implemented, tanh or [x] sigmoid
		float lamda = 0.0001F;			// regulariization parameter
		
		
		
		//Get entities and relation data dictionaries
		try {
			//getEntitiesFromSocherFile("C://Users//Patrick//Documents//master arbeit//Neural-Tensor-Network-master//entities.txt");
			getEntitiesFromSocherFile("C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet//entities.txt");
			//getRelationsFromSocherFile("C://Users//Patrick//Documents//master arbeit//Neural-Tensor-Network-master//relations.txt");
			getRelationsFromSocherFile("C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet//relations.txt");	
			//traingDataTripples = getTrainingDataTripplesE1rE2("C://Users//Patrick//Documents//master arbeit//Neural-Tensor-Network-master//train.txt");
			traingDataTripples = getTrainingDataTripplesE1rE2("C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet//train.txt");
			devDataTripples = getDevDataTripplesE1rE2Label("C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet//dev.txt");
			testDataTripples = getTestDataTripplesE1rE2Label("C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet//test.txt");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		//Get word vectors
		getWordVectorsFromMatFile("C:\\Users\\Patrick\\Documents\\master arbeit\\original_code\\data\\Wordnet\\initEmbed.mat");
		
		TrainingBatchJobDataFactory tbj = new TrainingBatchJobDataFactory(traingDataTripples,batchSize,corrupt_size,numEntities);
		
		// Create the NTN and set the parameters for the NTN
		NTN t = new NTN(numWVdimensions, tbj.getNumOfentities(), numRelations, 10, sliceSize, 0, 0.0001F);
		t.setEntitiesNumWord_global(entitiesNumWord);
		t.setWorvectorWordVec_global(worvectorWordVec);
		t.connectDatafactory(tbj);
		double[] theta = t.getTheta_inital().data().asDouble();
		
		for (int i = 0; i < numIterations; i++) { 
			//Create a training batch by picking up (random) samples from training data	
			tbj.generateNewTrainingBatchJob();
			
			//Optimize the network using the training batch
			
			//TODO Minimize iterations of optimizer to 5 
			 IOptimizer.Result res = (new LBFGSMinimizer()).minimize(t, theta, new LBFGSMinimizer.Opts());
			 //System.out.println("result: " + DoubleArrays.toString(res.minArg));
			 System.out.println("result of iteration i is computed: "+i);
			 theta = res.minArg;
			 
			
		}
		
		// save optimized theta paramters
		Calendar myCal = Calendar.getInstance();
		try {
			Nd4j.writeTxt( Nd4j.toFlattened(Nd4j.rand(3,4)) , "C:\\Users\\Patrick\\Documents\\master arbeit\\theta_opt"+myCal.get(Calendar.DATE)+".txt", ",");
			System.out.println("model saved!");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// Load test data to calculate predictions
		INDArray best_theresholds = t.computeBestThresholds(convertDoubleArrayToFlattenedINDArray(theta), devDataTripples);
		System.out.println("best_theresholds: "+best_theresholds);
		
		//Calculate accuracy of the predictions and accuracy
		INDArray predictions = t.getPrediction(convertDoubleArrayToFlattenedINDArray(theta), testDataTripples, best_theresholds);
		
		
	}
	
	static INDArray getSliceOfaTensor(INDArray tensor, int numOfSlice){
		//create the slice array	
		INDArray slice = Nd4j.zeros(tensor.slices(), tensor.slice(0).rows());
		//check for maximum possible slices
		if (numOfSlice<tensor.slice(0).columns()) {
			//extract the column of numOfSlice from each of the false slices
			for (int i = 0; i < tensor.slices(); i++) {
				//extract with getScalar, because slice.getColumn(1) on getSlice() returns false values
				INDArray sliceRow = Nd4j.zeros(1, tensor.slice(0).rows());				
				for (int row = 0; row < tensor.slice(0).rows(); row++){
					//slice.putRow(i, tensor.slice(i).getColumn(numOfSlice).swapAxes(0, 1));
					sliceRow.put(row, tensor.slice(i).getScalar(row, numOfSlice));
				}
				slice.putRow(i, sliceRow);
			}
		}else {
			System.out.println("Your Tensor has only "+tensor.slice(0).columns()+"-1 slices to extract");
		}		
		return slice;	
	}

	static  void getEntitiesFromSocherFile(String path) throws IOException{
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int entities_counter = 0;
	    while (line != null) {
	    	entitiesNumWord.put(entities_counter, line);
	    	entitiesWordNum.put(line,entities_counter);
	    	line = br.readLine();
	    	entities_counter++;
		}   
	    br.close();
	    //number of entities need increased by one to handle the zero entry
	    numEntities = entities_counter+1;
	    System.out.println(numEntities + " Entities loaded");
	}
	
	static void getRelationsFromSocherFile(String path) throws IOException{
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int relations_counter = 0;
	    while (line != null) {
	    	relationsNumWord.put(relations_counter,line);
	    	relationsWordNum.put(line,relations_counter);
	    	line = br.readLine();
	    	relations_counter++;
		}   
	    br.close();
	    numRelations = relations_counter+1;
	    System.out.println(relations_counter+ " Relations loaded");

	}
	
	static ArrayList<Tripple> getTrainingDataTripplesE1rE2(String path) throws IOException{
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    ArrayList<Tripple> tripples = new ArrayList();
	    String line = br.readLine();
	    int trainings_tripple_counter = 0;
	    while (line != null) {
	    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
	    	int rel = relationsWordNum.get(line.split("\\s")[1]);
	    	int e2 = entitiesWordNum.get(line.split("\\s")[2]);
	    	trainingDataNumTripple.put(trainings_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
	    	tripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
	    	line = br.readLine();
	    	trainings_tripple_counter++;
		}   
	    br.close();
	    System.out.println(trainings_tripple_counter+" Training Examples loaded...");
	    return tripples;

	}
	
	static ArrayList<Tripple> getDevDataTripplesE1rE2Label(String path) throws IOException{
		//get test data
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    ArrayList<Tripple> dev_tripples = new ArrayList();
	    String line = br.readLine();
	    int dev_tripple_counter = 0;
	    while (line != null) {
	    	//System.out.println("line: "+line.split("\\s")[0]+"|"+line.split("\\s")[1]+"|"+line.split("\\s")[2]+"|"+line.split("\\s")[3]+"|");
	    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
	    	int rel = relationsWordNum.get(line.split("\\s")[1]);
	    	int e2 = entitiesWordNum.get(line.split("\\s")[2]); 
	    	int label = Integer.parseInt(line.split("\\s")[3]);
	    	devDataNumTripple.put(dev_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
	    	dev_tripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
	    	line = br.readLine();
	    	dev_tripple_counter++;
		}   
	    br.close();
	    System.out.println(dev_tripple_counter +" Dev Examples loaded...");
	    return dev_tripples;

	}
	
	static ArrayList<Tripple> getTestDataTripplesE1rE2Label(String path) throws IOException{
		//get test data
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    ArrayList<Tripple> test_tripples = new ArrayList();
	    String line = br.readLine();
	    int test_tripple_counter = 0;
	    while (line != null) {
	    	//System.out.println("line: "+line.split("\\s")[0]+"|"+line.split("\\s")[1]+"|"+line.split("\\s")[2]+"|"+line.split("\\s")[3]+"|");
	    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
	    	int rel = relationsWordNum.get(line.split("\\s")[1]);
	    	int e2 = entitiesWordNum.get(line.split("\\s")[2]); 
	    	int label = Integer.parseInt(line.split("\\s")[3]);
	    	testDataNumTripple.put(test_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
	    	test_tripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
	    	line = br.readLine();
	    	test_tripple_counter++;
		}   
	    br.close();
	    System.out.println(test_tripple_counter +" Test Examples loaded...");
	    return test_tripples;

	}
	
	static void getWordVectorsFromMatFile(String path){
		try {
			MatFileReader matfilereader = new MatFileReader(path);
			//words
			MLCell words_mat = (MLCell) matfilereader.getMLArray("words");
			//word embeddings
			MLArray wordvectors_mat = (MLArray) matfilereader.getMLArray("We");
			MLDouble mlArrayDouble = (MLDouble) wordvectors_mat;
			//INDArray wordvectormatrix = Nd4j.zeros(100,1); //100 dimension of word embeddings
			String word;
			for (int i = 0; i < mlArrayDouble.getSize()/100; i++) {
				//System.out.println("word: "+words_mat.get(i).contentToString().substring(7,words_mat.get(i).contentToString().lastIndexOf("'")));				
				//load word vector
				word = words_mat.get(i).contentToString().substring(7,words_mat.get(i).contentToString().lastIndexOf("'"));
				vocabNumWord.put(i, word);
				vocabWordNum.put(word, i);
				INDArray wordvector = Nd4j.zeros(100,1);
				for (int j = 0; j < 100; j++) {
					wordvector.put(j, 0, mlArrayDouble.get(i, j));	
				}
				//wordvectormatrix.putColumn(i, wordvector);
				worvectorNumVec.put(i, wordvector);
				worvectorWordVec.put(word, wordvector);					
			}
			System.out.println(worvectorNumVec.size() +" Word Vectors loaded...");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private static INDArray convertDoubleArrayToFlattenedINDArray(double[] dArray){
		INDArray d = Nd4j.zeros(dArray.length);
		for (int i = 0; i < dArray.length; i++) {
			d.putScalar(i, dArray[i]);
		}
		return d;
	}

}
