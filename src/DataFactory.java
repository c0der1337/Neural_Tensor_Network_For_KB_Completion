

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;

public class DataFactory{
	private static DataFactory instance;
	
	ArrayList<Tripple> trainingTripples = new ArrayList<Tripple>(); //data for learning the paramters, without labels
	ArrayList<Tripple> devTripples = new ArrayList<Tripple>(); //data for learning the threshold, with labels
	ArrayList<Tripple> testTripples = new ArrayList<Tripple>(); //data for learning the threshold, with labels
		 
	private HashMap<Integer, String> entitiesNumWord = new HashMap<Integer, String>();
	private HashMap<String, Integer> entitiesWordNum = new HashMap<String, Integer>();
	private HashMap<Integer, String> relationsNumWord = new HashMap<Integer, String>();
	private HashMap<String, Integer> relationsWordNum = new HashMap<String, Integer>();
	
	private HashMap<Integer, Tripple> trainingDataNumTripple = new HashMap<Integer, Tripple>();
	private HashMap<Integer, Tripple> devDataNumTripple = new HashMap<Integer, Tripple>();
	private HashMap<Integer, Tripple> testDataNumTripple = new HashMap<Integer, Tripple>();
	
	private HashMap<Integer, String> vocabNumWord = new HashMap<Integer, String>(); //contains vocab of word vectors and index of a word
	private HashMap<String, Integer> vocabWordNum = new HashMap<String, Integer>();
	private static HashMap<String, INDArray> worvectorWordVec = new HashMap<String, INDArray>(); // return word vector for a given word as string
	private static HashMap<Integer, INDArray> worvectorNumVec = new HashMap<Integer, INDArray>(); // return word vector for a given word index
	
	private int numOfentities;
	private int numOfRelations;
	
	private int batch_size;
	private int corrupt_size;
	

	// a random collection inclusive corrupt examples is created by evoking generateNewTrainingBatchJob()
	ArrayList<Tripple> batchjob = new ArrayList<Tripple>();  // contains the data of a batch training job to optimize paramters

	// Singelton
	public static DataFactory getInstance (int _batch_size, int _corrupt_size) {
		if (DataFactory.instance == null) {
			DataFactory.instance = new DataFactory (_batch_size, _corrupt_size);
		    }
		    return DataFactory.instance;
	
	}
	
	private DataFactory(int _batch_size, int _corrupt_size){
		batch_size = _batch_size;
		corrupt_size = _corrupt_size;
	}
	public int getNumOfentities() {
		return numOfentities;
	}
	
	public ArrayList<Tripple> getBatchJobTripplesOfRelation(int _relation_index){
		ArrayList<Tripple> tripplesOfThisRelationFromBatchJob = new ArrayList<Tripple>();
		//System.out.println("batchjob size: "+batchjob.size());
		for (int i = 0; i < batchjob.size(); i++) {
			if (batchjob.get(i).getIndex_relation()==_relation_index) {
				tripplesOfThisRelationFromBatchJob.add(trainingTripples.get(i));
			}
		}
		return tripplesOfThisRelationFromBatchJob;
	}
	
	public ArrayList<Tripple> getTripplesOfRelation(int _relation_index, ArrayList<Tripple> _listWithTripples){
		ArrayList<Tripple> tripples = new ArrayList<Tripple>();
		//System.out.println("batchjob size: "+batchjob.size());
		for (int i = 0; i < _listWithTripples.size(); i++) {
			if (_listWithTripples.get(i).getIndex_relation()==_relation_index) {
				tripples.add(_listWithTripples.get(i));
			}
		}
		return tripples;
	}

	public ArrayList<Tripple> getAllTrainingTripples() {
		return trainingTripples;
	}

	public void setTrainingTripples(ArrayList<Tripple> tripples) {
		this.trainingTripples = tripples;
	}
	public void generateNewTrainingBatchJob(){
		//TODO if batchjob is greater than amount of triples, start from zero again
		Collections.shuffle(trainingTripples);
		for (int h = 0; h < corrupt_size; h++) {
			for (int i = 0; i < batch_size; i++) {
				batchjob.add(trainingTripples.get(i));
			}
		}
		System.out.println("Training Batch Job created and contains of "+batchjob.size()+" Trippels.");
	}
	public void getEntity1vectormatrixOfBatchJob(){
		for (int i = 0; i < i; i++) {
			
		}
	}
	
	public INDArray getINDArrayOfTripples(ArrayList<Tripple> _tripples){
		INDArray tripplesMatrix = Nd4j.zeros(_tripples.size(),3);
		
		for (int i = 0; i < _tripples.size(); i++) {
			tripplesMatrix.put(i,0, _tripples.get(i).getIndex_entity1());
			tripplesMatrix.put(i,1, _tripples.get(i).getIndex_relation());
			tripplesMatrix.put(i,2, _tripples.get(i).getIndex_entity2());
			System.out.println("tripplesMatrix: "+tripplesMatrix);
		}
		
		return tripplesMatrix;
	}
	
	public void loadEntitiesFromSocherFile(String path) throws IOException{
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
	    numOfentities = entities_counter+1;
	    System.out.println(numOfentities + " Entities loaded");
	}
	
	public void loadRelationsFromSocherFile(String path) throws IOException{
		// example: _has_instance
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
	    numOfRelations = relations_counter+1;
	    System.out.println(numOfRelations+ " Relations loaded");

	}
	
	public void loadTrainingDataTripplesE1rE2(String path) throws IOException{
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int trainings_tripple_counter = 0;
	    while (line != null) {
	    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
	    	int rel = relationsWordNum.get(line.split("\\s")[1]);
	    	int e2 = entitiesWordNum.get(line.split("\\s")[2]);
	    	trainingDataNumTripple.put(trainings_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
	    	trainingTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
	    	line = br.readLine();
	    	trainings_tripple_counter++;
		}   
	    br.close();
	    //generate the corrupt 3. tripple
		for (Tripple tripple : trainingTripples) {	
			Random rand = new Random();
			int random_corrupt_entity = rand.nextInt((corrupt_size - 1) + 1) + 1; //rand.nextInt((max - min) + 1) + min		
			tripple.setIndex_entity3_corrupt(random_corrupt_entity);
		}
	    System.out.println(trainings_tripple_counter+" Training Examples loaded...");
	}
	
	void loadDevDataTripplesE1rE2Label(String path) throws IOException{
		//get dev data for calculation the threshold
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int dev_tripple_counter = 0;
	    while (line != null) {
	    	//System.out.println("line: "+line.split("\\s")[0]+"|"+line.split("\\s")[1]+"|"+line.split("\\s")[2]+"|"+line.split("\\s")[3]+"|");
	    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
	    	int rel = relationsWordNum.get(line.split("\\s")[1]);
	    	int e2 = entitiesWordNum.get(line.split("\\s")[2]); 
	    	int label = Integer.parseInt(line.split("\\s")[3]);
	    	devDataNumTripple.put(dev_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
	    	devTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
	    	line = br.readLine();
	    	dev_tripple_counter++;
		}   
	    br.close();
	    System.out.println(dev_tripple_counter +" Dev Examples loaded...");
	}
	
	void loadTestDataTripplesE1rE2Label(String path) throws IOException{
		//get test data
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int test_tripple_counter = 0;
	    while (line != null) {
	    	//System.out.println("line: "+line.split("\\s")[0]+"|"+line.split("\\s")[1]+"|"+line.split("\\s")[2]+"|"+line.split("\\s")[3]+"|");
	    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
	    	int rel = relationsWordNum.get(line.split("\\s")[1]);
	    	int e2 = entitiesWordNum.get(line.split("\\s")[2]); 
	    	int label = Integer.parseInt(line.split("\\s")[3]);
	    	testDataNumTripple.put(test_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
	    	testTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
	    	line = br.readLine();
	    	test_tripple_counter++;
		}   
	    br.close();
	    System.out.println(test_tripple_counter +" Test Examples loaded...");

	}
	
	void loadWordVectorsFromMatFile(String path){
		//load word vectors with a dimension of 100 from a matlab mat file
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
	public int getNumOfRelations() {
		return numOfRelations;
	}
	
	public INDArray createVectorsForEachEntityByWordVectors(INDArray entity_vectors){
		for (int i = 0; i < entitiesNumWord.size(); i++) {
			String entity_name;
			try {
				entity_name = entitiesNumWord.get(i).substring(2, entitiesNumWord.get(i).lastIndexOf("_"));
			} catch (Exception e) {
				entity_name =entitiesNumWord.get(i).substring(2);			
			}
			//System.out.println("entity: "+entitiesNumWord.get(i)+" | word: "+entity_name);			
			if (entity_name.contains("_")) { //whitespaces are _
				INDArray entityvector = Nd4j.zeros(100, 1);
				for (int j = 0; j <entitiesNumWord.get(i).split("_").length; j++) {
					try {
						entityvector = entityvector.add(worvectorWordVec.get(entity_name.split("_")[j]));
					} catch (Exception e) {
						entityvector = entityvector.add(worvectorWordVec.get("unknown"));
					}			
				}
				entityvector = entityvector.div(entity_name.split("_").length);
				entity_vectors.putColumn(i, entityvector);
			}else{
				//if no word vector available, use "unknown" word vector
				try {
					entity_vectors.putColumn(i, worvectorWordVec.get(entity_name));
				} catch (Exception e) {
					entity_vectors.putColumn(i, worvectorWordVec.get("unknown"));
				}			
			}		
		}
		return entity_vectors;
	}

	public ArrayList<Tripple> getDevTripples() {
		return devTripples;
	}
	public ArrayList<Tripple> getTestTripples() {
		return testTripples;
	}

}
