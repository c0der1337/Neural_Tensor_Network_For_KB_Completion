

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
	
	private HashMap<Integer, String> entitiesNumtranslateWord = new HashMap<Integer, String>();
	private HashMap<String, String> entitiesWordtranslateWord = new HashMap<String, String>();
	
	private HashMap<Integer, String> relationsNumWord = new HashMap<Integer, String>();
	private HashMap<String, Integer> relationsWordNum = new HashMap<String, Integer>();
	
	private HashMap<Integer, Tripple> trainingDataNumTripple = new HashMap<Integer, Tripple>();
	private HashMap<Integer, Tripple> devDataNumTripple = new HashMap<Integer, Tripple>();
	private HashMap<Integer, Tripple> testDataNumTripple = new HashMap<Integer, Tripple>();
	
	private HashMap<Integer, String> vocabNumWord = new HashMap<Integer, String>(); //contains vocab of word vectors and index of a word
	private HashMap<String, Integer> vocabWordNum = new HashMap<String, Integer>();
	private static HashMap<String, INDArray> worvectorWordVec = new HashMap<String, INDArray>(); // return word vector for a given word as string
	private static HashMap<Integer, INDArray> worvectorNumVec = new HashMap<Integer, INDArray>(); // return word vector for a given word index
	private INDArray wordVectorMaxtrixLoaded; //matrix contains the original loaded word vectors, size: Word_embedding * numOfWords, every column contains a word vector
	
	private ArrayList<String> vocab = new ArrayList<String>(); //List with words of entities for only loading word vectors, that need for entity vector creation
	private ArrayList<String> vocabDE = new ArrayList<String>(); //List with words of entities for only loading word vectors, that need for entity vector creation
	
	private HashMap<Integer, String> vocabNumWordDE = new HashMap<Integer, String>(); //contains vocab of word vectors and index of a word
	private HashMap<String, Integer> vocabWordNumDE = new HashMap<String, Integer>();
	private static HashMap<String, INDArray> worvectorWordVecDE = new HashMap<String, INDArray>();
	private static HashMap<Integer, INDArray> worvectorNumVecDE = new HashMap<Integer, INDArray>();
	
	
	private int numOfentities;
	private int numOfRelations;
	private int numOfWords; 
	private int batch_size;
	private int corrupt_size;
	private int embeddings_size;
	private boolean reduced_RelationSize;
	private int reduceRelationToSize=2;
	private boolean german;
	

	// a random collection inclusive corrupt examples is created by evoking generateNewTrainingBatchJob()
	private ArrayList<Tripple> batchjob = new ArrayList<Tripple>();  // contains the data of a batch training job to optimize paramters

	// Singelton
	public static DataFactory getInstance (int _batch_size, int _corrupt_size, int _embedding_size, boolean _reduce_RelationSize, boolean _german) {
		if (DataFactory.instance == null) {
			DataFactory.instance = new DataFactory (_batch_size, _corrupt_size, _embedding_size, _reduce_RelationSize, _german);
		    }
		    return DataFactory.instance;
	
	}
	
	private DataFactory(int _batch_size, int _corrupt_size,int _embedding_size, boolean _reduce_RelationSize, boolean _german){
		batch_size = _batch_size;
		corrupt_size = _corrupt_size;
		embeddings_size = _embedding_size;
		reduced_RelationSize = _reduce_RelationSize;
		german = _german;
	}
	public int getNumOfentities() {
		return numOfentities;
	}
	
	public ArrayList<Tripple> getBatchJobTripplesOfRelation(int _relation_index){
		ArrayList<Tripple> tripplesOfThisRelationFromBatchJob = new ArrayList<Tripple>();
		//System.out.println("batchjob size: "+batchjob.size()+" | trainingTripples size: "+trainingTripples.size());
		for (int i = 0; i < batchjob.size(); i++) {
			if (batchjob.get(i).getIndex_relation()==_relation_index) {
				tripplesOfThisRelationFromBatchJob.add(batchjob.get(i));
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


	public void generateNewTrainingBatchJob(){
		batchjob.clear();
		//TODO wieder zulassen: Collections.shuffle(trainingTripples);
		Random rand = new Random();
		// int[] e3 = {9167, 2100, 30636, 13136, 22825, 24943};
		// int[] e3 = {9167, 2100, 30636, 13136, 22825, 24943,1500};
		// int e3counter=0;
		for (int h = 0; h < corrupt_size; h++) {
			for (int i = 0; i < batch_size; i++) {
				//Set e3 as random corrupt tripple			
				//min =0, max = maxIndexNumOfEntity << numOfEntity-1
				// int randomNum = rand.nextInt((max - min) + 1) + min;
				int random_corrupt_entity = rand.nextInt(((numOfentities-1) - 0) + 1) + 0;
				batchjob.add(new Tripple(trainingTripples.get(i), random_corrupt_entity));
				// batchjob.add(new Tripple(trainingTripples.get(i), e3[e3counter]));
				// e3counter++;
				
			}//System.out.println("e1: "+trainingTripples.get(i).getEntity1()+" | e3:"+trainingTripples.get(i).getIndex_entity3_corrupt());
		}
		System.out.println("Training Batch Job created and contains of "+batchjob.size()+" Trippels.");
	}
	public void getEntity1vectormatrixOfBatchJob(){
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
	    	//get words from entity 	
	    	String entity_name_clear; //clear name without _  __name_ -> name
			try {
				entity_name_clear = line.substring(2, line.lastIndexOf("_"));
			} catch (Exception e) {
				entity_name_clear =line.substring(2);			
			}
			//System.out.println("entity_name_clear: "+entity_name_clear);			
			
			if (entity_name_clear.contains("_")) { //whitespaces are _
				//Entity conains of more than one word
				for (int j = 0; j <entity_name_clear.split("_").length; j++) {
						vocab.add(entity_name_clear.split("_")[j]);
				}
			}else{
				// Entity conains of only one word
				vocab.add(entity_name_clear);
			}
			
	    	line = br.readLine();
	    	entities_counter++;
		}  
	    br.close();
	    //number of entities need increased by one to handle the zero entry
	    numOfentities = entities_counter;
	    System.out.println(numOfentities + " Entities loaded, containing of "+vocab.size()+" different words| last entity:"+entitiesNumWord.get(entitiesNumWord.size()-1));
	}
	
	public void loadRelationsFromSocherFile(String path) throws IOException{
		// example: _has_instance
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int relations_counter = 0;
  

	    if (reduced_RelationSize==false) {
		    while (line!=null) {
		    	relationsNumWord.put(relations_counter,line);
		    	relationsWordNum.put(line,relations_counter);
		    	line = br.readLine();
		    	relations_counter++;
			} 
			numOfRelations = relations_counter;
		}else{
		    while (relations_counter<reduceRelationToSize) {
		    	relationsNumWord.put(relations_counter,line);
		    	relationsWordNum.put(line,relations_counter);
		    	line = br.readLine();
		    	relations_counter++;
			} 
	    	numOfRelations = relations_counter;
		}
	    br.close();
	    System.out.println(numOfRelations+ " Relations loaded reduced relation size is: "+reduced_RelationSize);

	}
	
	public void loadTrainingDataTripplesE1rE2(String path) throws IOException{
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int trainings_tripple_counter = 0;
	    while (line != null) {
	    	//System.out.println(trainings_tripple_counter+ " line: "+line.split("\\s")[0]+"|"+line.split("\\s")[1]+"|"+line.split("\\s")[2]);
	    	if (reduced_RelationSize==false) {
		    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
		    	int rel = relationsWordNum.get(line.split("\\s")[1]);
		    	int e2 = entitiesWordNum.get(line.split("\\s")[2]);
		    	trainingDataNumTripple.put(trainings_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
		    	trainingTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
		    	line = br.readLine();
		    	trainings_tripple_counter++;
	    	}else{	
	    		if (line.split("\\s")[1].equals(relationsNumWord.get(0))|line.split("\\s")[1].equals(relationsNumWord.get(1))) {
	    			int e1 = entitiesWordNum.get(line.split("\\s")[0]);
			    	int rel = relationsWordNum.get(line.split("\\s")[1]);
			    	int e2 = entitiesWordNum.get(line.split("\\s")[2]);
			    	trainingDataNumTripple.put(trainings_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
			    	trainingTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2] ));
			    	line = br.readLine();
			    	trainings_tripple_counter++;
				}else{
					line = br.readLine();
				}
					
	    	}
		}   
	    br.close();
	    System.out.println(trainings_tripple_counter+" Training Examples loaded...");
	}
	
	public void loadDevDataTripplesE1rE2Label(String path) throws IOException{
		//get dev data for calculation the threshold
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int dev_tripple_counter = 0;
	    while (line != null) {
	    	//System.out.println("line: "+line.split("\\s")[0]+"|"+line.split("\\s")[1]+"|"+line.split("\\s")[2]+"|"+line.split("\\s")[3]+"|");
	    	if (reduced_RelationSize==false) {
		    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
		    	int rel = relationsWordNum.get(line.split("\\s")[1]);
		    	int e2 = entitiesWordNum.get(line.split("\\s")[2]); 
		    	int label = Integer.parseInt(line.split("\\s")[3]);
		    	devDataNumTripple.put(dev_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
		    	devTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
		    	line = br.readLine();
		    	dev_tripple_counter++;
	    	}else{
	    		if (line.split("\\s")[1].equals(relationsNumWord.get(0))) {
	    			int e1 = entitiesWordNum.get(line.split("\\s")[0]);
			    	int rel = relationsWordNum.get(line.split("\\s")[1]);
			    	int e2 = entitiesWordNum.get(line.split("\\s")[2]); 
			    	int label = Integer.parseInt(line.split("\\s")[3]);
			    	devDataNumTripple.put(dev_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
			    	devTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
			    	line = br.readLine();
			    	dev_tripple_counter++;
				}else{
					line = br.readLine();
				}
	    	}
		}   
	    br.close();
	    System.out.println(dev_tripple_counter +" Dev Examples loaded...");
	}
	
	public void loadTestDataTripplesE1rE2Label(String path) throws IOException{
		//get test data
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int test_tripple_counter = 0;
	    while (line != null) {
	    	if (reduced_RelationSize==false) {
		    	//System.out.println("line: "+line.split("\\s")[0]+"|"+line.split("\\s")[1]+"|"+line.split("\\s")[2]+"|"+line.split("\\s")[3]+"|");
		    	int e1 = entitiesWordNum.get(line.split("\\s")[0]);
		    	int rel = relationsWordNum.get(line.split("\\s")[1]);
		    	int e2 = entitiesWordNum.get(line.split("\\s")[2]); 
		    	int label = Integer.parseInt(line.split("\\s")[3]);
		    	testDataNumTripple.put(test_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
		    	testTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
		    	line = br.readLine();
		    	test_tripple_counter++;
	    	}else{
	    		if (line.split("\\s")[1].equals(relationsNumWord.get(0))) {
	    			int e1 = entitiesWordNum.get(line.split("\\s")[0]);
			    	int rel = relationsWordNum.get(line.split("\\s")[1]);
			    	int e2 = entitiesWordNum.get(line.split("\\s")[2]); 
			    	int label = Integer.parseInt(line.split("\\s")[3]);
			    	testDataNumTripple.put(test_tripple_counter,new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
			    	testTripples.add(new Tripple(e1, line.split("\\s")[0], rel, line.split("\\s")[1], e2, line.split("\\s")[2],label ));
			    	line = br.readLine();
			    	test_tripple_counter++;
	    		}else{
					line = br.readLine();
				}
	    	}
	    		
		}   
	    br.close();
	    System.out.println(test_tripple_counter +" Test Examples loaded...");

	}
	
	public void loadWordVectorsFromMatFile(String path, boolean optimizedLoad){
		//load word vectors with a dimension of 100 from a matlab mat file
		try {
			MatFileReader matfilereader = new MatFileReader(path);
			//words
			MLCell words_mat = (MLCell) matfilereader.getMLArray("words");
			//word embeddings
			MLArray wordvectors_mat = (MLArray) matfilereader.getMLArray("We");
			MLDouble mlArrayDouble = (MLDouble) wordvectors_mat;
			System.out.println("mlArrayDouble"+mlArrayDouble.getM()+"|"+mlArrayDouble.getN()+"|"+mlArrayDouble.getNDimensions()+"|"+mlArrayDouble.getSize());
			//INDArray wordvectormatrix = Nd4j.zeros(100,1); //100 dimension of word embeddings
			String word;
			int wvCounter=0;
			for (int i = 0; i < mlArrayDouble.getSize()/100; i++) {
				//System.out.println("word: "+words_mat.get(i).contentToString().substring(7,words_mat.get(i).contentToString().lastIndexOf("'")));				
				//load word vector
				word = words_mat.get(i).contentToString().substring(7,words_mat.get(i).contentToString().lastIndexOf("'"));
				vocab.add("unknown");
				if (optimizedLoad==true) {
					//only load word vectors which are needed for entity vectors
					if (vocab.contains(word)) { //look up if there is an entity with this word
						vocabNumWord.put(wvCounter, word);
						vocabWordNum.put(word, wvCounter);
						INDArray wordvector = Nd4j.zeros(100,1);
						for (int j = 0; j < 100; j++) {
							wordvector.put(j, 0, mlArrayDouble.get(i, j));	
						}
						
						worvectorNumVec.put(wvCounter, wordvector);
						worvectorWordVec.put(word, wordvector);	
						wvCounter++;
					}
					
				}else{

					vocabNumWord.put(wvCounter, word);
					vocabWordNum.put(word, wvCounter);
					INDArray wordvector = Nd4j.ones(100,1);
					for (int j = 0; j < 100; j++) {
						wordvector.put(j, 0, mlArrayDouble.get(i, j));	
					}
					worvectorNumVec.put(wvCounter, wordvector);
					worvectorWordVec.put(word, wordvector);	
					//System.out.println(word+": "+wordvector);
					wvCounter++;
				}
								
			}
			numOfWords = worvectorNumVec.size();
			System.out.println(worvectorNumVec.size() +" Word Vectors loaded... Counter: "+wvCounter);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		//create a word vector matrix
		wordVectorMaxtrixLoaded = Nd4j.zeros(100, numOfWords);
		for (int i = 0; i < numOfWords; i++) {
			//System.out.println("worvectorNumVec.get(i): "+worvectorNumVec.get(i));
			wordVectorMaxtrixLoaded.putColumn(i, worvectorNumVec.get(i));
		}
		//System.out.println("wordVectorMaxtrixLoaded col 1: "+wordVectorMaxtrixLoaded.getColumn(1));
		//System.out.println("word vector matrix ready..."+wordVectorMaxtrixLoaded);

	}
	
	public INDArray createVectorsForEachEntityByWordVectors(){
		//Method doesnt used the new in readed word vectors
		INDArray entity_vectors = Nd4j.zeros(embeddings_size,numOfentities);
		for (int i = 0; i < entitiesNumWord.size(); i++) {
			String entity_name; //clear name without _  __name_ -> name
			try {
				entity_name = entitiesNumWord.get(i).substring(2, entitiesNumWord.get(i).lastIndexOf("_"));
			} catch (Exception e) {
				entity_name =entitiesNumWord.get(i).substring(2);			
			}
			//System.out.println("entity: "+entitiesNumWord.get(i)+" | word: "+entity_name);			

			if (entity_name.contains("_")) { //whitespaces are _
				//Entity conains of more than one word
				INDArray entityvector = Nd4j.zeros(embeddings_size, 1);
				int counterOfWordVecs = 0;

				for (int j = 0; j < entity_name.split("_").length; j++) {
					try {
						entityvector = entityvector.add(worvectorWordVec.get(entity_name.split("_")[j]));
					} catch (Exception e) {
						//if no word vector available, use "unknown" word vector
						entityvector = entityvector.add(worvectorWordVec.get("unknown"));
					}
					counterOfWordVecs++;
				}
				
				entityvector = entityvector.div(counterOfWordVecs);
				entity_vectors.putColumn(i, entityvector);
			}else{
				// Entity contains of only one word
				/*if (entity_name.equals("absorption") | entity_name.equals("centering")) {
					System.out.println("entity_name: "+entity_name +"wv: "+worvectorWordVec.get(entity_name));
				}*/
				
				try {
					entity_vectors.putColumn(i, worvectorWordVec.get(entity_name));
				} catch (Exception e) {
					// if no word vector available, use "unknown" word vector
					entity_vectors.putColumn(i, worvectorWordVec.get("unknown"));
				}			
			}		
		}
		return entity_vectors;
	}
	
	public void loadGermanDewack100Vectors(String datapath) throws IOException{
		/*Load translation list
		__city_1|stadt
		__jurisprudence_2|gesetz*/
		
		FileReader fr = new FileReader(datapath+"translated_entities.txt");
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int entity_counter = 0;
	    while (line != null) {
	    	//System.out.println(line.split("\\|")[0]+"-"+line.split("\\|")[1]);
		    entitiesWordtranslateWord.put(line.split("\\|")[0], line.split("\\|")[1]);
		    entitiesNumtranslateWord.put(entity_counter, line.split("\\|")[1]);
		    
		    // Generating a vocabulary for german words based on entities
		    for (int i = 0; i < line.split("\\_").length; i++) {
		    	 if (!vocabDE.contains(line.split("\\_")[i])) {
		    		 vocabDE.add(line.split("\\_")[i]);
				}
			}
	    	line = br.readLine();
	    	entity_counter++;
		}   
	    br.close();
	    vocabDE.add("unbekannt");
	    
	    System.out.println(entity_counter + "Entity translation loaded with a vocab size of: "+vocabDE.size());
	    
		//Load german word vectors
		fr = new FileReader(datapath+"dewac_vectors100.bin");
		br = new BufferedReader(fr);
		//1st line: get information about vectors
		line = br.readLine();
	    //System.out.println(line.split("\\s")[0]+" german word vectors available with dimension size: " + line.split("\\s")[1]);
	    line = br.readLine();
	    //System.out.println(""+line);
	    int wvCounterDe = 0;
	    String word;
	    while (line != null) {
	    	word = line.split("\\s")[0];
	    	if (wvCounterDe<100) {
				//System.out.println("word: "+word);
			}
	    	if (vocabDE.contains(word)) {
				//load wordvector
	    		vocabNumWordDE.put(wvCounterDe, word);
				vocabWordNumDE.put(word, wvCounterDe);
	    		INDArray wv = Nd4j.create(100,1);
	    		for (int i = 1; i < embeddings_size+1; i++) {
	    			wv.putScalar(i-1, Double.parseDouble(line.split("\\s")[i]));
				}
	    		worvectorWordVecDE.put(word, wv);
	    		worvectorNumVecDE.put(wvCounterDe,wv);
	    		wvCounterDe++;
	    	}
	    	line = br.readLine();
		}
	    br.close();
	    
	    numOfWords = worvectorNumVecDE.size();
		System.out.println(worvectorNumVecDE.size() +" geman Word Vectors loaded... Counter: "+wvCounterDe);
		
		//quick and ... fix
		vocabNumWord = vocabNumWordDE;
		vocabWordNum = vocabWordNumDE;
		
    	//create a word vector matrix
		wordVectorMaxtrixLoaded = Nd4j.zeros(100, numOfWords);
		for (int i = 0; i < numOfWords; i++) {
			//same as for english word vectors, because it is used in the cost function to calculate the entity vectors
			wordVectorMaxtrixLoaded.putColumn(i, worvectorNumVecDE.get(i));
			//wordVectorMaxtrixLoaded reduce values
		}   
		System.out.println("word vector matrix with german words is ready..."+wordVectorMaxtrixLoaded);
	}
	
	public INDArray createEntityVectorsByWordEmbeddings(INDArray updatedWVMatrix){
		INDArray entity_vectors = Nd4j.zeros(embeddings_size,numOfentities);
		int wv_not_found_counter =0;
		for (int i = 0; i < numOfentities; i++) {
			String entity_name=null; //clear name without _  __name_ -> name
			if(german ==false){
				try {
					entity_name = entitiesNumWord.get(i).substring(2, entitiesNumWord.get(i).lastIndexOf("_"));
				} catch (Exception e) {
					entity_name =entitiesNumWord.get(i).substring(2);			
				}
				//System.out.println("entity: "+entitiesNumWord.get(i)+" | word: "+entity_name);
			}else{
				entity_name = entitiesNumtranslateWord.get(i);
			}
			
			if (entity_name.contains("_")) { //whitespaces are _
				//Entity conains of more than one word
				INDArray entityvector = Nd4j.zeros(embeddings_size, 1);
				int counterOfWordVecs = 0;
				for (int j = 0; j <entity_name.split("_").length; j++) {
					try {
						//System.out.println("Entity "+i+": WV column: "+vocabWordNum.get(entity_name.split("_")[j]));
						//System.out.println("WV: "+updatedWVMatrix.getColumn(vocabWordNum.get(entity_name.split("_")[j])));
						entityvector = entityvector.add(updatedWVMatrix.getColumn(vocabWordNum.get(entity_name.split("_")[j])));
					} catch (Exception e) {
						//if no word vector available, use "unknown" word vector
						wv_not_found_counter++;
						if(german ==false){
							//System.out.println("entity_name.split()[j]): "+entity_name.split("_")[j]);
							entityvector = entityvector.add(updatedWVMatrix.getColumn(vocabWordNum.get("unknown")));
							/*System.out.println("19503: "+vocabNumWord.get(19503));
							System.out.println("31157: "+vocabNumWord.get(31157));
							System.out.println("Entity "+i+": WV column: UNKNOWN"+vocabWordNum.get("unknown"));
							System.out.println("WV: "+updatedWVMatrix.getColumn(vocabWordNum.get("unknown")));*/
						}else{
							entityvector = entityvector.add(updatedWVMatrix.getColumn(vocabWordNum.get("unbekannt")));
						}
					}
					counterOfWordVecs++;
				}
				entityvector = entityvector.div(counterOfWordVecs);
				entity_vectors.putColumn(i, entityvector);
			}else{
				// Entity contains of only one word
				try {
					entity_vectors.putColumn(i, updatedWVMatrix.getColumn(vocabWordNum.get(entity_name)));
					//System.out.println("Entity "+i+": WV column: "+vocabWordNum.get(entity_name));
					//System.out.println("WV: "+updatedWVMatrix.getColumn(vocabWordNum.get(entity_name)));
				} catch (Exception e) {
					// if no word vector available, use "unknown" word vector
					wv_not_found_counter++;
					//System.out.println("entity_name: "+entity_name);
					if(german ==false){
						entity_vectors.putColumn(i, updatedWVMatrix.getColumn(vocabWordNum.get("unknown")));
						/*System.out.println("19503: "+vocabNumWord.get(19503));
						System.out.println("Entity "+i+": WV column: UNKNOWN"+vocabWordNum.get("unknown"));
						System.out.println("WV: "+updatedWVMatrix.getColumn(vocabWordNum.get("unknown")));*/
					}else{
						entity_vectors.putColumn(i, updatedWVMatrix.getColumn(vocabWordNum.get("unbekannt")));
					}
				}			
			}		
		}
		System.out.println("Entity Vectors created. For "+wv_not_found_counter+" words were no Word Embedding found, instead used Word -unknown- ");
		return entity_vectors;
	}
	
	public int entityLength(int entityIndexNum){
		//of how much words contains this entity
		// return: 1 means 1 word | -3 because of other _		
		try {
			return entitiesNumWord.get(entityIndexNum).split("_").length-3;
		} catch (Exception e) {
			// TODO: handle exception
			System.out.println("entityIndexNum: "+entityIndexNum+" | "+entitiesNumWord.get(entityIndexNum)+" | false vocab word by entitynum: "+vocabNumWord.get(entityIndexNum));
			return 1;
		}
		
		
	}

	public ArrayList<Tripple> getDevTripples() {
		return devTripples;
	}
	public ArrayList<Tripple> getTestTripples() {
		return testTripples;
	}
	public int getNumOfWords() {
		return numOfWords;
	}

	public int getNumOfRelations() {
		return numOfRelations;
	}
	
	public INDArray getWordVectorMaxtrixLoaded() {
		return wordVectorMaxtrixLoaded;
	}
	public INDArray getEntitiy1IndexNumbers(ArrayList<Tripple> list){
		//number is corresponding to column in entityvectors matrix
		INDArray e1_list = Nd4j.create(list.size());
		for (int i = 0; i < list.size(); i++) {
			e1_list.putScalar(i, list.get(i).getIndex_entity1());
		}
		return e1_list;
	}
	public INDArray getEntitiy2IndexNumbers(ArrayList<Tripple> list){
		//number is corresponding to column in entityvectors matrix
		INDArray e2_list = Nd4j.create(list.size());
		for (int i = 0; i < list.size(); i++) {
			e2_list.putScalar(i, list.get(i).getIndex_entity2());
		}
		return e2_list;
	}
	public INDArray getRelIndexNumbers(ArrayList<Tripple> list){
		//number is corresponding to column in entityvectors matrix
		INDArray rel_list = Nd4j.create(list.size());
		for (int i = 0; i < list.size(); i++) {
			rel_list.putScalar(i, list.get(i).getIndex_relation());
		}
		return rel_list;
	}
	public INDArray getEntitiy3IndexNumbers(ArrayList<Tripple> list){
		//number is corresponding to column in entityvectors matrix
		INDArray e3_list = Nd4j.create(list.size());
		for (int i = 0; i < list.size(); i++) {
			e3_list.putScalar(i, list.get(i).getIndex_entity3_corrupt());
		}
		return e3_list;
	}
	public int[] getWordIndexes(int entityIndex){
		int[] wordIndexes = new int[entityLength(entityIndex)];
		if (entityLength(entityIndex)==0) {
			//System.out.println("+++++ "+entitiesNumWord.get(entityIndex) +" entityLength(entityIndex)"+entityLength(entityIndex));
			//exception for corrupt training data: entityIndexNum: 9847 | __2 |
			wordIndexes = new int[1];
		}
		
		// get words of entity	
		String entity_name; //clear name without _  __name_ -> name
		try {
			entity_name = entitiesNumWord.get(entityIndex).substring(2, entitiesNumWord.get(entityIndex).lastIndexOf("_"));
		} catch (Exception e) {
			entity_name =entitiesNumWord.get(entityIndex).substring(2);			
		}
		
		// get word indexes
		if (entity_name.contains("_")) { //whitespaces are _
			//Entity conains of more than one word
			for (int j = 0; j <entity_name.split("_").length; j++) {
				try {
					wordIndexes[j] = vocabWordNum.get(entity_name.split("_")[j]);
				} catch (Exception e) {
					//if no word vector available, use "unknown" word vector
					wordIndexes[j] = vocabWordNum.get("unknown");
				}			
			}
		}else{
			// Entity conains of only one word
			try {
				wordIndexes[0] = vocabWordNum.get(entity_name);
			} catch (Exception e) {
				// if no word vector available, use "unknown" word vector
				wordIndexes[0] = vocabWordNum.get("unknown");
			}			
		}	
		//System.out.println("wordIndexes: "+ wordIndexes.length + " | "+wordIndexes[0]);		
		return wordIndexes;
	}
	
	

}
