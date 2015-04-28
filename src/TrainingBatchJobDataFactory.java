

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TrainingBatchJobDataFactory{
	ArrayList<Tripple> trainingTripples = new ArrayList<Tripple>(); //data for learning the paramters, without labels
	ArrayList<Tripple> devTripples = new ArrayList<Tripple>(); //data for learning the threshold, with labels
	ArrayList<Tripple> testTripples = new ArrayList<Tripple>(); //data for learning the threshold, with labels
	int batch_size;
	int corrupt_size;
	int numOfentities;
	// contains a random collection of from trainingTripples for learning the paramters
	// a random collection inclusive corrupt examples is created by evoking generateNewTrainingBatchJob()
	ArrayList<Tripple> batchjob = new ArrayList<Tripple>(); 
	
	
	
	TrainingBatchJobDataFactory(ArrayList<Tripple> _trainingTripples, int _batch_size, int _corrupt_size, int _numOfentities ){
		trainingTripples = _trainingTripples;
		batch_size = _batch_size;
		corrupt_size = _corrupt_size;
		numOfentities = _numOfentities;		
		//set corrupt 3. tripple
		for (Tripple tripple : _trainingTripples) {	
			Random rand = new Random();
			int random_corrupt_entity = rand.nextInt((corrupt_size - 1) + 1) + 1; //rand.nextInt((max - min) + 1) + min		
			tripple.setIndex_entity3_corrupt(random_corrupt_entity);
		}

		/*for (int i = 0; i < corrupt_size; i++) {
			batchjob.addAll(batchjob);
		}
		System.out.println("batchjob size: "+batchjob.size());*/
		
		/*System.out.println("shuffled tripples: " +tripples.get(0).index_entity1);
		System.out.println("shuffled tripples: " +tripples.get(1).index_entity1);
		System.out.println("shuffled tripples: " +tripples.get(2).index_entity1);
		System.out.println("shuffled tripples: " +tripples.get(3).index_entity1);
		System.out.println("shuffled tripples e3: " +tripples.get(0).index_entity3_corrupt);
		System.out.println("shuffled tripples e3: " +tripples.get(1).index_entity3_corrupt);*/
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

}
