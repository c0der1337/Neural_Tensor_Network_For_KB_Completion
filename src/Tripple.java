

import java.util.Random;

import org.apache.commons.math3.random.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Tripple {
	private String entity1, relation, entity2, entity3_corrupt;
	private int index_entity1, index_relation, index_entity2, index_entity3_corrupt;
	private INDArray wordvector_entity1, wordvector_entity2;
	private int label; // for dev data -1=false, 1=true
	private double score; // for dev data to save the score
	private int prediction; // for test data -1=false, 1=true
	
	Tripple(String entity1, String relation, String entity2){
		this.entity1 = entity1;
		this.relation = relation;
		this.entity1 = entity1;
	}
	Tripple(int index_entity1, String entity1, int index_relation, String relation, int index_entity2, String entity2){	
		this.index_entity1 = index_entity1;
		this.index_relation = index_relation;
		this.index_entity2 = index_entity2;
		this.entity1 = entity1;
		this.relation = relation;
		this.entity2 = entity2;
	}
	Tripple(int index_entity1, String entity1, int index_relation, String relation, int index_entity2, String entity2, int label){	
		this.index_entity1 = index_entity1;
		this.index_relation = index_relation;
		this.index_entity2 = index_entity2;
		this.entity1 = entity1;
		this.relation = relation;
		this.entity2 = entity2;
		this.label = label;
	}
	
	
	//Getter und Setter
	public String getEntity1() {
		return entity1;
	}
	public void setEntity1(String entity1) {
		this.entity1 = entity1;
	}
	public String getRelation() {
		return relation;
	}
	public void setRelation(String relation) {
		this.relation = relation;
	}
	public String getEntity2() {
		return entity2;
	}
	public void setEntity2(String entity2) {
		this.entity2 = entity2;
	}
	public int getIndex_entity1() {
		return index_entity1;
	}
	public void setIndex_entity1(int index_entity1) {
		this.index_entity1 = index_entity1;
	}
	public int getIndex_relation() {
		return index_relation;
	}
	public void setIndex_relation(int index_relation) {
		this.index_relation = index_relation;
	}
	public int getIndex_entity2() {
		return index_entity2;
	}
	public void setIndex_entity2(int index_entity2) {
		this.index_entity2 = index_entity2;
	}
	public INDArray getWordvector_entity1() {
		return wordvector_entity1;
	}
	public void setWordvector_entity1(INDArray wordvector_entity1) {
		this.wordvector_entity1 = wordvector_entity1;
	}
	public INDArray getWordvector_entity2() {
		return wordvector_entity2;
	}
	public void setWordvector_entity2(INDArray wordvector_entity2) {
		this.wordvector_entity2 = wordvector_entity2;
	}

	public void setEntity3_corrupt(String entity3_corrupt) {
		this.entity3_corrupt = entity3_corrupt;
	}
	public int getIndex_entity3_corrupt() {
		return index_entity3_corrupt;
	}
	public int getIndex_entity3_corruptRANDOM(int maxNumOfEntity) {	
		Random rand = new Random();
		int random_corrupt_entity = rand.nextInt((maxNumOfEntity - 1) + 1) + 1;
		return random_corrupt_entity;
	}
	public void setIndex_entity3_corrupt(int index_entity3_corrupt) {
		this.index_entity3_corrupt = index_entity3_corrupt;
	}
	public int getPrediction() {
		return prediction;
	}


	public void setPrediction(int prediction) {
		this.prediction = prediction;
	}
	public int getLabel() {
		return label;
	}	
	public double getScore() {
		return score;
	}
	public void setScore(double score) {
		this.score = score;
	}


	@Override
	public String toString() {
		// TODO Auto-generated method stub
		
		return "Tripple: e1:"+entity1+"("+index_entity1+") "+relation+"("+index_relation+") "+"e2: "+entity2+"("+index_entity2+") ";
	}

}
