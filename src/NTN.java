

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;



import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.activation.*;

import edu.umass.nlp.optimize.IDifferentiableFn;
import edu.umass.nlp.utils.BasicPair;
import edu.umass.nlp.utils.IPair;

public class NTN implements IDifferentiableFn {
	
	int embeddingSize; 			// 100 size of a single word vector
	int numberOfEntities;
	int numberOfRelations;
	int batchSize; 				// training batch size
	int sliceSize;				// 3 number of slices in tensor
	int word_indicies;
	int activation_function;	// 0 - tanh, 1 - sigmoid
	float lamda;				// regulariization parameter
	INDArray theta_inital; 		// stacked paramters after initlization of NTN
	int dimension_for_minimizer;// dimensions / size of theta for the minimizer
   
	//Network Parameters - Integer identifies the relation, there are different parameters for each relation
	HashMap<Integer, INDArray> w;
	HashMap<Integer, INDArray> v;
	HashMap<Integer, INDArray> b;
	HashMap<Integer, INDArray> u;
	
	// For ComputeAt Implementation
	static HashMap<Integer, String> entitiesNumWord_global;
	static HashMap<String, INDArray> worvectorWordVec_global;
	TrainingBatchJobDataFactory tbj;

	NTN(	int embeddingSize, 			// 100 size of a single word vector
			int numberOfEntities,		
			int numberOfRelations,		// number of different relations
			int batchSize, 				// training batch size original: 20.000
			int sliceSize, 				// 3 number of slices in tensor
			int activation_function,	// 0 - tanh, 1 - sigmoid
			float lamda){  				// regulariization parameter
		
		
		//Initialize word vectors randomly with each element in the range [-r, r] 
		
		//Load Word Vectors
		
		// Initialize the parameters of the network
		w = new HashMap<Integer, INDArray>(); 	
		v = new HashMap<Integer, INDArray>();
		b = new HashMap<Integer, INDArray>();
		u = new HashMap<Integer, INDArray>();
		
		this.embeddingSize = embeddingSize;
		this.numberOfEntities = numberOfEntities;
		this.numberOfRelations = numberOfRelations;
		this.batchSize = batchSize;
		this.sliceSize = sliceSize;
		this.lamda = lamda;
		
		for (int i = 0; i < numberOfRelations; i++) {
			w.put(i, Nd4j.rand(new int[]{embeddingSize,embeddingSize,sliceSize}));
			// TODO better initalization of w
			v.put(i, Nd4j.zeros(2*embeddingSize,sliceSize));
			b.put(i, Nd4j.zeros(1,sliceSize));
			u.put(i, Nd4j.ones(sliceSize,1));				
		}
		
		// Unroll the parameters into a vector		
		theta_inital = parametersToStack(w, v, b, u);		
		dimension_for_minimizer = theta_inital.data().asDouble().length;
		
	}
	
	public INDArray getTheta_inital() {
		return theta_inital;
	}

	public ArrayList neuralTensorNetworkCost(TrainingBatchJobDataFactory tbj, HashMap<Integer, String> entitiesNumWord,HashMap<String, INDArray> worvectorWordVec){
		// NOT MORE IN USE ---> go to computeAt()
		// This was the inital cost function before implementing the interface IDifferentiableFn to minimizw via LBFGS
		
		//Get stack of network parameters
		
		// Initialize entity vectors and their gradient as matrix of zeros
		INDArray entity_vectors = Nd4j.zeros(embeddingSize, numberOfEntities);
		INDArray entity_vectors_grad = Nd4j.zeros(embeddingSize, numberOfEntities);
		
		//Assign entity vectors to be the mean of word vectors involved
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
		
		//Add the word vectors to the tbj - necessay?
		
		
		// Initialize cost as zero
		Float cost = 0F;
		
		//Make dictionaries for parameter gradients
		HashMap<Integer, INDArray> w_grad = new HashMap<Integer, INDArray>();
		HashMap<Integer, INDArray> v_grad = new HashMap<Integer, INDArray>();
		HashMap<Integer, INDArray> u_grad = new HashMap<Integer, INDArray>();
		HashMap<Integer, INDArray> b_grad = new HashMap<Integer, INDArray>();
		
		for (int r = 0; r < numberOfRelations; r++) {
			
			//Make a list of examples for the ith relation
			ArrayList<Tripple> tripplesOfRelationR = tbj.getBatchJobTripplesOfRelation(r);
			System.out.println(tripplesOfRelationR.size()+" Trainingsexample for relation r="+r);
			
			//Get entity lists for examples of ith relation
			INDArray wordvectors_for_entities1 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
			INDArray wordvectors_for_entities2 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
			INDArray wordvectors_for_entities3 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});		
			INDArray wordvectors_for_entities1_neg = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
			INDArray wordvectors_for_entities2_neg = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
			
			// Get entity vectors for examples of the ith relation
			for (int j = 0; j < tripplesOfRelationR.size(); j++) {
				Tripple tripple = tripplesOfRelationR.get(j);
				wordvectors_for_entities1.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity1()));
				wordvectors_for_entities2.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity2()));
				wordvectors_for_entities3.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity3_corrupt()));
			}
			arrayInfo(wordvectors_for_entities2, "wordvectors_for_entities2");
			// Choose entity vectors and lists based on random
			if (Math.random()>0.5) {
				wordvectors_for_entities1_neg = wordvectors_for_entities1;
				wordvectors_for_entities2_neg = wordvectors_for_entities3;
			}else{
				wordvectors_for_entities1_neg = wordvectors_for_entities3;
				wordvectors_for_entities2_neg = wordvectors_for_entities2;
			}
			arrayInfo(wordvectors_for_entities1, "wordvectors_for_entities1");
				
			// Initialize pre-activations of the tensor network as matrix of zeros
			//System.out.println("Number of training tripples for this relation: "+tripplesOfRelationR.size());
			INDArray preactivation_pos = Nd4j.zeros(sliceSize, tripplesOfRelationR.size());
			INDArray preactivation_neg = Nd4j.zeros(sliceSize, tripplesOfRelationR.size());

			//Add contribution of term containing W
			INDArray wOfThisRelation = w.get(r);
			for (int slice = 0; slice < sliceSize; slice++) {
				INDArray sliceOfW = getSliceOfaTensor(wOfThisRelation, slice);		
				INDArray dotproduct = sliceOfW.mmul(wordvectors_for_entities2);
				INDArray dotproduct_neg = sliceOfW.mmul(wordvectors_for_entities2_neg);
				INDArray result = Nd4j.sum(wordvectors_for_entities1.mul(dotproduct), 0);
				INDArray result_neg = Nd4j.sum(wordvectors_for_entities1_neg.mul(dotproduct_neg), 0);
				preactivation_pos.putRow(slice, result);
				preactivation_neg.putRow(slice, result_neg);
			}
			//Add contribution of terms containing V and b
			INDArray bOfThisRelation_T = b.get(r).transpose();
			INDArray vOfThisRelation_T= v.get(r).transpose();
			INDArray vstack = Nd4j.vstack(wordvectors_for_entities1, wordvectors_for_entities2);	
			INDArray dotproduct = vOfThisRelation_T.mmul(vstack);;
			INDArray dotproduct_neg = vOfThisRelation_T.mmul(Nd4j.vstack(wordvectors_for_entities1_neg, wordvectors_for_entities2_neg));
			INDArray temp = dotproduct.addColumnVector(bOfThisRelation_T);
			preactivation_pos = preactivation_pos.add(temp);
			preactivation_neg = preactivation_neg.add(dotproduct_neg.addColumnVector(bOfThisRelation_T));
			
			// Apply the activation function
			//INDArray activation_pos = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", preactivation_pos));
			//INDArray activation_neg = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", preactivation_neg));			
			INDArray activation_pos = Transforms.sigmoid(preactivation_pos);
			INDArray activation_neg = Transforms.sigmoid(preactivation_neg);
			//System.out.println("activation: " +activation_pos);
			// Calculate scores for positive and negative examples
			INDArray score_pos = u.get(r).transpose().mmul(activation_pos);
			INDArray score_neg = u.get(r).transpose().mmul(activation_neg);
			//System.out.println("score_pos: "+score_pos);
			//System.out.println("score_neg: "+score_neg);

			//Filter for training examples, (that already predicted correct and dont need to be in account for further optimization of the paramters)			
			INDArray wrong_filter = Nd4j.ones(score_pos.columns(),1);
			//arrayInfo(wrong_filter, "wrong_filter");
			
			for (int i = 0; i < score_pos.columns(); i++) {
				//System.out.println("score pos: " + score_pos.getRow(0).getFloat(i)+1 +" > "+score_neg.getRow(0).getFloat(i));
				if (score_pos.getRow(0).getFloat(i)+1 > score_neg.getRow(0).getFloat(i)) {
					wrong_filter.put(i,0, 1.0);
				}else{
					wrong_filter.put(i,0, 0.0);
				}
			}
			//System.out.println("wrong filter: "+wrong_filter);
			
			//Add max-margin term to the cost
			// System.out.println("1: "+ Nd4j.sum(score_pos.sub(score_neg.add(1))));
			// System.out.println("2: "+ Nd4j.sum((score_pos.sub(score_neg)).add(Nd4j.ones(1,score_pos.columns()))));
			cost = Nd4j.sum(wrong_filter.mul((score_pos.sub(score_neg)).add(Nd4j.ones(1,score_pos.columns())))).getFloat(0);
			// System.out.println("cost: " + cost);
			
			//Initialize 'W[i]' and 'V[i]' gradients as matrix of zero
			w_grad.put(r, Nd4j.zeros(new int[]{embeddingSize,embeddingSize,sliceSize}));
			v_grad.put(r, Nd4j.zeros(2*embeddingSize,sliceSize));
			
			//Number of examples contributing to error
			int numOfWrongExamples = Nd4j.sum(wrong_filter).getInt(0);
			//System.out.println("numOfWrongExamples: "+numOfWrongExamples);
			
			//Filter matrices using 'wrong_filter'		
			activation_pos = activation_pos.mulRowVector(wrong_filter);
			activation_neg = activation_neg.mulRowVector(wrong_filter);
			INDArray wordvectors_for_entities1_rel = wordvectors_for_entities1.mulRowVector(wrong_filter);
			INDArray wordvectors_for_entities2_rel = wordvectors_for_entities2.mulRowVector(wrong_filter);
			INDArray wordvectors_for_entities1_rel_neg = wordvectors_for_entities1_neg.mulRowVector(wrong_filter);
			INDArray wordvectors_for_entities2_rel_neg = wordvectors_for_entities2_neg.mulRowVector(wrong_filter);
			
			//Filter entity lists using 'wrong_filter'
				
			//Calculate U[i] gradient EXPLANATION WHY U gradient is activation see Socher NNAL Vortrag Slide 51
			u_grad.put(r, Nd4j.sum(activation_pos.sub(activation_neg),1).reshape(sliceSize, 1)) ;
			
			
			//Calculate U * f'(z) terms useful for gradient calculation
			//activation function is sigmoid
			INDArray temp_pos_all = activationDifferential(activation_pos).mulColumnVector(u.get(r));
			INDArray temp_neg_all = activationDifferential(activation_neg).mulColumnVector(u.get(r).neg());
			
			// Calculate 'b[i]' gradient
			b_grad.put(r, Nd4j.sum(temp_pos_all.add(temp_neg_all),1).reshape(1, sliceSize)) ;
			
			// Variables required for sparse matrix calculation
			INDArray values = Nd4j.ones(numOfWrongExamples);
			INDArray rows = Nd4j.arange(0, numOfWrongExamples+1);	
			
			//TODO Calculate spare matrixes
			//waiting for spare matrix lib
			
			INDArray w_grad_for_r = Nd4j.create(embeddingSize,embeddingSize,sliceSize);
			ArrayList<INDArray> w_grad_slices = new ArrayList<INDArray>();
			for (int k = 0; k < sliceSize; k++) {				
				// U * f'(z) values corresponding to one slice
				INDArray temp_pos = temp_pos_all.getRow(k);
				INDArray temp_neg = temp_pos_all.getRow(k);
				
				//Calculate 'k'th slice of 'W[i]' gradient			
				INDArray dot1 = (wordvectors_for_entities1_rel.mulRowVector(temp_pos)).mmul(wordvectors_for_entities2_rel.transpose());
				INDArray dot2 = (wordvectors_for_entities1_rel_neg.mulRowVector(temp_neg)).mmul(wordvectors_for_entities2_rel_neg.transpose());
				INDArray w_grad_k_slice = dot1.add(dot2);
				
				// PRÜFEN, AUSKOMMENTIEREN WENN ND4J SLICE ISSUE BEHOBEN IST: Illegal assignment, must be of same length
				//w_grad_for_r.putSlice(k, w_grad_k_slice);
				//w_grad.put(r, w_grad_for_r);
				w_grad_slices.add(w_grad_k_slice);
				
				//Calculate 'k'th slice of 'V[i]' gradient				
				INDArray eVstack = Nd4j.vstack(wordvectors_for_entities1_rel,wordvectors_for_entities2_rel);
				INDArray eVstack_neg = Nd4j.vstack(wordvectors_for_entities1_rel_neg,wordvectors_for_entities2_rel_neg);
				INDArray temparray = (eVstack.mulRowVector(temp_pos)).add(eVstack_neg.mulRowVector(temp_neg));	
				INDArray sum_v = Nd4j.sum(eVstack.mulRowVector(temp_pos).add(eVstack_neg.mulRowVector(temp_neg)),1);
				v_grad.get(r).putColumn(k, sum_v);
				
				// TODO Add contribution of 'V[i]' term in the entity vectors' gradient				
				/*INDArray vOfThisRelation = v.get(r);
				INDArray kth_slice_of_v = vOfThisRelation.getColumn(k); //slice is the column
				INDArray v_pos = kth_slice_of_v.mmul(temp_pos);
				INDArray v_neg = kth_slice_of_v.mmul(temp_neg);
				
				//entity_vectors_grad = entity_vectors_grad;
				//get all entity vectors for e2 of this relation from this training batch, equal to entity_vectors[:, e2.tolist()]
				
				INDArray e2_entity_vectors_for_r_and_batch;
				INDArray sliceOfW = getSliceOfaTensor(wOfThisRelation, k);		*/
				
				// TODO Add contribution of 'W[i]' term in the entity vectors' gradient
				//TODO - Tensor problematic of ND4J and sparse vectors			
				/*INDArray v_pos_e1 = Nd4j.zeros(embeddingSize, tripplesOfRelationR.size()); //V_pos[:self.embedding_size, :]
				INDArray v_pos_e2 = Nd4j.zeros(embeddingSize, tripplesOfRelationR.size()); //V_pos[self.embedding_size:, :]
				INDArray v_neg_e1 = Nd4j.zeros(embeddingSize, tripplesOfRelationR.size()); //V_neg[:self.embedding_size, :]
				INDArray v_neg_e2 = Nd4j.zeros(embeddingSize, tripplesOfRelationR.size());
				for (int i = 0; i < embeddingSize; i++) {		
					if(i<(embeddingSize)){
						v_pos_e1.put(i, v_pos.getRow(i));
						v_neg_e1.put(i, v_neg.getRow(i));
					}else{
						v_pos_e2.put(i, v_pos.getRow(i));
						v_neg_e2.put(i, v_neg.getRow(i));
					}
				}
				INDArray temp1 = v_pos_e1.mul(wordvectors_for_entities1_rel);
				INDArray temp2 = v_pos_e2.mul(wordvectors_for_entities2_rel);
				INDArray temp3 = v_neg_e1.mul(wordvectors_for_entities1_neg);
				INDArray temp4 = v_neg_e2.mul(wordvectors_for_entities2_neg);
				// !!!! entity_vector_grad = entity_vector_grad.add(temp1.add(temp2).add(3).add(temp4));
				*/

			}
			// Normalize the gradients with the training batch size
			INDArray w_grad_for_r2 = w_grad.get(r);
			INDArray v_grad_for_r = v_grad.get(r);
			w_grad_for_r = w_grad_for_r2.div(batchSize);
			v_grad_for_r = v_grad_for_r.div(batchSize);
			INDArray b_grad_for_r = b_grad.get(r);
			b_grad_for_r = b_grad_for_r.div(batchSize);
			INDArray u_grad_for_r = u_grad.get(r);
			u_grad_for_r = u_grad_for_r.div(batchSize);
			
		}
		
		// Initialize word vector gradients as a matrix of zeros
		INDArray word_vector_grad = Nd4j.zeros(embeddingSize, numberOfEntities);
		
		// TODO Calculate word vector gradients from entity gradients
		/*int entity_len = numberOfWords;
		for (int i = 0; i < numberOfEntities; i++) {
			//TODO
		}*/
		
		// Normalize word vector gradients and cost by the training batch size
		//word_vector_grad = word_vector_grad.div(batchSize);

		cost = cost / batchSize;
		
		// Get unrolled gradient vector
		INDArray theta_grad = parametersToStack(w_grad, v_grad, b_grad, u_grad);
		INDArray theta = parametersToStack(w,v,b,u);
		//Add regularization term to the cost and gradient
		//cost = cost + 0.5*lamda * np.sum(theta * theta)
		cost = cost + (0.5F * (lamda * Nd4j.sum(parametersToStack(w,v,b,u).mul(parametersToStack(w,v,b,u))).getFloat(0)));
		//System.out.println("Overall Cost: "+cost);
		//theta_grad = theta_grad + lamda * theta;
		//System.out.println("theta: "+theta);
		//System.out.println("theta * lamda: "+theta.mul(lamda));
		theta_grad = theta_grad.add(theta.mul(lamda));
		//RETURN: cost, theta_grad
		ArrayList<Object> cost_theta_grad = new ArrayList<>();
		cost_theta_grad.add(cost);
		cost_theta_grad.add(theta_grad);
		return cost_theta_grad;
		
		
	}
	
	public INDArray activationDifferential(INDArray activation){
		//for a sigmoid activation function:
		//Ableitung der sigmoid function f(z) -> f'(z) -> (z * (1 - z))
		// z = activation_pos		
		return activation.mul((Nd4j.ones(activation.rows(), activation.columns()).sub(activation)));
	}
	public void arrayInfo(INDArray array, String name){
		System.out.println("_"+name + ": rows: "+array.rows()+"|cols: "+array.columns());
	}
	public INDArray getSliceOfaTensor(INDArray tensor, int numOfSlice){
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

	
	@Override
	public IPair<Double, double[]> computeAt(double[] _theta) {
		// IN USE, Cost / Loss function 
		//load input paramter(theta) into java variables for further computations
		stackToParameters(convertDoubleArrayToFlattenedINDArray(_theta));
		
		// Initialize entity vectors and their gradient as matrix of zeros
		INDArray entity_vectors = Nd4j.zeros(embeddingSize, numberOfEntities);
		INDArray entity_vectors_grad = Nd4j.zeros(embeddingSize, numberOfEntities);
		
		//Assign entity vectors to be the mean of word vectors involved
		
		entity_vectors= createVectorsForEachEntityByWordVectors(entity_vectors);
		
		//TODO Add the word vectors to the tbj - necessay?
		
		
		// Initialize cost as zero
		Float cost = 0F;
		
		//use hasmaps to store parameter gradients for each relation
		HashMap<Integer, INDArray> w_grad = new HashMap<Integer, INDArray>();
		HashMap<Integer, INDArray> v_grad = new HashMap<Integer, INDArray>();
		HashMap<Integer, INDArray> u_grad = new HashMap<Integer, INDArray>();
		HashMap<Integer, INDArray> b_grad = new HashMap<Integer, INDArray>();
		
		for (int r = 0; r < numberOfRelations; r++) {			
			//Make a list of examples / tripples for the ith relation
			ArrayList<Tripple> tripplesOfRelationR = tbj.getBatchJobTripplesOfRelation(r);
			System.out.println(tripplesOfRelationR.size()+" Trainingsexample for relation r="+r);
			
			//Get entity lists for examples of ith relation
			INDArray wordvectors_for_entities1 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
			INDArray wordvectors_for_entities2 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
			INDArray wordvectors_for_entities3 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});		
			INDArray wordvectors_for_entities1_neg = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
			INDArray wordvectors_for_entities2_neg = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
			
			// Get entity vectors for examples of the ith relation
			for (int j = 0; j < tripplesOfRelationR.size(); j++) {
				Tripple tripple = tripplesOfRelationR.get(j);
				wordvectors_for_entities1.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity1()));
				wordvectors_for_entities2.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity2()));
				wordvectors_for_entities3.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity3_corrupt()));
			}
			//arrayInfo(wordvectors_for_entities2, "wordvectors_for_entities2");
			// Choose entity vectors and lists based on random
			if (Math.random()>0.5) {
				wordvectors_for_entities1_neg = wordvectors_for_entities1;
				wordvectors_for_entities2_neg = wordvectors_for_entities3;
			}else{
				wordvectors_for_entities1_neg = wordvectors_for_entities3;
				wordvectors_for_entities2_neg = wordvectors_for_entities2;
			}
			//arrayInfo(wordvectors_for_entities1, "wordvectors_for_entities1");
				
			// Initialize pre-activations of the tensor network as matrix of zeros
			//System.out.println("Number of training tripples for this relation: "+tripplesOfRelationR.size());
			INDArray preactivation_pos = Nd4j.zeros(sliceSize, tripplesOfRelationR.size());
			INDArray preactivation_neg = Nd4j.zeros(sliceSize, tripplesOfRelationR.size());
	
			//Add contribution of term containing W
			INDArray wOfThisRelation = w.get(r);
			for (int slice = 0; slice < sliceSize; slice++) {
				INDArray sliceOfW = getSliceOfaTensor(wOfThisRelation, slice);		
				INDArray dotproduct = sliceOfW.mmul(wordvectors_for_entities2);
				INDArray dotproduct_neg = sliceOfW.mmul(wordvectors_for_entities2_neg);
				INDArray result = Nd4j.sum(wordvectors_for_entities1.mul(dotproduct), 0);
				INDArray result_neg = Nd4j.sum(wordvectors_for_entities1_neg.mul(dotproduct_neg), 0);
				preactivation_pos.putRow(slice, result);
				preactivation_neg.putRow(slice, result_neg);
			}
			//Add contribution of terms containing V and b
			INDArray bOfThisRelation_T = b.get(r).transpose();
			INDArray vOfThisRelation_T= v.get(r).transpose();
			INDArray vstack = Nd4j.vstack(wordvectors_for_entities1, wordvectors_for_entities2);	
			INDArray dotproduct = vOfThisRelation_T.mmul(vstack);;
			INDArray dotproduct_neg = vOfThisRelation_T.mmul(Nd4j.vstack(wordvectors_for_entities1_neg, wordvectors_for_entities2_neg));
			INDArray temp = dotproduct.addColumnVector(bOfThisRelation_T);
			preactivation_pos = preactivation_pos.add(temp);
			preactivation_neg = preactivation_neg.add(dotproduct_neg.addColumnVector(bOfThisRelation_T));
			
			// Apply the activation function
			//INDArray activation_pos = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", preactivation_pos));
			//INDArray activation_neg = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", preactivation_neg));			
			INDArray activation_pos = Transforms.sigmoid(preactivation_pos);
			INDArray activation_neg = Transforms.sigmoid(preactivation_neg);	
			//System.out.println("activation: " +activation_pos);
			
			// Calculate scores for positive and negative examples
			INDArray score_pos = u.get(r).transpose().mmul(activation_pos);
			INDArray score_neg = u.get(r).transpose().mmul(activation_neg);
			//System.out.println("score_pos: "+score_pos);
			//System.out.println("score_neg: "+score_neg);
	
			//Filter for training examples, (that already predicted correct and dont need to be in account for further optimization of the paramters)			
			INDArray wrong_filter = Nd4j.ones(score_pos.columns(),1);
			//arrayInfo(wrong_filter, "wrong_filter");
			
			//System.out.println("");
			for (int i = 0; i < score_pos.columns(); i++) {
				//System.out.println("score pos: " + score_pos.getRow(0).getFloat(i)+1 +" > "+score_neg.getRow(0).getFloat(i));
				if (score_pos.getRow(0).getFloat(i)+1 > score_neg.getRow(0).getFloat(i)) {
					wrong_filter.put(i,0, 1.0);
				}else{
					wrong_filter.put(i,0, 0.0);
				}
			}
			//System.out.println("wrong filter: "+wrong_filter);
			
			//Add max-margin term to the cost
			//System.out.println("1: "+ Nd4j.sum(score_pos.sub(score_neg.add(1))));
			//System.out.println("2: "+ Nd4j.sum((score_pos.sub(score_neg)).add(Nd4j.ones(1,score_pos.columns()))));
			cost = Nd4j.sum(wrong_filter.mul((score_pos.sub(score_neg)).add(Nd4j.ones(1,score_pos.columns())))).getFloat(0);
			//System.out.println("cost: " + cost);
			
			//Initialize 'W[i]' and 'V[i]' gradients as matrix of zero
			w_grad.put(r, Nd4j.zeros(new int[]{embeddingSize,embeddingSize,sliceSize}));
			v_grad.put(r, Nd4j.zeros(2*embeddingSize,sliceSize));
			
			//Number of examples contributing to error
			int numOfWrongExamples = Nd4j.sum(wrong_filter).getInt(0);
			//System.out.println("numOfWrongExamples: "+numOfWrongExamples);
			
			//Filter matrices using 'wrong_filter'		
			activation_pos = activation_pos.mulRowVector(wrong_filter);
			activation_neg = activation_neg.mulRowVector(wrong_filter);
			INDArray wordvectors_for_entities1_rel = wordvectors_for_entities1.mulRowVector(wrong_filter);
			INDArray wordvectors_for_entities2_rel = wordvectors_for_entities2.mulRowVector(wrong_filter);
			INDArray wordvectors_for_entities1_rel_neg = wordvectors_for_entities1_neg.mulRowVector(wrong_filter);
			INDArray wordvectors_for_entities2_rel_neg = wordvectors_for_entities2_neg.mulRowVector(wrong_filter);
			
			//Filter entity lists using 'wrong_filter'
				
			//Calculate U[i] gradient EXPLANATION WHY U gradient is activation see Socher NNAL Vortrag Slide 51
			u_grad.put(r, Nd4j.sum(activation_pos.sub(activation_neg),1).reshape(sliceSize, 1)) ;
			
			
			//Calculate U * f'(z) terms useful for gradient calculation
			//activation function is sigmoid
			INDArray temp_pos_all = activationDifferential(activation_pos).mulColumnVector(u.get(r));
			INDArray temp_neg_all = activationDifferential(activation_neg).mulColumnVector(u.get(r).neg());
			
			// Calculate 'b[i]' gradient
			b_grad.put(r, Nd4j.sum(temp_pos_all.add(temp_neg_all),1).reshape(1, sliceSize)) ;
			
			// Variables required for sparse matrix calculation
			INDArray values = Nd4j.ones(numOfWrongExamples);
			INDArray rows = Nd4j.arange(0, numOfWrongExamples+1);	
			
			//TODO Calculate sparse matrixes
			//not implemented in ND4j
			
			INDArray w_grad_for_r = Nd4j.create(embeddingSize,embeddingSize,sliceSize);
			ArrayList<INDArray> w_grad_slices = new ArrayList<INDArray>();
			for (int k = 0; k < sliceSize; k++) {				
				// U * f'(z) values corresponding to one slice
				INDArray temp_pos = temp_pos_all.getRow(k);
				INDArray temp_neg = temp_pos_all.getRow(k);
				
				//Calculate 'k'th slice of 'W[i]' gradient			
				INDArray dot1 = (wordvectors_for_entities1_rel.mulRowVector(temp_pos)).mmul(wordvectors_for_entities2_rel.transpose());
				INDArray dot2 = (wordvectors_for_entities1_rel_neg.mulRowVector(temp_neg)).mmul(wordvectors_for_entities2_rel_neg.transpose());
				INDArray w_grad_k_slice = dot1.add(dot2);
				
				// PRÜFEN, AUSKOMMENTIEREN WENN ND4J SLICE ISSUE BEHOBEN IST: Illegal assignment, must be of same length
				//w_grad_for_r.putSlice(k, w_grad_k_slice);
				//w_grad.put(r, w_grad_for_r);
				w_grad_slices.add(w_grad_k_slice);
				
				//Calculate 'k'th slice of 'V[i]' gradient				
				INDArray eVstack = Nd4j.vstack(wordvectors_for_entities1_rel,wordvectors_for_entities2_rel);
				INDArray eVstack_neg = Nd4j.vstack(wordvectors_for_entities1_rel_neg,wordvectors_for_entities2_rel_neg);
				INDArray temparray = (eVstack.mulRowVector(temp_pos)).add(eVstack_neg.mulRowVector(temp_neg));	
				INDArray sum_v = Nd4j.sum(eVstack.mulRowVector(temp_pos).add(eVstack_neg.mulRowVector(temp_neg)),1);
				v_grad.get(r).putColumn(k, sum_v);
				
				// TODO Add contribution of 'V[i]' term in the entity vectors' gradient				
				/*INDArray vOfThisRelation = v.get(r);
				INDArray kth_slice_of_v = vOfThisRelation.getColumn(k); //slice is the column
				INDArray v_pos = kth_slice_of_v.mmul(temp_pos);
				INDArray v_neg = kth_slice_of_v.mmul(temp_neg);
				
				//entity_vectors_grad = entity_vectors_grad;
				//get all entity vectors for e2 of this relation from this training batch, equal to entity_vectors[:, e2.tolist()]
				
				INDArray e2_entity_vectors_for_r_and_batch;
				INDArray sliceOfW = getSliceOfaTensor(wOfThisRelation, k);		*/
				
				// TODO Add contribution of 'W[i]' term in the entity vectors' gradient
				//TODO - Tensor problematic of ND4J and sparse vectors			
				/*INDArray v_pos_e1 = Nd4j.zeros(embeddingSize, tripplesOfRelationR.size()); //V_pos[:self.embedding_size, :]
				INDArray v_pos_e2 = Nd4j.zeros(embeddingSize, tripplesOfRelationR.size()); //V_pos[self.embedding_size:, :]
				INDArray v_neg_e1 = Nd4j.zeros(embeddingSize, tripplesOfRelationR.size()); //V_neg[:self.embedding_size, :]
				INDArray v_neg_e2 = Nd4j.zeros(embeddingSize, tripplesOfRelationR.size());
				for (int i = 0; i < embeddingSize; i++) {		
					if(i<(embeddingSize)){
						v_pos_e1.put(i, v_pos.getRow(i));
						v_neg_e1.put(i, v_neg.getRow(i));
					}else{
						v_pos_e2.put(i, v_pos.getRow(i));
						v_neg_e2.put(i, v_neg.getRow(i));
					}
				}
				INDArray temp1 = v_pos_e1.mul(wordvectors_for_entities1_rel);
				INDArray temp2 = v_pos_e2.mul(wordvectors_for_entities2_rel);
				INDArray temp3 = v_neg_e1.mul(wordvectors_for_entities1_neg);
				INDArray temp4 = v_neg_e2.mul(wordvectors_for_entities2_neg);
				// !!!! entity_vector_grad = entity_vector_grad.add(temp1.add(temp2).add(3).add(temp4));
				*/
	
			}
			// Normalize the gradients with the training batch size
			INDArray w_grad_for_r2 = w_grad.get(r);
			INDArray v_grad_for_r = v_grad.get(r);
			w_grad_for_r = w_grad_for_r2.div(batchSize);
			v_grad_for_r = v_grad_for_r.div(batchSize);
			INDArray b_grad_for_r = b_grad.get(r);
			b_grad_for_r = b_grad_for_r.div(batchSize);
			INDArray u_grad_for_r = u_grad.get(r);
			u_grad_for_r = u_grad_for_r.div(batchSize);
			
		}
		
		// Initialize word vector gradients as a matrix of zeros
		INDArray word_vector_grad = Nd4j.zeros(embeddingSize, numberOfEntities);
		
		// TODO Calculate word vector gradients from entity gradients
		/*int entity_len = numberOfWords;
		for (int i = 0; i < numberOfEntities; i++) {
			//TODO
		}*/
		
		// Normalize word vector gradients and cost by the training batch size
		//word_vector_grad = word_vector_grad.div(batchSize);
	
		cost = cost / batchSize;
		
		// Get unrolled gradient vector
		INDArray theta_grad = parametersToStack(w_grad, v_grad, b_grad, u_grad);
		INDArray theta = parametersToStack(w,v,b,u);
		//Add regularization term to the cost and gradient
		//cost = cost + 0.5*lamda * np.sum(theta * theta)
		cost = cost + (0.5F * (lamda * Nd4j.sum(parametersToStack(w,v,b,u).mul(parametersToStack(w,v,b,u))).getFloat(0)));
		//System.out.println("Overall Cost: "+cost);
		//theta_grad = theta_grad + lamda * theta;
		//System.out.println("theta: "+theta);
		//System.out.println("theta * lamda: "+theta.mul(lamda));
		theta_grad = theta_grad.add(theta.mul(lamda));
		
		//RETURN values old: cost, theta_grad
		//ArrayList<Object> cost_theta_grad = new ArrayList<>();
		//cost_theta_grad.add(cost);
		//cost_theta_grad.add(theta_grad);
		
		// IPair<Double, double[]>
		 return BasicPair.make( (double)cost, theta_grad.data().asDouble() );
	}

	@Override
	public int getDimension() {
		return dimension_for_minimizer;
	}
	
	private INDArray parametersToStack(HashMap<Integer, INDArray> w, HashMap<Integer, INDArray> v, HashMap<Integer, INDArray> b, HashMap<Integer, INDArray> u){
		// NOTE: flatten doesnt work as numpy or matlab !!!!
		
		// Initialize the 'theta' vector and 'decode_info' for the network configuration
		INDArray theta_return = Nd4j.zeros(0,0);
		ArrayList theta = new ArrayList<INDArray>();
		
		ArrayList decode_info = new ArrayList<HashMap>();
		HashMap decode_cell = new HashMap<Integer, Integer[]>();
		
		//w:
		for (int j = 0; j < w.size(); j++) {
			//Store the configuration and concatenate to the unrolled vector
			decode_cell.put(j, w.get(j).shape());
			theta_return = Nd4j.concat(0, Nd4j.toFlattened(theta_return), Nd4j.toFlattened(w.get(j)) );
		}
		//arrayInfo(theta_return, "after w theta_return");
		
		//Store the configuration dictionary of the argument
		decode_info.add(decode_cell);
		decode_cell.clear();
		//v:
		for (int j = 0; j < v.size(); j++) {
			//Store the configuration and concatenate to the unrolled vector
			decode_cell.put(j, v.get(j).shape());
			theta_return = Nd4j.concat(0, Nd4j.toFlattened(theta_return), Nd4j.toFlattened(v.get(j)) );
		}
		//arrayInfo(theta_return, "after v theta_return");
		
		//Store the configuration dictionary of the argument
		decode_info.add(decode_cell);
		decode_cell.clear();
		
		//b:
		for (int j = 0; j < b.size(); j++) {
			//Store the configuration and concatenate to the unrolled vector
			decode_cell.put(j, b.get(j).shape());
			theta_return = Nd4j.concat(0, Nd4j.toFlattened(theta_return), Nd4j.toFlattened(b.get(j)) );
		}
		//arrayInfo(theta_return, "after b theta_return");
		
		//Store the configuration dictionary of the argument
		decode_info.add(decode_cell);
		decode_cell.clear();
		
		//U:
		for (int j = 0; j < u.size(); j++) {
			//Store the configuration and concatenate to the unrolled vector
			decode_cell.put(j, u.get(j).shape());
			theta_return = Nd4j.concat(0, Nd4j.toFlattened(theta_return), Nd4j.toFlattened(u.get(j)) );
		}
		//arrayInfo(theta_return, "after u theta_return");
		
		//Store the configuration dictionary of the argument
		decode_info.add(decode_cell);
		decode_cell.clear();
				
		//TODO for word embeddings
		/*for (int j = 0; j < vec.vocab().numWords(); j++) {
			//Store the configuration and concatenate to the unrolled vector
			decode_cell.put(j, vec.getWordVectorMatrix(vec.vocab().wordAtIndex(j)).shape());
			theta.add(Nd4j.concat(theta.size(), Nd4j.toFlattened(vec.getWordVectorMatrix(vec.vocab().wordAtIndex(j)))));
		}
		//Store the configuration dictionary of the argument
		decode_info.add(decode_cell);
		decode_cell.clear();
		*/
		
		// return theta, decode_info
		ArrayList returnparameters = new ArrayList<ArrayList>();
		returnparameters.add(theta);
		returnparameters.add(decode_info);
		
		
		return theta_return;		
		
	}
	
	private void stackToParameters(INDArray theta){
		//Read the configuration from concatenate flattened vector to the specific paramters: w,v,b,u,...
		int readposition = 0;
		int w_size = embeddingSize*embeddingSize*sliceSize; // number of values for paramter w for one relation
		int v_size = 2* embeddingSize * sliceSize;
		int b_size = 1* sliceSize;
		int u_size = sliceSize*1;
		
		//load w:
		for (int r = 0; r < w.size(); r++) {		
			// decode 100 * 100 * 3 = w_size
			// r represents a relation, i input value into the paramter between 0 and size
			// read value caluclated by readposition			
			for (int i = 0; i < w_size; i++) {
				w.get(r).put(i, theta.getScalar(readposition++));
			}
			//System.out.println("ParamToStackMethod: r: "+r+" w.size:"+w.size());
		}
		//load v:
		for (int r = 0; r < v.size(); r++) {
			for (int i = 0; i < v_size; i++) {
				v.get(r).put(i, theta.getScalar(readposition++));
			}
		}
		//load b:
		for (int r = 0; r < b.size(); r++) {
			for (int i = 0; i < b_size; i++) {
				b.get(r).put(i, theta.getScalar(readposition++));
			}
		}
		//load u:
		for (int r = 0; r < u.size(); r++) {
			for (int i = 0; i < u_size; i++) {
				u.get(r).put(i, theta.getScalar(readposition++));
			}
		}
		//TODO for word embeddings
	}
	private static double[] convertFlattenedINDArrayToDoubleArray(INDArray arr){
		double[] doublearray = arr.data().asDouble();
		return doublearray;
	}
	private void createEntityVectors(){
		//TODO 
	}
	
	public TrainingBatchJobDataFactory getDatafactory() {
		return tbj;
	}

	public void connectDatafactory(TrainingBatchJobDataFactory tbj) {
		this.tbj = tbj;
	}

	public HashMap<Integer, String> getEntitiesNumWord_global() {
		return entitiesNumWord_global;
	}

	public void setEntitiesNumWord_global(
			HashMap<Integer, String> entitiesNumWord_global) {
		this.entitiesNumWord_global = entitiesNumWord_global;
	}

	public HashMap<String, INDArray> getWorvectorWordVec_global() {
		return worvectorWordVec_global;
	}

	public void setWorvectorWordVec_global(
			HashMap<String, INDArray> worvectorWordVec_global) {
		this.worvectorWordVec_global = worvectorWordVec_global;
	}
	private static INDArray convertDoubleArrayToFlattenedINDArray(double[] dArray){
		INDArray d = Nd4j.zeros(dArray.length);
		for (int i = 0; i < dArray.length; i++) {
			d.putScalar(i, dArray[i]);
		}
		return d;
	}
	public INDArray computeBestThresholds(INDArray _theta, ArrayList<Tripple> _devTrippels){
		//load paramter w,v,b,u, (wordvectors, not implemented now)
		stackToParameters(_theta);
		
		// create entity vectors from word vectors
		INDArray entity_vectors = Nd4j.zeros(embeddingSize, numberOfEntities);
		entity_vectors= createVectorsForEachEntityByWordVectors(entity_vectors);
		//arrayInfo(entity_vectors, "entity_vectors");
		
		INDArray dev_scores = Nd4j.zeros(_devTrippels.size());
		INDArray entityVector1 = Nd4j.zeros(embeddingSize);
		INDArray entityVector2 = Nd4j.zeros(embeddingSize);
		
		for (int i = 0; i < _devTrippels.size(); i++) {
			//Get entity 1 and 2 for examples of ith relation
			//TODO using method calculateScoreOfaTripple() for getting the score
			Tripple tripple = _devTrippels.get(i);
			entityVector1 = entity_vectors.getColumn(tripple.getIndex_entity1());
			entityVector2 = entity_vectors.getColumn(tripple.getIndex_entity1());
			int rel = tripple.getIndex_relation();
			
			// TODO concat instate vstack used, because Issue #87 in nd4j
			//INDArray entity_stack = Nd4j.vstack(entityVector1,entityVector2);
			INDArray entity_stack = Nd4j.concat(0,entityVector1,entityVector2);
			
			// Calculate the prdediction score for the ith example
			INDArray devscore_temp = Nd4j.zeros(1,1);
			
			for (int slice = 0; slice < sliceSize; slice++) {
				INDArray dotproduct1 = (getSliceOfaTensor(w.get(rel),slice)).mmul(entityVector2);
				INDArray dotproduct2 = entityVector1.transpose().mmul(dotproduct1);				
				INDArray dotproduct3 =  v.get(rel).getColumn(slice).transpose().mmul(entity_stack);
				INDArray score = (u.get(rel).getRow(slice)).mul(dotproduct2).add(dotproduct3).add(b.get(rel).getColumn(slice));
				devscore_temp = devscore_temp.add(score);
			}
			dev_scores.put(i, devscore_temp);
			tripple.setScore(devscore_temp.getDouble(0));
			//System.out.println(tripple + " | score: "+tripple.getScore());
		}
		
		// Maximum and Minimum of the soces
		INDArray score_min = Nd4j.min(dev_scores);
		INDArray score_max = Nd4j.max(dev_scores);
		System.out.println("score min: "+score_min);
		System.out.println("score max: "+score_max);
		
		// Initialize thereshold and accuracies
		INDArray best_theresholds = Nd4j.zeros(numberOfRelations,1);
		INDArray best_accuracies = Nd4j.zeros(numberOfRelations,1);
		
		for (int i = 0; i < numberOfRelations; i++) {
			best_theresholds.put(i, score_min);
			best_theresholds.putScalar(i, -1);
		}
		
		double score_temp = score_min.getDouble(0); // contains the value of the score that classifies a tripple as correct or incorrect
		double interval = 0.01; // the value that updates the score_temp to find a better thereshold for classification of correct or in correct
		
		//Check for the best accuracy at intervals betweeen 'score_min' and 'score_max'
		
		while (score_temp <= score_max.getDouble(0)) {
			//Check accuracy for the ith relation
			for (int i = 0; i < numberOfRelations; i++) {			
				ArrayList<Tripple> tripplesOfThisRelation = tbj.getTripplesOfRelation(i, _devTrippels);
				double temp_accuracy=0;
				
				//compare the score of each tripple with the label
				for (int j = 0; j < tripplesOfThisRelation.size(); j++) {
					//double scoreOfThisTripple = tripplesOfThisRelation.get(j).getScore();
					//System.out.println("if: "+tripplesOfThisRelation.get(j).getScore()+" <= "+score_temp);
					if (tripplesOfThisRelation.get(j).getScore() <= score_temp) {
						
						//scoreOfThisTripple = 1;	//classification of this tripple as correct	
						if (tripplesOfThisRelation.get(j).getLabel() == 1) {
							temp_accuracy = temp_accuracy +1;
						}
					}else{
						//scoreOfThisTripple = -1; //classification of this tripple as incorrect
						if (tripplesOfThisRelation.get(j).getLabel() == -1) {
							temp_accuracy = temp_accuracy +1;
						}
					}
					//if (scoreOfThisTripple == tripplesOfThisRelation.get(j).getLabel()) {
					//	temp_accuracy = temp_accuracy +1;
					//} 	
				}
				//System.out.println("temp_accuracy: "+temp_accuracy);
				temp_accuracy = temp_accuracy / tripplesOfThisRelation.size(); //current accuracy of prediction for this relation
				//System.out.println("temp_accuracy for "+i+" relation: "+temp_accuracy);
				//If the accuracy is better, update the threshold and accuracy values
				if (temp_accuracy > best_accuracies.getDouble(i)) {
					best_accuracies.putScalar(i, temp_accuracy);
					best_theresholds.putScalar(i, score_temp);
				}
				score_temp = score_temp + interval;
			}
		}
		//return the best theresholds for the prediction
		return best_theresholds;
		
		
	}
	
	private static INDArray createVectorsForEachEntityByWordVectors(INDArray entity_vectors){

		for (int i = 0; i < entitiesNumWord_global.size(); i++) {
			String entity_name;
			try {
				entity_name = entitiesNumWord_global.get(i).substring(2, entitiesNumWord_global.get(i).lastIndexOf("_"));
			} catch (Exception e) {
				entity_name =entitiesNumWord_global.get(i).substring(2);			
			}
			//System.out.println("entity: "+entitiesNumWord.get(i)+" | word: "+entity_name);			
			if (entity_name.contains("_")) { //whitespaces are _
				INDArray entityvector = Nd4j.zeros(100, 1);
				for (int j = 0; j <entitiesNumWord_global.get(i).split("_").length; j++) {
					try {
						entityvector = entityvector.add(worvectorWordVec_global.get(entity_name.split("_")[j]));
					} catch (Exception e) {
						entityvector = entityvector.add(worvectorWordVec_global.get("unknown"));
					}			
				}
				entityvector = entityvector.div(entity_name.split("_").length);
				entity_vectors.putColumn(i, entityvector);
			}else{
				//if no word vector available, use "unknown" word vector
				try {
					entity_vectors.putColumn(i, worvectorWordVec_global.get(entity_name));
				} catch (Exception e) {
					entity_vectors.putColumn(i, worvectorWordVec_global.get("unknown"));
				}			
			}		
		}
		return entity_vectors;
	}

	public INDArray getPrediction(INDArray _theta, ArrayList<Tripple> _testTripples, INDArray _bestThresholds){
		// load paramter w,v,b,u, (wordvectors, not implemented now)
		stackToParameters(_theta);
		
		// create entity vectors from word vectors
		INDArray entity_vectors = Nd4j.zeros(embeddingSize, numberOfEntities);
		entity_vectors= createVectorsForEachEntityByWordVectors(entity_vectors);
		
		// initialize array to store the predictions of in- and correct tripples of the test data
		INDArray predictions = Nd4j.zeros(_testTripples.size());
		INDArray accuracy = Nd4j.zeros(_testTripples.size()); // if predcition == lable -> 1 else 0
		System.out.println("_testTripples.size(): "+_testTripples.size());
		for (int i = 0; i < _testTripples.size(); i++) {
			double score = calculateScoreOfaTripple(_testTripples.get(i), entity_vectors);
			//System.out.println("get score:" + _testTripples.get(i).getScore());	
			// calculate prediction based on previously calculate thersholds
			if(score<= _bestThresholds.getDouble(_testTripples.get(i).getIndex_relation())){
				//tripple is predicted as correct
				_testTripples.get(i).setPrediction(1);
				predictions.putScalar(i, 1);
				//compare tripple prediction with label
				if (_testTripples.get(i).getLabel()==1) {
					accuracy.putScalar(i, 1);
				}else{
					accuracy.putScalar(i, 0);
				}
			}else{
				//tripple is predicted as incorrect
				_testTripples.get(i).setPrediction(-1);
				predictions.putScalar(i, -1);
				//compare tripple prediction with label
				if (_testTripples.get(i).getLabel()==-1) {
					accuracy.putScalar(i, 1);
				}else{
					accuracy.putScalar(i, 0);
				}
			}
				
		}
		System.out.println("Accuracy of predictions: " + Nd4j.mean(accuracy));
		return predictions; //array with predictions
		
	}
	private double calculateScoreOfaTripple(Tripple _tripple, INDArray _entity_vectors){
		//Get entity 1 and 2 for examples of ith relation
		INDArray entityVector1 = _entity_vectors.getColumn(_tripple.getIndex_entity1());
		INDArray entityVector2 = _entity_vectors.getColumn(_tripple.getIndex_entity1());
		int rel = _tripple.getIndex_relation();
		
		// TODO concat instate vstack used, because Issue #87 in nd4j
		//INDArray entity_stack = Nd4j.vstack(entityVector1,entityVector2);
		INDArray entity_stack = Nd4j.concat(0,entityVector1,entityVector2);
		
		// Calculate the prdediction score for the ith example
		double score_temp=0;
		
		for (int slice = 0; slice < sliceSize; slice++) {
			INDArray dotproduct1 = (getSliceOfaTensor(w.get(rel),slice)).mmul(entityVector2);
			INDArray dotproduct2 = entityVector1.transpose().mmul(dotproduct1);				
			INDArray dotproduct3 =  v.get(rel).getColumn(slice).transpose().mmul(entity_stack);
			INDArray score = (u.get(rel).getRow(slice)).mul(dotproduct2).add(dotproduct3).add(b.get(rel).getColumn(slice));
			score_temp = score_temp + score.getDouble(0);
		}
		_tripple.setScore(score_temp);
		//System.out.println(tripple + " | score: "+tripple.getScore());
		return score_temp;
	}
}
