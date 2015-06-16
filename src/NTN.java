

import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import edu.stanford.nlp.ling.tokensregex.types.AssignableExpression;
import edu.stanford.nlp.optimization.DiffFunction;
import edu.umass.nlp.optimize.IDifferentiableFn;
import edu.umass.nlp.utils.BasicPair;
import edu.umass.nlp.utils.IPair;

public class NTN implements IDifferentiableFn, DiffFunction {
	
	int embeddingSize; 			// 100 size of a single word vector
	int numberOfEntities;
	int numberOfRelations;
	int numOfWords; 			// number of word vectors
	int batchSize; 				// training batch size
	int sliceSize;				// 3 number of slices in tensor
	float lamda;				// regulariization parameter
	INDArray theta_inital; 		// stacked paramters after initlization of NTN
	int dimension_for_minimizer;// dimensions / size of theta for the minimizer
	String activationFunc;		// "tanh" or "sigmoid"
   
	//Network Parameters - Integer identifies the relation, there are different parameters for each relation
	HashMap<Integer, Tensor> w;
	HashMap<Integer, INDArray> v;
	HashMap<Integer, INDArray> b;
	HashMap<Integer, INDArray> u;
	INDArray wordvectors;
	
	int update;							
	DataFactory tbj;

	NTN(	int embeddingSize, 			// 100 size of a single word vector
			int numberOfEntities,		
			int numberOfRelations,		// number of different relations
			int numberOfWords,			// number of word vectors
			int batchSize, 				// training batch size original: 20.000
			int sliceSize, 				// 3 number of slices in tensor
			String activation_function,	// 0 - tanh, 1 - sigmoid
			DataFactory tbj,			// data management unit
			float lamda) throws IOException{  				// regulariization parameter
		
		// Initialize the parameters of the network
		w = new HashMap<Integer, Tensor>();
		v = new HashMap<Integer, INDArray>();
		b = new HashMap<Integer, INDArray>();
		u = new HashMap<Integer, INDArray>();
		
		this.embeddingSize = embeddingSize;
		this.numberOfEntities = numberOfEntities;
		this.numberOfRelations = numberOfRelations;
		this.batchSize = batchSize;
		this.sliceSize = sliceSize;
		this.lamda = lamda;
		this.numOfWords = numberOfWords;
		this.activationFunc = activation_function;
		update = 0;
		double r = 1 / Math.sqrt(2*embeddingSize); // r is used for a better initialization of w
		
		for (int i = 0; i < numberOfRelations; i++) {
			w.put(i, new Tensor(100, 100, 3, 2));
			v.put(i, Nd4j.zeros(2*embeddingSize,sliceSize));
			b.put(i, Nd4j.zeros(1,sliceSize));
			u.put(i, Nd4j.ones(sliceSize,1));				
		}
		
		// Initialize WordVectors via loaded Vectors
		wordvectors = tbj.getWordVectorMaxtrixLoaded();
		// For random wordVector initialization:
		//wordvectors = Nd4j.rand(embeddingSize,numberOfWords).muli(0.0001);
		
		// Unroll the parameters into a vector
		theta_inital = parametersToStack(w, v, b, u, wordvectors);
		dimension_for_minimizer = theta_inital.data().asDouble().length;
	}
	
	/**
	 * Returns the cost/loss for the current paramters and return optimized paramters 
	 * @param  _theta  	an double array with the flattened paramters of the network
	 * @return      	an IPair that contains of cost and optimized network paramters
	 * @see         ...
	 */
	@Override
	public IPair<Double, double[]> computeAt(double[] _theta) {
		// Load input paramter(theta) into corresponding INDArray variables for loss / cost computation
		String started = ""+new Date().toString();
		System.out.println("time 1: "+new Date().toString());
		stackToParameters(new Util().convertDoubleArrayToFlattenedINDArray(_theta));

		// Initialize entity vectors and their gradient as matrix of zeros
		INDArray entity_vectors = Nd4j.zeros(embeddingSize, numberOfEntities);
		INDArray entity_vectors_grad = Nd4j.zeros(embeddingSize, numberOfEntities);
		
		//Assign entity vectors to be the mean of word vectors involved		
		//entity_vectors= tbj.createVectorsForEachEntityByWordVectors(wordvectors);
		entity_vectors = tbj.createVectorsForEachEntityByWordVectorsWithPreloadIdices(wordvectors);
		
		// Initialize cost as zero
		Float cost = 0F;
		
		// Use hashmaps to store parameter gradients for each relation
		HashMap<Integer, Tensor> w_grad = new HashMap<Integer, Tensor>();
		HashMap<Integer, INDArray> v_grad = new HashMap<Integer, INDArray>();
		HashMap<Integer, INDArray> u_grad = new HashMap<Integer, INDArray>();
		HashMap<Integer, INDArray> b_grad = new HashMap<Integer, INDArray>();
		
		for (int r = 0; r < numberOfRelations; r++) {
			System.out.println("time 2: "+new Date().toString());
			// Get a list of examples / tripples for the ith relation
			ArrayList<Tripple> tripplesOfRelationR = tbj.getBatchJobTripplesOfRelation(r);
		
			System.out.println(tripplesOfRelationR.size()+" Trainingsexample for relation r="+r+"of "+numberOfRelations);
			//Are there training tripples availabe (if the batchjob size is very small, not for each realation is tripple available
			if (tripplesOfRelationR.size() != 0) {			
				// Initlize entity and rel index lists
				INDArray e1 = tbj.getEntitiy1IndexNumbers(tripplesOfRelationR);
				INDArray e2 = tbj.getEntitiy2IndexNumbers(tripplesOfRelationR);
				INDArray rel = tbj.getRelIndexNumbers(tripplesOfRelationR);
				INDArray e3 = tbj.getEntitiy3IndexNumbers(tripplesOfRelationR);
				
				// Initilize entity vector lists with zeros
				INDArray entityVectors_e1 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
				INDArray entityVectors_e2 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
				INDArray entityVectors_e3 = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});		
				INDArray entityVectors_e1_neg = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
				INDArray entityVectors_e2_neg = Nd4j.zeros(new int[]{embeddingSize,tripplesOfRelationR.size()});
				INDArray e1_neg = Nd4j.zeros(e1.shape());
				INDArray e2_neg = Nd4j.zeros(e1.shape());
				
				// Get only entity vectors of training examples of the this / rth relation
				for (int j = 0; j < tripplesOfRelationR.size(); j++) {
					Tripple tripple = tripplesOfRelationR.get(j);
					entityVectors_e1.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity1()));
					entityVectors_e2.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity2()));
					entityVectors_e3.putColumn(j, entity_vectors.getColumn(tripple.getIndex_entity3_corrupt()));
				}
				// Choose entity vectors for negative training example based on random
				if (Math.random()>0.5) {
					entityVectors_e1_neg = entityVectors_e1;
					entityVectors_e2_neg = entityVectors_e3;
					e1_neg = e1;
					e2_neg = e3;
				}else{
					entityVectors_e1_neg = entityVectors_e3;
					entityVectors_e2_neg = entityVectors_e2;
					e1_neg = e3;
					e2_neg = e2;
				}
					
				// Initialize pre-activations of the tensor network as matrix of zeros
				INDArray preactivation_pos = Nd4j.zeros(sliceSize, tripplesOfRelationR.size());
				INDArray preactivation_neg = Nd4j.zeros(sliceSize, tripplesOfRelationR.size());
				System.out.println("time 3: "+new Date().toString());
				// Add contribution of W
				//T INDArray wOfThisRelation = w.get(r);
				Tensor wOfThisRelation = w.get(r);
				for (int slice = 0; slice < sliceSize; slice++) {
					
					//T INDArray sliceOfW=wOfThisRelation.slice(slice);
					INDArray sliceOfW=wOfThisRelation.getSlice(slice);
					if (entityVectors_e1.columns()>1) {
						preactivation_pos.putRow(slice, Nd4j.sum(entityVectors_e1.mul(sliceOfW.mmul(entityVectors_e2)), 0));
						preactivation_neg.putRow(slice, Nd4j.sum(entityVectors_e1_neg.mul(sliceOfW.mmul(entityVectors_e2_neg)), 0));
					}else{
						preactivation_pos.put(slice, Nd4j.sum(entityVectors_e1.mul(sliceOfW.mmul(entityVectors_e2)), 0));
						preactivation_neg.put(slice, Nd4j.sum(entityVectors_e1_neg.mul(sliceOfW.mmul(entityVectors_e2_neg)), 0));
					}
				}

				System.out.println("time 4: "+new Date().toString());
				// Add contribution of V / W2
				INDArray vOfThisRelation_T= v.get(r).transpose();
				INDArray vstack_pos; 
				INDArray vstack_neg; 
				if (entityVectors_e1.columns()>1) {
					vstack_pos = Nd4j.vstack(entityVectors_e1, entityVectors_e2);
					vstack_neg = Nd4j.vstack(entityVectors_e1_neg, entityVectors_e2_neg);
				}else{
					//Using concat because of ND4j bug
					vstack_pos = Nd4j.concat(0,entityVectors_e1,entityVectors_e2);
					vstack_neg = Nd4j.concat(0,entityVectors_e1_neg, entityVectors_e2_neg);
				}
				//vstack = Nd4j.vstack(e, entityVectors_e2.linearView());	

				// Add contribution of bias b
				INDArray bOfThisRelation_T = b.get(r).transpose();
				
				preactivation_pos = preactivation_pos.add(vOfThisRelation_T.mmul(vstack_pos).addColumnVector(bOfThisRelation_T));
				preactivation_neg = preactivation_neg.add(vOfThisRelation_T.mmul(vstack_neg).addColumnVector(bOfThisRelation_T));
				
				// Apply the activation function
				INDArray z_activation_pos = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunc, preactivation_pos));
				INDArray z_activation_neg = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunc, preactivation_neg));	
				
				// Calculate scores for positive and negative examples
				INDArray score_pos = u.get(r).transpose().mmul(z_activation_pos);
				INDArray score_neg = u.get(r).transpose().mmul(z_activation_neg);

				System.out.println("time 5: "+new Date().toString());
				//Filter for training examples, (that already predicted correct and dont need to be in account for further optimization of the paramters)			
				// https://groups.google.com/forum/?hl=en#!topic/recursive-deep-learning/chDXI1S2RHU
				INDArray indx = Nd4j.ones(score_pos.columns(),1); // indx: currently all entries are used for further optimization [1111111]
				for (int i = 0; i < score_pos.columns(); i++) {
					// Compare of the correct Tripple (e1,e2) is scored higher +1 than the corresponding corrupt Tripple (e1,e3)
					if (score_pos.getRow(0).getFloat(i)+1 > score_neg.getRow(0).getFloat(i)) {
						indx.put(i,0, 1.0);
					}else{
						indx.put(i,0, 0.0);
					}
				}
				
				//Add max-margin term to the cost
				cost = cost + Nd4j.sum(indx.mul(score_pos.sub(score_neg).add(1))).getFloat(0);
				update = update + Nd4j.sum(indx).getInt(0);
				
				//Initialize W and V gradients as matrix of zero
				//T w_grad.put(r, Nd4j.zeros(new int[]{embeddingSize,embeddingSize,sliceSize}));
				w_grad.put(r, new Tensor(embeddingSize, embeddingSize, sliceSize, 0));
				v_grad.put(r, Nd4j.zeros(2*embeddingSize,sliceSize));
				
				//Number of examples contributing to error
				int numOfWrongExamples = Nd4j.sum(indx).getInt(0);
	
				// For filtering matrixes, first get array with columns where indx is 1
				int[] columns = new int[numOfWrongExamples];
				int counter=0;
				for (int i = 0; i < indx.length(); i++) {
					if(indx.getInt(i) == 1){
						columns[counter] = i;
						counter++;
					}
				}
				
				INDArray z_activation_pos_fil = Nd4j.create(sliceSize,columns.length);
				INDArray z_activation_neg_fil = Nd4j.create(sliceSize,columns.length);
				INDArray entVecE1Rel = entityVectors_e1.getColumns(columns);
				INDArray entVecE2Rel = entityVectors_e2.getColumns(columns);
				INDArray entVecE1Rel_neg = entityVectors_e1_neg.getColumns(columns);
				INDArray entVecE2Rel_neg = entityVectors_e2_neg.getColumns(columns);
				
				// Filter entity lists using 'wrong_filter'
				INDArray e1_filtered = Nd4j.zeros(columns.length);
				INDArray e2_filtered = Nd4j.zeros(columns.length);
				INDArray rel_filtered = Nd4j.zeros(columns.length);
				INDArray e1_neg_filtered = Nd4j.zeros(columns.length);
				INDArray e2_neg_filtered = Nd4j.zeros(columns.length);

				for (int i = 0; i < columns.length; i++) { // TODO use getColumns() after Issue #89 Nd4j is fixed (higher than v0.0.3.5.5.3)
					if (sliceSize ==1) {
						z_activation_pos_fil.put(i, z_activation_pos.getScalar(columns[i]));
						z_activation_neg_fil.put(i, z_activation_neg.getScalar(columns[i]));
					}else{
						z_activation_pos_fil.putColumn(i, z_activation_pos.getColumn(columns[i]));
						z_activation_neg_fil.putColumn(i, z_activation_neg.getColumn(columns[i]));
					}
					
					e1_filtered.put(i, e1.getScalar(columns[i]));
					e2_filtered.put(i, e2.getScalar(columns[i]));
					rel_filtered.put(i, rel.getScalar(columns[i]));
					e1_neg_filtered.put(i, e1_neg.getScalar(columns[i]));
					e2_neg_filtered.put(i, e2_neg.getScalar(columns[i]));
				}
				System.out.println("time 6: "+new Date().toString());	
				// Calculate U[i] gradient
				if (tripplesOfRelationR.size()!=0 & numOfWrongExamples!=0) {
					u_grad.put(r, Nd4j.sum(z_activation_pos_fil.sub(z_activation_neg_fil),1).reshape(sliceSize,1)) ;
				}else{
					// Filling with ones if there is no training example for this relation in the training batch
					u_grad.put(r, Nd4j.zeros(sliceSize,1));
				}
					
				//Calculate U * f'(z) terms useful for other gradient calculation
				INDArray temp_pos_all = activationDifferential(z_activation_pos_fil).mulColumnVector(u.get(r));
				INDArray temp_neg_all = activationDifferential(z_activation_neg_fil).mulColumnVector(u.get(r).mul(-1));
				
				// Calculate 'b[i]' gradient
				if (tripplesOfRelationR.size()!=0 & numOfWrongExamples!=0) {
					b_grad.put(r, Nd4j.sum(temp_pos_all.add(temp_neg_all),1).reshape(1, sliceSize)) ;
				}else{
					// Filling with zeros if there is no training example for this relation in the training batch
					b_grad.put(r, Nd4j.zeros(1,sliceSize));
				}
				System.out.println("time 7: "+new Date().toString());			
				// Values necessary for further compressed sparse row computations (not implemented in ND4j, with dense matrix: bad performance)
				INDArray values = Nd4j.ones(numOfWrongExamples);
				INDArray rows = Nd4j.arange(0, numOfWrongExamples+1);	
				
				// INDArray e1_sparse = new Util().getDenseMatrixWithSparseMatrixCRSData(values,e1_filtered,rows,numOfWrongExamples,numberOfEntities);
				// INDArray e2_sparse = new Util().getDenseMatrixWithSparseMatrixCRSData(values,e2_filtered,rows,numOfWrongExamples,numberOfEntities);
				// INDArray e1_neg_sparse = new Util().getDenseMatrixWithSparseMatrixCRSData(values,e1_neg_filtered,rows,numOfWrongExamples,numberOfEntities);
				// INDArray e2_neg_sparse = new Util().getDenseMatrixWithSparseMatrixCRSData(values,e2_neg_filtered,rows,numOfWrongExamples,numberOfEntities);
				
				//Initialize w gradient for this relation
				//INDArray w_grad_for_r = Nd4j.create(sliceSize,embeddingSize,embeddingSize); //55
				//w_grad.put(r,w_grad_for_r);
				
				if (tripplesOfRelationR.size()!=0 & numOfWrongExamples!=0) {
					System.out.println("time 8: "+new Date().toString());
					for (int k = 0; k < sliceSize; k++) {				
						// U * f'(z) values corresponding to one slice
						INDArray temp_pos = temp_pos_all.getRow(k);
						INDArray temp_neg = temp_neg_all.getRow(k);
						
						//Calculate 'k'th slice of 'W[i]' gradient
						INDArray w_grad_k_slice;
						if (entVecE2Rel.columns()>1) {
							w_grad_k_slice = (entVecE1Rel.mulRowVector(temp_pos)).mmul(entVecE2Rel.transpose()).add(entVecE1Rel_neg.mulRowVector(temp_neg).mmul(entVecE2Rel_neg.transpose()));
						}else{
							w_grad_k_slice = (entVecE1Rel.mulRowVector(temp_pos)).mmul(entVecE2Rel.transpose()).add(entVecE1Rel_neg.mulRowVector(temp_neg).mmul(entVecE2Rel_neg.transpose()));
						}

						w_grad.get(r).setSlice(k, w_grad_k_slice);
						
						//Calculate 'k'th slice of V gradient				
						INDArray sum_v;
						if (temp_pos.columns()>1) {
							sum_v = Nd4j.sum((Nd4j.vstack(entVecE1Rel,entVecE2Rel).mulRowVector(temp_pos)).add((Nd4j.vstack(entVecE1Rel_neg,entVecE2Rel_neg).mulRowVector(temp_neg))),1);
						}else{
							//create vstack because of a bug when is there only one column
							//INDArray entVecE1RelVstack = Nd4j.vstack(Nd4j.zeros(entVecE1Rel.rows(),entVecE1Rel.columns()).putColumn(0, entVecE1Rel.getColumn(0)),Nd4j.zeros(entVecE2Rel.rows(),entVecE2Rel.columns()).putColumn(0, entVecE2Rel.getColumn(0)));
							//INDArray entVecE1Rel_negVstack = Nd4j.vstack(Nd4j.zeros(entVecE1Rel_neg.rows(),entVecE1Rel_neg.columns()).putColumn(0, entVecE1Rel_neg.getColumn(0)),Nd4j.zeros(entVecE2Rel_neg.rows(),entVecE2Rel_neg.columns()).putColumn(0, entVecE2Rel_neg.getColumn(0)));
							INDArray entVecE1RelVstack = Nd4j.concat(0, entVecE1Rel, entVecE2Rel);
							INDArray entVecE1Rel_negVstack = Nd4j.concat(0, entVecE1Rel_neg, entVecE2Rel_neg);
							sum_v = (entVecE1RelVstack.mulRowVector(temp_pos)).add((entVecE1Rel_negVstack).mulRowVector(temp_neg));
						}
						v_grad.get(r).putColumn(k, sum_v);
						System.out.println("time 9: "+new Date().toString());
						// Add contribution of V term in the entity vectors' gradient	
						INDArray kth_slice_of_v = v.get(r).getColumn(k); //slice is the column
						INDArray v_neg; INDArray v_pos;
						if (sliceSize==1) {
							 v_pos = kth_slice_of_v.transpose().mmul(temp_pos);
							 v_neg = kth_slice_of_v.transpose().mmul(temp_neg);
						}else{
							 v_pos = kth_slice_of_v.mmul(temp_pos);
							 v_neg = kth_slice_of_v.mmul(temp_neg);//55
						}
						System.out.println("time 10: "+new Date().toString());
						INDArray temp = v_pos.get(NDArrayIndex.interval(0,embeddingSize),NDArrayIndex.interval(0,v_pos.columns()));
						INDArray v1 = new Util().MatrixX_mmul_CSRMatrix(temp, values,e1_filtered,rows,numOfWrongExamples,numberOfEntities);
						INDArray v2 = new Util().MatrixX_mmul_CSRMatrix(v_pos.get(NDArrayIndex.interval(embeddingSize,2*embeddingSize),NDArrayIndex.interval(0,v_pos.columns())), values,e2_filtered,rows,numOfWrongExamples,numberOfEntities);
						INDArray v3 = new Util().MatrixX_mmul_CSRMatrix(v_neg.get(NDArrayIndex.interval(0,embeddingSize),NDArrayIndex.interval(0,v_pos.columns())), values,e1_neg_filtered,rows,numOfWrongExamples,numberOfEntities);
						INDArray v4 = new Util().MatrixX_mmul_CSRMatrix(v_neg.get(NDArrayIndex.interval(embeddingSize,2*embeddingSize),NDArrayIndex.interval(0,v_pos.columns())), values,e2_neg_filtered,rows,numOfWrongExamples,numberOfEntities);
						System.out.println("time 11: "+new Date().toString());
						entity_vectors_grad = entity_vectors_grad.add(v1).add(v2).add(v3).add(v4);
						
						// Add contribution of 'W[i]' term in the entity vectors' gradient
						INDArray sliceWT = w.get(r).getSlice(k).transpose();
						INDArray sliceW = w.get(r).getSlice(k);
						INDArray w1 = new Util().MatrixX_mmul_CSRMatrix(sliceW.mmul(entVecE2Rel).mulRowVector(temp_pos), values,e1_filtered,rows,numOfWrongExamples,numberOfEntities);
						INDArray w2 = new Util().MatrixX_mmul_CSRMatrix(sliceWT.mmul(entVecE1Rel).mulRowVector(temp_pos), values,e2_filtered,rows,numOfWrongExamples,numberOfEntities);
						INDArray w3 = new Util().MatrixX_mmul_CSRMatrix(sliceW.mmul(entVecE2Rel_neg).mulRowVector(temp_neg), values,e1_neg_filtered,rows,numOfWrongExamples,numberOfEntities);
						INDArray w4 = new Util().MatrixX_mmul_CSRMatrix(sliceWT.mmul(entVecE1Rel_neg).mulRowVector(temp_neg), values,e2_neg_filtered,rows,numOfWrongExamples,numberOfEntities);
						System.out.println("time 12: "+new Date().toString());
						entity_vectors_grad = entity_vectors_grad.add(w1).add(w2).add(w3).add(w4);
						System.out.println("time 13: "+new Date().toString());
					}
				}else{
					// Filling with zeros if there is no training example for this relation in the training batch
					// System.out.println("no trainingsexample, gradients are zero");			
				}
				
				// Normalize the gradients with the training batch size
				for (int i = 0; i < sliceSize; i++) {
					w_grad.get(r).getSlice(i).divi(batchSize);
				}
				v_grad.get(r).divi(batchSize);
				b_grad.get(r).divi(batchSize);
				u_grad.get(r).divi(batchSize);
			}else{
				//System.out.println("next relation, maybe there is a training tripple availabe...");		
				w_grad.put(r, new Tensor(embeddingSize,embeddingSize,sliceSize,0));
				v_grad.put(r, Nd4j.zeros(2*embeddingSize,sliceSize));
				b_grad.put(r, Nd4j.zeros(1,sliceSize));
				u_grad.put(r, Nd4j.zeros(sliceSize,1));	
			}
			
		}
		// Initialize word vector gradients as a matrix of zeros
		INDArray word_vector_grad = Nd4j.zeros(embeddingSize, numOfWords);
		
		// Calculate word vector gradients from entity gradients
		for (int i = 0; i < numberOfEntities; i++) {
			int numOfWordsInEntity = tbj.entityLength(i);
			int[] wordindexes = tbj.getWordIndexes(i);
			if (Nd4j.sum(entity_vectors_grad.getColumn(i)).getFloat(0) != 0) {
				INDArray entity_vector_grad_column = entity_vectors_grad.getColumn(i);
				// Normalize by number of words
				entity_vector_grad_column.divi(numOfWordsInEntity);
				// Add entity vector gradient into word_vector_grad.
				for (int j = 0; j < wordindexes.length; j++) {
					INDArray wvgrad_Column = word_vector_grad.getColumn(wordindexes[j]);
					word_vector_grad.putColumn(wordindexes[j], word_vector_grad.getColumn(wordindexes[j]).add(entity_vector_grad_column));
				}
				
			}
		}
		// Normalize word vector gradients and cost by the training batch size
		word_vector_grad = word_vector_grad.div(batchSize);
		cost = cost / batchSize;
	
		// Get stacked gradient vector and parameter vector
		INDArray theta_grad = parametersToStack(w_grad, v_grad, b_grad, u_grad,word_vector_grad);
		INDArray theta = parametersToStack(w,v,b,u,wordvectors);
		
		//Add regularization term to the cost and gradient	
		cost = cost + (0.5F * (lamda * Nd4j.sum(theta.mul(theta)).getFloat(0)));
		theta_grad = theta_grad.add(theta.mul(lamda));
		System.out.println("Cost: "+cost+"| Amount of Tripples updated: "+update+"| start: "+started+" |end: "+new Date().toString());
		
		/*Alternative RETURN values old: cost, theta_grad
		ArrayList<Object> cost_theta_grad = new ArrayList<>();
		cost_theta_grad.add(cost);
		cost_theta_grad.add(theta_grad);
		*/
		
		// IPair<Double, double[]>
		 return BasicPair.make( (double)cost, theta_grad.data().asDouble() );
	}

	@Override
	public int getDimension() {
		return dimension_for_minimizer;
	}
	
	private INDArray parametersToStack(HashMap<Integer, Tensor> w, HashMap<Integer, INDArray> v, HashMap<Integer, INDArray> b, HashMap<Integer, INDArray> u, INDArray wordvectors){
		// NOTE: flatten doesnt work as numpy or matlab !!!!
		// Initialize the 'theta' vector
		INDArray theta_return = w.get(0).getSlice(0).linearView();
		theta_return = Nd4j.concat(0, theta_return, w.get(0).getSlice(1).linearView() );
		theta_return = Nd4j.concat(0, theta_return, w.get(0).getSlice(2).linearView() );
		
		//w:
		for (int j = 1; j < w.size(); j++) {
			// concatenate to stack
			for (int i = 0; i < sliceSize; i++) {
				theta_return = Nd4j.concat(0, theta_return, w.get(j).getSlice(i).linearView() );
			}
		}
		
		//v:
		for (int j = 0; j < v.size(); j++) {
			// concatenate to stack
			theta_return = Nd4j.concat(0, Nd4j.toFlattened(theta_return), Nd4j.toFlattened(v.get(j)) );
		}

		//b:
		for (int j = 0; j < b.size(); j++) {
			// concatenate to stack
			theta_return = Nd4j.concat(0, Nd4j.toFlattened(theta_return), Nd4j.toFlattened(b.get(j)) );
		}

		//U:
		for (int j = 0; j < u.size(); j++) {
			// concatenate to stack
			theta_return = Nd4j.concat(0, Nd4j.toFlattened(theta_return), Nd4j.toFlattened(u.get(j)) );
		}
		
		//Word Vectors:
		theta_return = Nd4j.concat(0, Nd4j.toFlattened(theta_return), Nd4j.toFlattened(wordvectors) );	
		return theta_return;			
	}
	
	private void stackToParameters(INDArray theta){
		//System.out.println("Theta_input size"+theta.length());
		//Read the configuration from concatenate flattened vector to the specific paramters: w,v,b,u,...
		int readposition = 0;
		//int w_size = embeddingSize*embeddingSize*sliceSize; // number of values for paramter w for one relation
		int w_size = embeddingSize*embeddingSize;
		int v_size = 2* embeddingSize * sliceSize;
		int b_size = 1* sliceSize;
		int u_size = sliceSize*1;
		int wordvectors_size = embeddingSize*numOfWords;
		
		INDArray readinvector;
		
		//load w:
		for (int r = 0; r < w.size(); r++) {		
			for (int j = 0; j < sliceSize; j++) {
				for (int i = 0; i < w_size; i++) {
					w.get(r).getSlice(j).linearView().putScalar(i, theta.getDouble(readposition++));
				}
				
			}
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
		//load word vectors:
		for (int i = 0; i < wordvectors_size; i++) {
			wordvectors.put(i, theta.getScalar(readposition++));
		}
	}
	
	private void stackToParametersFromNumpy(INDArray theta){
		//System.out.println("Theta_input size"+theta.length());
		//Read the configuration from concatenate flattened vector to the specific paramters: w,v,b,u,...
		int readposition = 0;
		int w_size = embeddingSize*embeddingSize*sliceSize; // number of values for paramter w for one relation
		int v_size = 2* embeddingSize * sliceSize;
		int b_size = 1* sliceSize;
		int u_size = sliceSize*1;
		int wordvectors_size = embeddingSize*numOfWords;
		
		INDArray readinvector;
		
		//load w:
		for (int r = 0; r < w.size(); r++) {		
			// decode 100 * 100 * 3 = w_size
			// r represents a relation, i input value into the paramter between 0 and size
			// read value caluclated by readposition
			readinvector = Nd4j.create(w_size);
			readinvector.putRow(0, theta.get(NDArrayIndex.interval(readposition,readposition+w_size)).getRow(0));
			//T System.out.println("w.get(r):"+w.get(r).length());
			for (int i = 0; i < w_size; i++) {
				//w.get(r).put(i, readinvector.getScalar(readposition++));
			}
			//w.get(r).transposei();
			//System.out.println("ParamToStackMethod: r: "+r+" w.size:"+w.size());
		}
		//load v:
		
		for (int r = 0; r < v.size(); r++) {
			INDArray vNumpy = Nd4j.create(sliceSize,2*embeddingSize);
			for (int i = 0; i < v_size; i++) {
				vNumpy.put(i, theta.getScalar(readposition++));
			}
			v.put(r,vNumpy.transpose());
		}
		//load b:
		for (int r = 0; r < b.size(); r++) {
			INDArray bNumpy = Nd4j.create(sliceSize,1);
			for (int i = 0; i < b_size; i++) {
				b.get(r).put(i, theta.getScalar(readposition++));
			}
			b.put(r, bNumpy.transpose());
		}
		//load u:
		for (int r = 0; r < u.size(); r++) {
			INDArray uNumpy = Nd4j.create(1,sliceSize);
			for (int i = 0; i < u_size; i++) {
				u.get(r).put(i, theta.getScalar(readposition++));
			}
			u.put(r, uNumpy);
		}
		//load word vectors:
		for (int i = 0; i < wordvectors_size; i++) {
			INDArray wordvectors = Nd4j.create(numOfWords,embeddingSize);
			wordvectors.put(i, theta.getScalar(readposition++));
		}
		wordvectors.transposei();
	}

	public INDArray computeBestThresholds(INDArray _theta, ArrayList<Tripple> _devTrippels){
		//load paramter w,v,b,u, (wordvectors, not implemented now)
		stackToParameters(_theta);
		
		// create entity vectors from word vectors
		INDArray entity_vectors= this.getDatafactory().createVectorsForEachEntityByWordVectors();
		//arrayInfo(entity_vectors, "entity_vectors");
		
		INDArray dev_scores = Nd4j.zeros(_devTrippels.size());
		INDArray entityVector1 = Nd4j.zeros(embeddingSize);
		INDArray entityVector2 = Nd4j.zeros(embeddingSize);
		
		for (int i = 0; i < _devTrippels.size(); i++) {
			//Get entity 1 and 2 for examples of ith relation
			//TODO using method calculateScoreOfaTripple() for getting the score
			double score = calculateScoreOfaTripple(_devTrippels.get(i), entity_vectors);
			dev_scores.putScalar(i, score);
			//System.out.println(tripple.toString() + " | score: "+tripple.getScore());
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
						
						//Label of this tripple = 1;	//classification of this tripple as correct	
						if (tripplesOfThisRelation.get(j).getLabel() == 1) {
							temp_accuracy = temp_accuracy +1;
						}
					}else{
						//Label of this tripple = -1; //classification of this tripple as incorrect
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
	

	public INDArray getPrediction(INDArray _theta, ArrayList<Tripple> _testTripples, INDArray _bestThresholds){
		// load paramter w,v,b,u, (wordvectors, not implemented now)
		stackToParameters(_theta);
		
		// create entity vectors from word vectors
		 INDArray entity_vectors= this.getDatafactory().createVectorsForEachEntityByWordVectors();
		
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
		INDArray entityVector2 = _entity_vectors.getColumn(_tripple.getIndex_entity2());
		int rel = _tripple.getIndex_relation();
		
		// TODO concat instate vstack used, because Issue #87 in nd4j
		//INDArray entity_stack = Nd4j.vstack(entityVector1,entityVector2);
		INDArray entity_stack = Nd4j.concat(0,entityVector1,entityVector2);
		
		// Calculate the prdediction score for the ith example
		double score_temp=0;
		
		for (int slice = 0; slice < sliceSize; slice++) {
			INDArray dotproduct1 = (w.get(rel).getSlice(slice)).mmul(entityVector2);
			INDArray dotproduct2 = entityVector1.transpose().mmul(dotproduct1);				
			INDArray dotproduct3 =  v.get(rel).getColumn(slice).transpose().mmul(entity_stack);
			INDArray score = (u.get(rel).getRow(slice)).mul(dotproduct2).add(dotproduct3).add(b.get(rel).getColumn(slice));
			score_temp = score_temp + score.getDouble(0);
		}
		_tripple.setScore(score_temp);
		//System.out.println(_tripple.toString() + " | score: "+_tripple.getScore()+" | label: "+_tripple.getLabel());
		return score_temp;
	}
	
	
	
	public INDArray getTheta_inital() {
		return theta_inital;
	}
	
	public INDArray activationDifferential(INDArray activation){
		//for a sigmoid activation function:
		//Ableitung der sigmoid function f(z) -> f'(z) -> (z * (1 - z))
		//Ableitung der tanh function f(z) -> f'(z) -> (1-tanh²(z))	
		//System.out.println(activationFunc+ " | "+activation);
			//System.out.println("X"+));
		//System.out.println("Transforms.pow(activation, 2): "+activation.mul(activation));
			//return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunc, activation).derivative());
		if (activationFunc.equals("tanh")) {
			return Nd4j.ones(activation.rows(), activation.columns()).sub(activation.mul(activation));
		}	else {
			return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunc, activation).derivative());
		}
			
		
	}

	
	public DataFactory getDatafactory() {
		return tbj;
	}

	public void connectDatafactory(DataFactory tbj) {
		this.tbj = tbj;
	}

	@Override
	public int domainDimension() {
		return dimension_for_minimizer;
	}

	@Override
	public double valueAt(double[] theta) {
		System.out.println("+++++ ++++++++++++++++++++valueAt()");
		return computeAt(theta).getFirst();
	}

	@Override
	public double[] derivativeAt(double[] theta) {
		System.out.println("+++++ +++++++++++++++++++++derivativeAt()");
		return computeAt(theta).getSecond();
	}
}
