

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Util {
	// contains support functions
	
	Util(){
		
	}
	
	/**
	 * Get slices of a tensor because of nd4j issue #84
	 */
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

	
	public INDArray convertDoubleArrayToFlattenedINDArray(double[] dArray){
		INDArray d = Nd4j.zeros(dArray.length);
		for (int i = 0; i < dArray.length; i++) {
			d.putScalar(i, dArray[i]);
		}
		return d;
	}
	
	public void arrayInfo(INDArray array, String name){
		System.out.println("_"+name + ": rows: "+array.rows()+"|cols: "+array.columns());
	}
	
	public double[] convertFlattenedINDArrayToDoubleArray(INDArray arr){
		double[] doublearray = arr.data().asDouble();
		return doublearray;
	}
	public INDArray getDenseMatrixWithSparseMatrixCRSData(INDArray values, INDArray entitylist, INDArray rows, int rowSize, int columns){
		INDArray denseMatrix = Nd4j.zeros(rowSize,columns);
		for (int i = 0; i < entitylist.length(); i++) {
			denseMatrix.put(i, entitylist.getInt(i), Nd4j.ones(1));
		}
		//System.out.println("denseMatrix: "+denseMatrix);
		return denseMatrix;
	}
	
}
