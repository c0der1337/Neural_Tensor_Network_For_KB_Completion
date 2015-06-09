

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Util {
	// contains support functions
	
	Util(){
		
	}
	
	/**
	 * Get slices of a tensor because of nd4j issue #84
	 */
	public INDArray _getSliceOfaTensor(INDArray tensor, int numOfSlice){
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
	
	public INDArray putSliceOfaTensor(INDArray tensor, int numOfSlice, INDArray slice){
		for (int c = 0; c < slice.rows(); c++) {
			for (int r = 0; r < slice.columns(); r++) {
				tensor.slice(c).put(r, numOfSlice, slice.getDouble(c,r));
			}
		}
		//System.out.println("return tensor: "+tensor);
		return tensor;

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
	
	public INDArray MatrixX_mmul_CSRMatrix(INDArray x, INDArray values, INDArray entitylist, INDArray rows, int sparseShape_rowSize, int sparseShape_columns){
		//System.out.println("x: "+x.shape()[0]+"|"+x.shape()[1]+" - values:"+values.shape()[0]+" - entitylist:"+entitylist.shape()[0]+" - rows: "+rows.shape()[0]+" m: "+sparseShape_rowSize+"n: "+sparseShape_columns);
		// x is a non sparse / dense matrix with same size of columns as the number of rows of the sparse matrix
		// values: (also called data) are the non zero values
		// entitylist is similar to indices for the columns
		// rows is the rowpointer (not correct implemented, because every row has only 1 value)
		
		//create the result array for this multiplication
		INDArray resultArr = Nd4j.zeros(x.rows(),sparseShape_columns);
		
		// multiply the dense matrix x with the sparse values stores in CSR format
		for (int rowsOfX = 0; rowsOfX < x.rows(); rowsOfX++) {
			for (int row = 0; row < rows.length()-1; row++) {
				//System.out.println("row: "+row);
				int column = entitylist.getInt(row);
				double currX = x.getDouble(rowsOfX,row);
				double currVal = values.getDouble(row);
				double z = currX  * currVal;
				z = z + resultArr.getDouble(rowsOfX,column);
				resultArr.put(rowsOfX,column, z);
				//System.out.println("Sparse values: row:" + rowsOfX+"column: "+column);
			}
		}
		return resultArr;
	}

	public void multCSRMAtrixWithVector2(INDArray x, INDArray values, INDArray entitylist, INDArray rows, int rowSize, int columns){
		System.out.println("x: "+ x);
		System.out.println("values: "+ values);
		System.out.println("entitylist: "+ entitylist);
		System.out.println("rows: "+rows);
		System.out.println("Dense: " +getDenseMatrixWithSparseMatrixCRSData( values, entitylist, rows, rowSize, columns));
		System.out.println("correct: "+x.mul(getDenseMatrixWithSparseMatrixCRSData( values, entitylist, rows, rowSize, columns)));
		//System.out.println("correct: "+getDenseMatrixWithSparseMatrixCRSData( values, entitylist, rows, rowSize, columns).mul(x));
		System.out.println("Multiplications starts for VectorX mul CSRData...");
		INDArray resultArr = Nd4j.zeros(1,columns);
		for (int row = 0; row < rows.length(); row++) {
			System.out.println("resultArr: "+resultArr);
			int column = entitylist.getInt(row);
			double currX = x.getDouble(row);
			double currVal = values.getDouble(row);
			double z = currX  * currVal;
			 z = z + resultArr.getDouble(column);
			resultArr.putScalar(column, z);
		}
		System.out.println("OUTPUT resultArr: "+resultArr);
		System.out.println("Multiplications ends...");
		System.out.println("+++++++++++++++++++++++++++");
		System.out.println("Multiplications starts for VectorX mul CSRData...");
		x = Nd4j.rand(2,3);
		System.out.println("x: "+ x);
		INDArray resultArr2 = Nd4j.zeros(2,columns);
		for (int rowsOfX = 0; rowsOfX < x.rows(); rowsOfX++) {
			for (int row = 0; row < rows.length(); row++) {
				System.out.println("resultArr: "+resultArr2);
				int column = entitylist.getInt(row);
				double currX = x.getDouble(rowsOfX,row);
				double currVal = values.getDouble(row);
				double z = currX  * currVal;
				z = z + resultArr2.getDouble(rowsOfX,column);
				resultArr2.put(rowsOfX,column, z);
			}
		}
		System.out.println("OUTPUT resultArr2: "+resultArr2);
		System.out.println("Multiplications ends...");
		
		for (int i = 0; i < x.length(); i++) {
			for (int k = rows.getInt(i); k < rows.getInt(i+1); k++) {
				double valueK = values.getDouble(k);
				int columnK = entitylist.getInt(k);
				System.out.println("valueK: "+valueK+"|entityListK: "+columnK);
				//double entityListK = entitylist.getInt(k);
				double res = valueK * x.getDouble(columnK);
				resultArr.putScalar(i, res + resultArr.getDouble(i));
				System.out.println("resultArr: "+resultArr);
			}
		}
	}
	
	public INDArray transpose(INDArray x){
		// Method was necessary because mmul with a tranposed matrix occured in a false result because of an issue in the transpose method, see ND4j issue 109 for instance
		INDArray tranposed = Nd4j.create(x.columns(),x.rows());
		
		for (int i = 0; i < x.columns(); i++) {
			tranposed.putRow(i, x.transpose().getRow(i));
		}
		
		
		return tranposed;
	}
	
	
}
