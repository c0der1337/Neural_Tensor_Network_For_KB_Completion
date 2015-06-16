import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Original from Standford Core NLP, modified to ND4j and my purpose with only a small number of functions.
 * These functions are befored implemented in the Util.java Class but seems better to handle a tensor as additional class.
 * This class defines a block tensor, somewhat like a three
 * dimensional matrix.  This can be created in various ways, such as
 * by providing an array of SimpleMatrix slices, by providing the
 * initial size to create a 0-initialized tensor, or by creating a
 * random matrix.
 *
 * @author John Bauer
 * @author Richard Socher
 */
public class Tensor{
  private final INDArray[] slices;

  final int numRows;
  final int numCols;
  final int numSlices;

  /**
   * Creates a zero initialized tensor
   */
  public Tensor(int numRows, int numCols, int numSlices, int filling) {
    slices = new INDArray[numSlices];
    if (filling == 0) {
        for (int i = 0; i < numSlices; ++i) {
            slices[i] =  Nd4j.zeros(numRows, numCols);
          }
	}else if (filling == 1) {
        for (int i = 0; i < numSlices; ++i) {
            slices[i] =  Nd4j.ones(numRows, numCols);
          }
	}else if(filling== 2){
	    for (int i = 0; i < numSlices; ++i) {
	        slices[i] = Nd4j.rand(numRows,numCols);
	     }
	}

    this.numRows = numRows;
    this.numCols = numCols;
    this.numSlices = numSlices;
  }

  /**
   * Copies the data in the slices.  Slices are copied rather than
   * reusing the original SimpleMatrix objects.  Each slice must be
   * the same size.
   */
  public Tensor(INDArray[] slices) {
	System.out.println("NOT WORKING!");
    this.numRows = slices[0].rows();
    this.numCols = slices[0].columns();
    this.numSlices = slices.length;
    this.slices = new INDArray[slices.length];
    for (int i = 0; i < numSlices; ++i) {
      if (slices[i].rows() != numRows || slices[i].columns() != numCols) {
        throw new IllegalArgumentException("Slice " + i + " has matrix dimensions " + slices[i].rows() + "," + slices[i].columns() + ", expected " + numRows + "," + numCols);
      }
      //this.slices[i] = new SimpleMatrix(slices[i]);
    }
    
  }

  /**
   * Returns a randomly initialized tensor with values draft from the
   * uniform distribution between minValue and maxValue.
   */
  public static Tensor random(int numRows, int numCols, int numSlices) {
    Tensor tensor = new Tensor(numRows, numCols, numSlices, 0);
    for (int i = 0; i < numSlices; ++i) {
      tensor.slices[i] = Nd4j.rand(numRows,numCols);
    }
    return tensor;
  }

  /**
   * Number of rows in the tensor
   */
  public int numRows() {
    return numRows;
  }

  /**
   * Number of columns in the tensor
   */
  public int numCols() {
    return numCols;
  }

  /**
   * Number of slices in the tensor
   */
  public int numSlices() {
    return numSlices;
  }

  /**
   * Total number of elements in the tensor
   */
  public int getNumElements() {
    return numRows * numCols * numSlices;
  }

  public void put(int pos, double value) {
    for (int slice = 0; slice < numSlices; ++slice) {
      slices[slice].putScalar(pos, value);
    }
  }
  
  public void put(int row, int column, double value) {
	    for (int slice = 0; slice < numSlices; ++slice) {
	      slices[slice].put(row, column, value);
	    }
  }
  public void put(int slice, int row, int column, double value) {
	      slices[slice].put(row, column, value);
  }


  /**
   * Use the given <code>matrix</code> in place of <code>slice</code>.
   * Does not copy the <code>matrix</code>, but rather uses the actual object.
   */
  public void setSlice(int slice, INDArray matrix) {
    if (slice < 0 || slice >= numSlices) {
      throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
    }
    if (matrix.columns() != numCols) {
      throw new IllegalArgumentException("Incompatible matrix size.  Has " + matrix.columns() + " columns, tensor has " + numCols);
    }
    if (matrix.rows() != numRows) {
      throw new IllegalArgumentException("Incompatible matrix size.  Has " + matrix.rows() + " columns, tensor has " + numRows);
    }
    slices[slice] = matrix;
  }

  /**
   * Returns the SimpleMatrix at <code>slice</code>.
   * <br>
   * The actual slice is returned - do not alter this unless you know what you are doing.
   */
  public INDArray getSlice(int slice) {
    if (slice < 0 || slice >= numSlices) {
      throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
    }
    return slices[slice];
  }

  

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    for (int slice = 0; slice < numSlices; ++slice) {
      result.append("Slice " + slice + "\n");
      result.append(slices[slice]);
    }
    return result.toString();
  }
}