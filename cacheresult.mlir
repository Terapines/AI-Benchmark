module {
  func.func @softmax_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.muli %arg8, %arg2 : i32
    %1 = arith.muli %arg8, %arg3 : i32
    // Convert arg4 from i32 to index for scf.parallel
    %arg4_index = arith.index_cast %arg4 : i32 to index

    // Convert max reduction loop to scf.parallel with reduction
    %2 = scf.parallel (%arg11) = (%c0) to (%arg4_index) step (%c1) init (%cst) -> f32 {
      // Convert index induction variable to i32 for arithmetic operations
      %arg11_i32 = arith.index_cast %arg11 : index to i32
      %4 = arith.addi %0, %arg11_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
      %6 = affine.load %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
      scf.reduce(%6 : f32) {
      ^bb0(%lhs: f32, %rhs: f32):
        %max_val = arith.maxnumf %lhs, %rhs : f32
        scf.reduce.return %max_val : f32
      }
    }

    %3 = scf.for %arg11 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg12 = %cst_0) -> (f32)  : i32 {
      %4 = arith.addi %0, %arg11 : i32
      %5 = arith.index_cast %4 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
      %6 = affine.load %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
      %7 = arith.subf %6, %2 : f32
      %8 = math.exp %7 : f32
      %9 = arith.addf %arg12, %8 : f32
      %10 = arith.addi %1, %arg11 : i32
      %11 = arith.index_cast %10 : i32 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
      memref.store %8, %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1], offset: ?>>
      scf.yield %9 : f32
    }

    // Convert normalization loop to scf.parallel (no reduction needed)
    scf.parallel (%arg11) = (%c0) to (%arg4_index) step (%c1) {
      // Convert index induction variable to i32 for arithmetic operations
      %arg11_i32 = arith.index_cast %arg11 : index to i32
      %4 = arith.addi %1, %arg11_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
      %6 = affine.load %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
      %7 = arith.divf %6, %3 : f32
      memref.store %7, %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>>
    }
    return
  }
}
