module {
  func.func @resize_kernel(%arg0: memref<*xi8> {tt.divisibility = 16 : i32}, %arg1: memref<*xi8> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c7_i32 = arith.constant 7 : i32
    %c1_i32 = arith.constant 1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.muli %arg3, %c2_i32 : i32
    %1 = arith.muli %arg4, %c2_i32 : i32
    %2 = arith.shli %arg8, %c6_i32 : i32
    %3 = arith.shrsi %2, %c7_i32 : i32
    %4 = arith.shli %3, %c7_i32 : i32
    %5 = arith.subi %2, %4 : i32
    %6 = arith.subi %c128_i32, %5 : i32
    %7 = arith.addi %3, %c1_i32 : i32
    %8 = arith.subi %arg3, %c1_i32 : i32
    %9 = arith.minsi %7, %8 : i32
    %10 = arith.muli %arg9, %arg3 : i32
    %11 = arith.muli %10, %arg4 : i32
    %12 = arith.muli %3, %arg4 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = arith.muli %9, %arg4 : i32
    %15 = arith.addi %11, %14 : i32
    %16 = arith.muli %arg9, %0 : i32
    %17 = arith.muli %16, %1 : i32
    %18 = arith.muli %arg8, %1 : i32
    %19 = arith.addi %17, %18 : i32
    %20 = arith.subi %arg4, %c1_i32 : i32
    // Convert upper bound from i32 to index for scf.parallel
    %upper_bound = arith.index_cast %1 : i32 to index
    // Convert scf.for to scf.parallel
    scf.parallel (%arg11) = (%c0) to (%upper_bound) step (%c1) {
      // Convert index induction variable to i32 for arithmetic operations
      %arg11_i32 = arith.index_cast %arg11 : index to i32
      %21 = arith.shli %arg11_i32, %c6_i32 : i32
      %22 = arith.shrsi %21, %c7_i32 : i32
      %23 = arith.addi %13, %22 : i32
      %24 = arith.index_cast %23 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%24], sizes: [1], strides: [1] : memref<*xi8> to memref<1xi8, strided<[1], offset: ?>>
      %25 = affine.load %reinterpret_cast[0] : memref<1xi8, strided<[1], offset: ?>>
      %26 = arith.addi %15, %22 : i32
      %27 = arith.index_cast %26 : i32 to index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1], strides: [1] : memref<*xi8> to memref<1xi8, strided<[1], offset: ?>>
      %28 = affine.load %reinterpret_cast_0[0] : memref<1xi8, strided<[1], offset: ?>>
      %29 = arith.addi %22, %c1_i32 : i32
      %30 = arith.minsi %29, %20 : i32
      %31 = arith.addi %13, %30 : i32
      %32 = arith.index_cast %31 : i32 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%32], sizes: [1], strides: [1] : memref<*xi8> to memref<1xi8, strided<[1], offset: ?>>
      %33 = affine.load %reinterpret_cast_1[0] : memref<1xi8, strided<[1], offset: ?>>
      %34 = arith.addi %15, %30 : i32
      %35 = arith.index_cast %34 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1], strides: [1] : memref<*xi8> to memref<1xi8, strided<[1], offset: ?>>
      %36 = affine.load %reinterpret_cast_2[0] : memref<1xi8, strided<[1], offset: ?>>
      %37 = arith.shli %22, %c7_i32 : i32
      %38 = arith.subi %21, %37 : i32
      %39 = arith.subi %c128_i32, %38 : i32
      %40 = arith.extsi %25 : i8 to i32
      %41 = arith.muli %40, %39 : i32
      %42 = arith.extsi %33 : i8 to i32
      %43 = arith.muli %42, %38 : i32
      %44 = arith.addi %41, %43 : i32
      %45 = arith.shrsi %44, %c7_i32 : i32
      %46 = arith.extsi %28 : i8 to i32
      %47 = arith.muli %46, %39 : i32
      %48 = arith.extsi %36 : i8 to i32
      %49 = arith.muli %48, %38 : i32
      %50 = arith.addi %47, %49 : i32
      %51 = arith.shrsi %50, %c7_i32 : i32
      %52 = arith.muli %45, %6 : i32
      %53 = arith.muli %51, %5 : i32
      %54 = arith.addi %52, %53 : i32
      %55 = arith.shrsi %54, %c7_i32 : i32
      %56 = arith.trunci %55 : i32 to i8
      %57 = arith.addi %19, %arg11_i32 : i32
      %58 = arith.index_cast %57 : i32 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%58], sizes: [1], strides: [1] : memref<*xi8> to memref<1xi8, strided<[1], offset: ?>>
      memref.store %56, %reinterpret_cast_3[%c0] : memref<1xi8, strided<[1], offset: ?>>
    }
    return
  }
}
