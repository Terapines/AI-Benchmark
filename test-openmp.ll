; ModuleID = './test-openmp'
source_filename = "test-openmp.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@anon.fad58de7366495db4650cfefac2fcd61.0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@anon.fad58de7366495db4650cfefac2fcd61.1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 22, ptr @anon.fad58de7366495db4650cfefac2fcd61.0 }, align 8
@.str = private unnamed_addr constant [16 x i8] c"i=%d thread=%d\0A\00", align 1
@anon.fad58de7366495db4650cfefac2fcd61.2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @anon.fad58de7366495db4650cfefac2fcd61.0 }, align 8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @anon.fad58de7366495db4650cfefac2fcd61.2, i32 0, ptr @main.omp_outlined)
  ret i32 0
}

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @main.omp_outlined(ptr noalias noundef %0, ptr noalias noundef %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  store i32 0, ptr %7, align 4
  store i32 7, ptr %8, align 4
  store i32 1, ptr %9, align 4
  store i32 0, ptr %10, align 4
  %12 = load ptr, ptr %3, align 8
  %13 = load i32, ptr %12, align 4
  call void @__kmpc_for_static_init_4(ptr @anon.fad58de7366495db4650cfefac2fcd61.1, i32 %13, i32 34, ptr %10, ptr %7, ptr %8, ptr %9, i32 1, i32 1)
  %14 = load i32, ptr %8, align 4
  %15 = icmp sgt i32 %14, 7
  br i1 %15, label %16, label %17

16:                                               ; preds = %2
  br label %19

17:                                               ; preds = %2
  %18 = load i32, ptr %8, align 4
  br label %19

19:                                               ; preds = %17, %16
  %20 = phi i32 [ 7, %16 ], [ %18, %17 ]
  store i32 %20, ptr %8, align 4
  %21 = load i32, ptr %7, align 4
  store i32 %21, ptr %5, align 4
  br label %22

22:                                               ; preds = %34, %19
  %23 = load i32, ptr %5, align 4
  %24 = load i32, ptr %8, align 4
  %25 = icmp sle i32 %23, %24
  br i1 %25, label %26, label %37

26:                                               ; preds = %22
  %27 = load i32, ptr %5, align 4
  %28 = mul nsw i32 %27, 1
  %29 = add nsw i32 0, %28
  store i32 %29, ptr %11, align 4
  %30 = load i32, ptr %11, align 4
  %31 = call i32 @omp_get_thread_num()
  %32 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %30, i32 noundef %31)
  br label %33

33:                                               ; preds = %26
  br label %34

34:                                               ; preds = %33
  %35 = load i32, ptr %5, align 4
  %36 = add nsw i32 %35, 1
  store i32 %36, ptr %5, align 4
  br label %22

37:                                               ; preds = %22
  br label %38

38:                                               ; preds = %37
  call void @__kmpc_for_static_fini(ptr @anon.fad58de7366495db4650cfefac2fcd61.1, i32 %13)
  ret void
}

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) #2

declare i32 @printf(ptr noundef, ...) #3

declare i32 @omp_get_thread_num() #3

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) #2

; Function Attrs: nounwind
declare !callback !9 void @__kmpc_fork_call(ptr, i32, ptr, ...) #2

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"openmp", i32 51}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 1, !"ThinLTO", i32 0}
!7 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!8 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!9 = !{!10}
!10 = !{i64 2, i64 -1, i64 -1, i1 true}

^0 = module: (path: "./test-openmp", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "__kmpc_for_static_fini") ; guid = 1363247836708998380
^2 = gv: (name: "main.omp_outlined", summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 49, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^8), (callee: ^7), (callee: ^5), (callee: ^1)), refs: (^3, ^9)))) ; guid = 2597339879140655706
^3 = gv: (name: "anon.fad58de7366495db4650cfefac2fcd61.1", summaries: (variable: (module: ^0, flags: (linkage: private, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), varFlags: (readonly: 1, writeonly: 0, constant: 1), refs: (^10)))) ; guid = 6405650727423593244
^4 = gv: (name: "__kmpc_fork_call") ; guid = 7311423072296744854
^5 = gv: (name: "printf") ; guid = 7383291119112528047
^6 = gv: (name: "anon.fad58de7366495db4650cfefac2fcd61.2", summaries: (variable: (module: ^0, flags: (linkage: private, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), varFlags: (readonly: 1, writeonly: 0, constant: 1), refs: (^10)))) ; guid = 7702321721848097435
^7 = gv: (name: "omp_get_thread_num") ; guid = 8718096434148443913
^8 = gv: (name: "__kmpc_for_static_init_4") ; guid = 8974679430248103050
^9 = gv: (name: ".str", summaries: (variable: (module: ^0, flags: (linkage: private, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), varFlags: (readonly: 1, writeonly: 0, constant: 1)))) ; guid = 14962283754142114902
^10 = gv: (name: "anon.fad58de7366495db4650cfefac2fcd61.0", summaries: (variable: (module: ^0, flags: (linkage: private, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), varFlags: (readonly: 1, writeonly: 0, constant: 1)))) ; guid = 15606288848450720864
^11 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^4)), refs: (^6, ^2)))) ; guid = 15822663052811949562
^12 = flags: 8
^13 = blockcount: 0
