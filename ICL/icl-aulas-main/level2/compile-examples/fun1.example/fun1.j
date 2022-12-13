.class public fun1
.super java/lang/Object
.method public <init>()V
	aload_0
	invokenonvirtual java/lang/Object/<init>()V
	return
.end method
.method public static main([Ljava/lang/String;)V
	; set limits used by this method
	.limit locals 10
	.limit stack 256
	; initialize SL variable to null
	aconst_null
	astore_3
	; ---- END OF INTRO CODE
	new frame_0
	dup
	invokespecial frame_0/<init>()V
	dup
	aload_3
	putfield frame_0/sl Ljava/lang/Object;
	dup
	astore_3
	dup
	new closure_0
	dup
	invokespecial closure_0/<init>()V
	dup
	aload_3
	putfield closure_0/sl Lframe_0;
	putfield frame_0/v0 Ljava/lang/Object;
	pop
	aload_3
	getfield frame_0/v0 Ljava/lang/Object;
	checkcast closure_interface_int_int
	sipush 2
	invokeinterface closure_interface_int_int/call(I)I 2
	aload_3
	getfield frame_0/v0 Ljava/lang/Object;
	checkcast closure_interface_int_int
	sipush 3
	invokeinterface closure_interface_int_int/call(I)I 2
	iadd
	dup
	getstatic java/lang/System/out Ljava/io/PrintStream;
	swap
	invokevirtual java/io/PrintStream/println(I)V
	aload_3
	getfield frame_0/sl Ljava/lang/Object;
	astore_3
	; ---- START OF EPILOGUE CODE
	return
.end method
